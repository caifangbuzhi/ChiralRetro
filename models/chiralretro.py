from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import apply_edit_to_mol
from rdkit import Chem
from utils.collate_fn import get_batch_graphs
from utils.rxn_graphs import MolGraph, Vocab

from models.encoder import Global_Attention, MPNEncoder, CMPNNEncoder
from models.model_utils import (creat_edits_feats, index_select_ND,
                                unbatch_feats)
from rdkit.Chem.rdchem import ChiralType

CHIRALTAG_PARITY_DIR = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,  
}

def addH(pro_mol, reac_mol):
    pro_amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()  
        for atom in pro_mol.GetAtoms()}
    pro_idx_to_amap = {value: key for key,
        value in pro_amap_to_idx.items()}
    reac_amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()  
        for atom in reac_mol.GetAtoms()}
    reac_idx_to_amap = {value: key for key,
        value in reac_amap_to_idx.items()}
    
    p_max_amap = max([atom.GetAtomMapNum() for atom in pro_mol.GetAtoms()])
    r_max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    max_amap = max(r_max_amap, p_max_amap)
    
    H_ids = [a.GetIdx() for a in pro_mol.GetAtoms() if CHIRALTAG_PARITY_DIR[a.GetChiralTag()] != 0]
    p_max_idx = max([atom.GetIdx() for atom in pro_mol.GetAtoms()]) + 1
    r_max_idx = max([atom.GetIdx() for atom in reac_mol.GetAtoms()]) + 1
    max_amap += 1
      
    for id in H_ids:
        len1 = len(pro_mol.GetAtoms())
        len2 = len(reac_mol.GetAtoms())
        pro_mol = Chem.AddHs(pro_mol, onlyOnAtoms=[id])
        reac_mol = Chem.AddHs(reac_mol, onlyOnAtoms=[reac_amap_to_idx[pro_idx_to_amap[id]]])
        len3 = len(pro_mol.GetAtoms())
        len4 = len(reac_mol.GetAtoms())

        if len1 == len3 :
            continue
        if len2 == len4:
            mol_copy = Chem.RWMol(pro_mol)  
            mol_copy.RemoveAtom(p_max_idx) 
            pro_mol = mol_copy
            continue
        
        pro_atom = pro_mol.GetAtomWithIdx(p_max_idx)
        reac_atom = reac_mol.GetAtomWithIdx(r_max_idx)
        
        pro_atom.SetAtomMapNum(max_amap)
        reac_atom.SetAtomMapNum(max_amap)
        
        max_amap += 1
        p_max_idx += 1
        r_max_idx += 1

    return pro_mol, reac_mol


class ChiralRetro(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(ChiralRetro, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)  
        self.bond_outdim = len(bond_vocab)  
        self.device = device

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        if config['encoder'] == 'DMPNN':
            self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                    bond_fdim=config['n_bond_feat'],
                                    hidden_size=config['mpn_size'],
                                    depth=config['depth'],
                                    dropout=config['dropout_mpn'],
                                    atom_message=config['atom_message'], 
                                    device = self.device)
        elif config['encoder'] == 'CMPNN':
            self.encoder = CMPNNEncoder(config)

        self.W_vv = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)

        if config['use_attn']:   
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])

        self.atom_linear = nn.Sequential(    
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))   
        self.bond_linear = nn.Sequential(   
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))    

        self.graph_linear = nn.Sequential(  
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")
 
    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],
                            prod_scopes: Tuple[List], parity_atoms, prev_atom_hiddens: torch.Tensor = None,
                            prev_atom_scope: Tuple[List] = None) -> Tuple[torch.Tensor]:     
        """Computes the edit scores given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        prev_atom_hiddens: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)  
        parity_atoms = self.to_device(parity_atoms)
        atom_scope, bond_scope = prod_scopes

        if prev_atom_hiddens is None:    
            n_atoms = prod_tensors[0].size(0)    
            prev_atom_hiddens = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)
        if self.config['encoder'] == 'DMPNN':
            a_feats = self.encoder(prod_tensors, parity_atoms, mask=None)  
        elif self.config['encoder'] == 'CMPNN':
            a_feats = self.encoder(prod_tensors, prod_scopes)
        if self.config['use_attn']:
            feats, mask = creat_edits_feats(a_feats, atom_scope)
            attention_score, feats = self.attn(feats, mask)
            a_feats = unbatch_feats(feats, atom_scope)

        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:  
            n_atoms = a_feats.shape[0]
            new_ha = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]
            prev_atom_hiddens = new_ha

        assert a_feats.shape == prev_atom_hiddens.shape
        atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))  
        prev_atom_hiddens = atom_feats.clone()  
        prev_atom_scope = atom_scope    

        node_feats = atom_feats.clone()
        bond_starts = index_select_ND(atom_feats, index=prod_tensors[-2][:, 0])  
        bond_ends = index_select_ND(atom_feats, index=prod_tensors[-2][:, 1])
        bond_feats = torch.cat([bond_starts, bond_ends], dim=1)  

        graph_vecs = torch.stack(
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])   

        atom_outs = self.atom_linear(node_feats)     
        bond_outs = self.bond_linear(bond_feats)    
        graph_outs = self.graph_linear(graph_vecs)  
 
        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),    
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]
        for i in range(len(edit_scores)):
            has_nan = torch.isnan(edit_scores[i]).any()
            if has_nan:
                print("1")

        return edit_scores, prev_atom_hiddens, prev_atom_scope

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]]) -> Tuple[torch.Tensor]:   
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        """
        max_seq_len = len(prod_seq_inputs)
        assert len(prod_seq_inputs[0]) == 3  

        prev_atom_hiddens = None    
        prev_atom_scope = None
        seq_edit_scores = []     

        for idx in range(max_seq_len):   
            prod_tensors, prod_scopes, parity_atoms = prod_seq_inputs[idx]    
            edit_scores, prev_atom_hiddens, prev_atom_scope = self.compute_edit_scores(
                prod_tensors, prod_scopes,parity_atoms, prev_atom_hiddens, prev_atom_scope)
            seq_edit_scores.append(edit_scores)

        return seq_edit_scores

    def predict(self, react_smi, prod_smi: str, rxn_class: int = None, max_steps: int = 9):
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        done = False
        steps = 0
        edits = []
        edits_atom = []
        prev_atom_hiddens = None
        prev_atom_scope = None

        products = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(products)
        reactants = Chem.MolFromSmiles(react_smi)
        Chem.Kekulize(reactants)
        products, reactants = addH(products, reactants) 
        prod_graph = MolGraph(mol=Chem.Mol(products),
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        prod_tensors, prod_scopes, parity_atoms = get_batch_graphs(
            [prod_graph], use_rxn_class=use_rxn_class)

        while not done and steps <= max_steps:
            if prod_tensors[-1].size() == (1, 0):   
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            edit_logits, prev_atom_hiddens, prev_atom_scope = self.compute_edit_scores(
                prod_tensors, prod_scopes, parity_atoms, prev_atom_hiddens, prev_atom_scope)
            idx = torch.argmax(edit_logits[0])  
            val = edit_logits[0][idx]   

            max_bond_idx = products.GetNumBonds() * self.bond_outdim

            if idx.item() == len(edit_logits[0]) - 1:  
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            elif idx.item() < max_bond_idx:  
                bond_logits = edit_logits[0][:products.GetNumBonds(
                ) * self.bond_outdim]
                bond_logits = bond_logits.reshape(
                    products.GetNumBonds(), self.bond_outdim)   
                idx_tensor = torch.where(bond_logits == val)     
                try:
                    if all(len(indices) == 0 for indices in idx_tensor):
                        raise ValueError(f"No matching value found in bond_logits for the given val: {val}")
                except ValueError as e:
                    print(f"Error occurred: {e}")
                    print(f"edit_logits[0]: {edit_logits[0]}")
                    print(f"val: {val}")
                    print(f"bond_logits: {bond_logits}")
                    print(f"Does bond_logits contain val? {torch.any(torch.isclose(bond_logits, val, atol=1e-5))}")

                idx_tensor = [indices[-1] for indices in idx_tensor]     

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = products.GetBondWithIdx(
                    bond_idx).GetBeginAtom().GetAtomMapNum()
                a2 = products.GetBondWithIdx(
                    bond_idx).GetEndAtom().GetAtomMapNum()

                a1, a2 = sorted([a1, a2])
                edit_atom = [a1, a2]    
                edit = self.bond_vocab.get_elem(edit_idx)    
                
            else:
                atom_logits = edit_logits[0][max_bond_idx:-1]

                assert len(atom_logits) == products.GetNumAtoms() * \
                    self.atom_outdim
                atom_logits = atom_logits.reshape(
                    products.GetNumAtoms(), self.atom_outdim)
                idx_tensor = torch.where(atom_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()
                edit_atom = a1
                edit = self.atom_vocab.get_elem(edit_idx)

            try:
                products = apply_edit_to_mol(mol=Chem.Mol(
                    products), edit=edit, edit_atom=edit_atom)
                prod_graph = MolGraph(mol=Chem.Mol(
                    products),  rxn_class=rxn_class, use_rxn_class=use_rxn_class)
                prod_tensors, prod_scopes, parity_atoms = get_batch_graphs(
                    [prod_graph], use_rxn_class=use_rxn_class)

                edits.append(edit)
                edits_atom.append(edit_atom)
                steps += 1

            except:
                steps += 1
                continue

        return edits, edits_atom

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab

        return saveables
