import argparse
import copy
import os
import sys
from typing import Any, Tuple

import joblib
import torch
from rdkit import Chem

from utils.collate_fn import get_batch_graphs, prepare_edit_labels
from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)
from utils.rxn_graphs import MolGraph, RxnGraph, Vocab
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

def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:
    """ Apply edits to molecular graph """
    if edit[0] == 'Change Atom':
        edit_exe = AtomEditAction(
            edit_atom, *edit[1], action_vocab='Change Atom')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Delete Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Delete Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Change Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Change Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Add Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Add Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Attaching LG':
        edit_exe = AddGroupAction(
            edit_atom, edit[1], action_vocab='Attaching LG')
        new_mol = edit_exe.apply(mol)

    return new_mol

def pad_values(tensor):
    padded_tensor_list = []
    for value in tensor:
        padded_tensor_list.extend([value.item(), 0, 0, 0, 0, 0])  
    
    return torch.tensor(padded_tensor_list)

def pad_values_one(tensor):
    padded_tensor_list = []
    for value in tensor:
        padded_tensor_list.extend([value.item(), 1, 1, 1, 1, 1])  
    
    return torch.tensor(padded_tensor_list)

def pad_tensor(tensor, target_length):
    current_length = tensor.size(0)
    if current_length < target_length:
        padding_length = target_length - current_length
        padding_tensor = torch.zeros(padding_length)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    else:
        return tensor
    
def pad_tensor_one(tensor, target_length):
    current_length = tensor.size(0)
    if current_length < target_length:
        padding_length = target_length - current_length
        padding_tensor = torch.ones(padding_length)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    else:
        return tensor

def process_batch(batch_graphs, args): 
    lengths = torch.tensor([len(graph_seq)
                           for graph_seq in batch_graphs], dtype=torch.long)   
    max_length = max([len(graph_seq) for graph_seq in batch_graphs])           
    bond_vocab_file = f'./data/{args.dataset}/{args.mode}/bond_vocab.txt'
    atom_vocab_file = f'./data/{args.dataset}/{args.mode}/atom_lg_vocab.txt'
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))

    graph_seq_tensors = [] 
    edit_seq_labels = []   
    seq_mask = []          
    U_list = []
    Z_list = []
    D_list = []
    N_list = []

    for idx in range(max_length):  
        graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length-1)]).get_components(attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                      for i, length in enumerate(lengths)]  
        mask = (idx < lengths).long()   
        prod_graphs, edits, edit_atoms = list(zip(*graphs_idx)) 
        assert all([isinstance(graph, MolGraph) for graph in prod_graphs])  

        edit_labels = prepare_edit_labels(
            prod_graphs, edits, edit_atoms, bond_vocab, atom_vocab) 
        current_graph_tensors = get_batch_graphs(   
            prod_graphs, use_rxn_class=args.use_rxn_class)  
        
        graph_seq_tensors.append(current_graph_tensors)
        edit_seq_labels.append(edit_labels)
        seq_mask.append(mask)
        
        if idx == 0:
            for graph in prod_graphs:
                graph.count_chiral_centers()
                edit_len  = graph.num_bonds * bond_vocab.size() + graph.num_atoms * atom_vocab.size() + 1
                
                U = pad_tensor(pad_values(torch.tensor(graph.print_calculate_chiral_center_difference())), edit_len)
                Z = pad_tensor(pad_values(torch.tensor(graph.is_neighbor_bond())), edit_len)
                
                N = graph.getnum_chiral_center()
                if N == 0:
                    D = pad_tensor_one(pad_values_one(torch.tensor(graph.get_distance())), edit_len)
                else:
                    D = pad_tensor(pad_values_one(torch.tensor(graph.get_distance())), edit_len)
                
                U_list.append(U)
                Z_list.append(Z)
                D_list.append(D)
                N_list.append(N)
            N_list = torch.FloatTensor(N_list)
        

    seq_mask = torch.stack(seq_mask).long() 
    assert seq_mask.shape[0] == max_length
    assert seq_mask.shape[1] == len(batch_graphs)

    return graph_seq_tensors, edit_seq_labels, seq_mask, U_list, Z_list, D_list, N_list


def prepare_data(args: Any) -> None:
    """ 
    prepare data batches for edits prediction
    """
    datafile = f'./data/{args.dataset}/{args.mode}/{args.mode}.file.kekulized'
    rxns_data = joblib.load(datafile)

    batch_graphs = []   
    batch_num = 0

    if args.use_rxn_class:
        savedir = f'./data/{args.dataset}/{args.mode}/with_rxn_class/feat'
        chiral_savedir = f'./data/{args.dataset}/{args.mode}/with_rxn_class/chiral'
    else:
        savedir = f'./data/{args.dataset}/{args.mode}/without_rxn_class/feat'
        chiral_savedir = f'./data/{args.dataset}/{args.mode}/without_rxn_class/chiral'
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(chiral_savedir, exist_ok=True)

    for idx, rxn_data in enumerate(rxns_data):
        graph_seq = [] 
        rxn_smi = rxn_data.rxn_smi
        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        p_mol = Chem.MolFromSmiles(p)
        Chem.Kekulize(p_mol)
            
        if len(rxn_data.edits) > args.max_steps:    
            print(f'Edits step exceed max_steps. Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue
        
        p_mol,r_mol = addH(p_mol, r_mol) 
        int_mol = p_mol  
         
        for i, edit in enumerate(rxn_data.edits):  
            if int_mol is None:
                print("Interim mol is None")
                break
            if edit == 'Terminate':
                graph = RxnGraph(prod_mol=Chem.Mol( 
                    int_mol), edit_to_apply=edit, reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)
                graph_seq.append(graph)
                edit_exe = Termination(action_vocab='Terminate')
                try:
                    pred_mol = edit_exe.apply(Chem.Mol(int_mol))
                    final_smi = Chem.MolToSmiles(pred_mol) 
                except Exception as e:
                    final_smi = None
            else:
                graph = RxnGraph(prod_mol=Chem.Mol(int_mol), edit_to_apply=edit,   
                                 edit_atom=rxn_data.edits_atom[i], reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)    
                graph_seq.append(graph)
                int_mol = apply_edit_to_mol(
                    Chem.Mol(int_mol), edit, rxn_data.edits_atom[i])

        if len(graph_seq) == 0 or final_smi is None:
            print(f"No valid states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        batch_graphs.append(graph_seq)
        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(rxns_data)} {args.mode} reactions processed.")
            sys.stdout.flush()

        if (len(batch_graphs) % args.batch_size == 0) and len(batch_graphs):
            graph_seq_tensors, edit_seq_labels, seq_mask, U_list, Z_list, D_list, N_list = process_batch(batch_graphs, args)
            batch_tensors = graph_seq_tensors, edit_seq_labels, seq_mask   
            chiralmessage_batch_tensors = U_list, Z_list, D_list, N_list
            
            torch.save(batch_tensors, os.path.join(
                savedir, f'batch-{batch_num}.pt'))
            torch.save(chiralmessage_batch_tensors, os.path.join(
                chiral_savedir, f'batch-{batch_num}.pt'))

            batch_num += 1
            batch_graphs = []

    print(f"All {args.mode} reactions complete.")
    sys.stdout.flush()

    graph_seq_tensors, edit_seq_labels, seq_mask, U_list, Z_list, D_list, N_list = process_batch(batch_graphs, args)
    batch_tensors = graph_seq_tensors, edit_seq_labels, seq_mask  
    chiralmessage_batch_tensors = U_list, Z_list, D_list, N_list
    
    print("Saving..")
    torch.save(batch_tensors, os.path.join(
        savedir, f'batch-{batch_num}.pt'))
    torch.save(chiralmessage_batch_tensors, os.path.join(
        chiral_savedir, f'batch-{batch_num}.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_chiral')
    parser.add_argument('--mode', type=str, default='train',
                        help='Type of dataset being prepared: train or valid or test')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Number of shards")
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    parser.add_argument('--print_every', type=int,
                        default=1000, help='Print during preprocessing')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    prepare_data(args=args)


if __name__ == "__main__":
    main()
