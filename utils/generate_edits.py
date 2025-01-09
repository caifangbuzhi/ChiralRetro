from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)
from utils.chem import align_kekulize_pairs, get_atom_info, get_bond_info
from rdkit import Chem
from collections import namedtuple
from typing import Tuple

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

ReactionData = namedtuple(
    "ReactionData", ['rxn_smi', 'edits', 'edits_atom', 'rxn_class', 'rxn_id'])

def generate_reaction_edits(rxn_smi: str, kekulize: bool = False, rxn_class: int = None, rxn_id: str = None) -> ReactionData:
    r, p = rxn_smi.split(">>")
    react_mol = Chem.MolFromSmiles(r)
    prod_mol = Chem.MolFromSmiles(p)

    if react_mol is None or prod_mol is None:
        return None

    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()   
                  for atom in prod_mol.GetAtoms()}
    max_amap = max([atom.GetAtomMapNum() for atom in react_mol.GetAtoms()]) 
    for atom in react_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                  for atom in react_mol.GetAtoms()}    

    r_new, p_new = Chem.MolToSmiles(react_mol), Chem.MolToSmiles(prod_mol)
    rxn_smi_new = r_new + ">>" + p_new  
    if kekulize:
        react_mol, prod_mol = align_kekulize_pairs(react_mol, prod_mol)
        
    prod_mol, react_mol = addH(prod_mol, react_mol)
    
    prod_bonds = get_bond_info(prod_mol)    
    react_bonds = get_bond_info(react_mol) 

    edits = []  
    edits_atom = [] 
    bond_edits_atom = set() 

    for bond in prod_bonds:
        if bond not in react_bonds: 
            a1, a2 = bond   
            edit = BondEditAction(a1, a2, None, None,
                                  action_vocab='Delete Bond')
            edits.append(edit.get_tuple())
            edits_atom.append([a1, a2])
            bond_edits_atom.add(a1)
            bond_edits_atom.add(a2)

    for bond in prod_bonds: 
        if bond in react_bonds and prod_bonds[bond] != react_bonds[bond]:
            a1, a2 = bond
            edit = BondEditAction(
                a1, a2, *react_bonds[bond], action_vocab='Change Bond')
            edits.append(edit.get_tuple())
            edits_atom.append([a1, a2])
            bond_edits_atom.add(a1)
            bond_edits_atom.add(a2)

    for bond in react_bonds:    
        if bond not in prod_bonds:
            a1, a2 = bond
            if a1 in p_amap_idx and a2 in p_amap_idx:
                edit = BondEditAction(
                    a1, a2, *react_bonds[bond], action_vocab='Add Bond')
                edits.append(edit.get_tuple())
                edits_atom.append([a1, a2])
                bond_edits_atom.add(a1)
                bond_edits_atom.add(a2)

    prod_atoms = get_atom_info(prod_mol)   
    react_atoms = get_atom_info(react_mol)  
    atoms_only_in_react = []
    for atom in react_atoms:
        if atom not in prod_atoms:
            atoms_only_in_react.append(atom)

    if len(edits_atom) == 0:    
        for atom in prod_atoms:
            if prod_atoms[atom] != react_atoms[atom]:
                edit = AtomEditAction(
                    atom, *react_atoms[atom], action_vocab='Change Atom')
                edits.append(edit.get_tuple())
                edits_atom.append(atom)
    else:   
        for atom in prod_atoms:
            if prod_atoms[atom] != react_atoms[atom]:   
                if atom not in bond_edits_atom:
                    edit = AtomEditAction(
                        atom, *react_atoms[atom], action_vocab='Change Atom')
                    edits.append(edit.get_tuple())
                    edits_atom.append(atom)
                    
                else:   
                    if prod_atoms[atom][1] != react_atoms[atom][1]:
                        edit = AtomEditAction(
                            atom, *react_atoms[atom], action_vocab='Change Atom')
                        edits.append(edit.get_tuple())
                        edits_atom.append(atom)

    for bond in react_mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        a1, a2 = sorted([a1, a2])
        if a1 not in atoms_only_in_react and a2 in atoms_only_in_react:
            frags1 = Chem.FragmentOnBonds(react_mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])  
            frags1_smi = Chem.MolToSmiles(frags1)   
            frags1_smi = frags1_smi.split('.')  
            for smi in frags1_smi:
                mol = Chem.MolFromSmiles(smi)
                for a in mol.GetAtoms():
                    if a.GetSymbol() == '*':
                        atoms_only_in_react.append(a.GetAtomMapNum())   
                if all(a.GetAtomMapNum() in atoms_only_in_react for a in mol.GetAtoms()):   
                    smi = Chem.MolToSmiles(mol)
                    edit = AddGroupAction(a1, smi, action_vocab='Attaching LG') 
                    if edit.get_tuple() in edits:
                        continue
                    else:
                        edits.append(edit.get_tuple())  
                        edits_atom.append(a1)   
                elif any(a.GetAtomMapNum() in atoms_only_in_react for a in mol.GetAtoms() if a.GetAtomMapNum() != 0):
                    for bond in mol.GetBonds():
                        a3, a4 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
                        a3, a4 = sorted([a3, a4])
                        if a3 not in atoms_only_in_react and a3 == a1 and a4 in atoms_only_in_react and a4 != 0:
                            frags2 = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])  
                            frags2_smi = Chem.MolToSmiles(frags2)
                            frags2_smi = frags2_smi.split('.')
                            for smi in frags2_smi:
                                mol_2 = Chem.MolFromSmiles(smi)
                                if all(a.GetAtomMapNum() in atoms_only_in_react for a in mol_2.GetAtoms()):
                                    smi = Chem.MolToSmiles(mol_2)
                                    edit = AddGroupAction(
                                        a1, smi, action_vocab='Attaching LG')
                                    if edit.get_tuple() in edits:
                                        continue
                                    else:
                                        edits.append(edit.get_tuple())
                                        edits_atom.append(a1)
                                        
    final_edits = []
    for edit in edits:
        if edit[0] == 'Attaching LG':
            mol = Chem.MolFromSmiles(edit[1])
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            smi = Chem.MolToSmiles(mol)
            edit = tuple(['Attaching LG', smi])
        final_edits.append(edit)   

    edit = Termination(action_vocab='Terminate')
    final_edits.append(edit.get_tuple())   

    reaction_data = ReactionData(
        rxn_smi=rxn_smi_new, edits=final_edits, edits_atom=edits_atom, rxn_class=rxn_class, rxn_id=rxn_id)

    return reaction_data
