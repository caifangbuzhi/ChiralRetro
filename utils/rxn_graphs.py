from typing import List, Tuple, Union

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from utils.mol_features import get_atom_features, get_bond_features

import networkx as nx

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

CHIRALTAG_PARITY_DIR = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,  
}

def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown

    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY_DIR[atom.GetChiralTag()]


class MolGraph:
    """
    'MolGraph' represents the graph structure and featurization of a single molecule.

     A MolGraph computes the following attributes:

    * n_atoms: The number of atoms in the molecule.
    * n_bonds: The number of bonds in the molecule.
    * f_atoms: A mapping from an atom index to a list of atom features.
    * f_bonds: A mapping from a bond index to a list of bond features.
    * a2b: A mapping from an atom index to a list of incoming bond indices.
    * b2a: A mapping from a bond index to the index of the atom the bond originates from.
    * b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,   
            Reaction class for this reaction.   
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self.use_rxn_class = use_rxn_class
        self._build_mol()
        self._build_graph()
   
    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()   
        self.num_bonds = self.mol.GetNumBonds()  
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()  
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key,
                            value in self.amap_to_idx.items()}
 
    def _build_graph(self):
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype
         
        """Builds the graph attributes."""
        self.n_atoms = 0   
        self.n_bonds = 0  
        self.f_atoms = []   
        self.f_bonds = []  
        self.a2b = []  
        self.b2a = []  
        self.b2revb = []  
        self.parity_atoms = []
        self.directed_b2a = []

        self.f_atoms = [get_atom_features(
            atom, rxn_class=self.rxn_class, use_rxn_class=self.use_rxn_class) for atom in self.mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)   

        for _ in range(self.n_atoms):
            self.a2b.append([])
            
        for atom in self.mol.GetAtoms():
            self.parity_atoms.append(parity_features(atom))
                
        for a1 in range(self.n_atoms): 
            for a2 in range(a1 + 1, self.n_atoms):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = get_bond_features(bond)    

                self.f_bonds.append(self.f_atoms[a1] + f_bond)  
                self.f_bonds.append(self.f_atoms[a2] + f_bond)   

                b1 = self.n_bonds   
                b2 = b1 + 1
                self.directed_b2a.append([a1, a2])
                self.directed_b2a.append([a2, a1])
                self.a2b[a2].append(b1)   
                self.b2a.append(a1)
                self.a2b[a1].append(b2)   
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
               
    def count_chiral_centers(self) -> None:
        """Counts and marks chiral centers in the molecule."""
        chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=False)    
        self.chiral_centers = [idx for idx, _ in chiral_centers]
        for idx in self.chiral_centers:
            self.G_undir.nodes[idx]['chiral'] = True

    def getnum_chiral_center(self):
        return len(self.chiral_centers)
    
    def calculate_chiral_center_difference(self, bond: Tuple[int, int]) -> int:
        """Calculates the absolute difference in the number of chiral centers
        between the two parts of the molecule after breaking the bond.

        Parameters
        ----------
        bond: Tuple[int, int],
            The bond to break (represented by a tuple of atom indices).

        Returns
        -------
        int,
            The absolute difference in the number of chiral centers between the two parts.
        """
        G_temp = self.G_undir.copy()
        G_temp.remove_edge(*bond)
        
        components = List(nx.connected_components(G_temp))
        if len(components) != 2:
            raise ValueError("The molecule did not split into exactly two components.")

        chiral_counts = [sum(1 for node in component if G_temp.nodes[node].get('chiral', False)) for component in components]
        return abs(chiral_counts[0] - chiral_counts[1])
    
    def print_calculate_chiral_center_difference(self):
        u_list = []
        for id, value in self.G_undir.nodes(data='chiral'):
            if value == True:
                for u, v, attr in self.G_undir.edges(data='label'):
                    bond = (u, v)
                    G_temp = self.G_undir.copy()
                    G_temp.remove_edge(*bond)
                    
                    components = list(nx.connected_components(G_temp))

                    if len(components) == 2:
                        chiral_counts = [sum(1 for node in component if G_temp.nodes[node].get('chiral', False)) for component in components]
                        u_list.append(abs(chiral_counts[0] - chiral_counts[1]))
                    elif len(components) == 1:
                        u_list.append(0) 
                return u_list
        u_list.append([0] * self.num_bonds)
        return u_list[0]
    
    def is_neighbor_bond(self) -> List[int]:
        """Returns a list indicating if a bond is a neighbor to a chiral center."""
        neighbor_bond_list = []
        
        for u, v in self.G_undir.edges():
            if self.G_undir.nodes[u].get('chiral', False) or self.G_undir.nodes[v].get('chiral', False):
                neighbor_bond_list.append(1)
            else:
                neighbor_bond_list.append(0)
        
        return neighbor_bond_list
    
    def get_distance(self) -> List[int]:
        """Returns a list indicating the sum of distances from each bond to all chiral centers."""
        distance_list = []
        shortest_paths = {chiral: nx.shortest_path_length(self.G_undir, source=chiral) for chiral in self.chiral_centers}
        
        for u, v in self.G_undir.edges():
            distance_sum = 0
            for chiral in self.chiral_centers:
                distance_sum += min(shortest_paths[chiral][u], shortest_paths[chiral][v]) + 1
            distance_list.append(distance_sum)
            
        if all(x == 0 for x in distance_list):
            return [1] * len(distance_list)
        return distance_list
                


class RxnGraph:
    """
    RxnGraph contains the information of a reaction, like reactants, products. The edits associated with the reaction are also captured in edit labels.
    """
    def __init__(self, prod_mol: Chem.Mol, edit_to_apply: Tuple, edit_atom: List = [], reac_mol: Chem.Mol = None, rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edit_to_apply: Tuple,
            Edits to apply to the product molecule
        edit_atom: List,
            Edit atom of product molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.prod_graph = MolGraph(
            mol=prod_mol, rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        if reac_mol is not None:
            self.reac_mol = reac_mol
        self.edit_to_apply = edit_to_apply
        self.edit_atom = edit_atom
        self.rxn_class = rxn_class
 
    def get_components(self, attrs: List = ['prod_graph', 'edit_to_apply', 'edit_atom']) -> Tuple:
        """ 
        Returns the components associated with the reaction graph. 
        """
        attr_tuple = ()
        for attr in attrs:
            if hasattr(self, attr):
                attr_tuple += (getattr(self, attr),)     
            else:
                print(f"Does not have attr {attr}")

        return attr_tuple


class Vocab:
    """
    Vocab class to deal with vocabularies and other attributes.
    """

    def __init__(self, elem_list: List) -> None:
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list
        if isinstance(elem_list, dict):
            self.elem_list = list(elem_list.keys())
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}

    def __getitem__(self, a_type: Tuple) -> int:    
        return self.elem_to_idx[a_type]

    def get(self, elem: Tuple, idx: int = None) -> int:
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)

    def get_elem(self, idx: int) -> Tuple:
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]

    def __len__(self) -> int:
        return len(self.elem_list)

    def get_index(self, elem: Tuple) -> int:
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]

    def size(self) -> int:
        """Returns length of Vocab."""
        return len(self.elem_list)
