import re
import numpy as np
import moleculegraph
from rdkit import Chem
from rdkit.Chem import AllChem

from typing import List, Dict
from moleculegraph.molecule_utils import (
    graph_from_bonds,
    get_longest_path,
    bond_list_from_simple_path,
    get_diff_in_bond_lists,
    get_next_index,
    get_shortest_nontrivial_path,
)


class moleculegraph_syntax:
    def __init__(
        self, split="][", start="[", end="]", branch_operator="b", ring_operator="r"
    ):
        self.split = split
        self.start = start
        self.end = end
        self.branch_operator = branch_operator
        self.ring_operator = ring_operator
        return

    def get_from_string(self, molstring):
        mol_array = self.splitter(molstring)
        return self.get_from_array(mol_array)

    def get_from_array(self, mol_array):
        """
        - gets syntactic elemnts from a splitted molstring
        - rings and brances are marked with letters followed
          by a number giving the size
        """
        n = 0
        ff = np.zeros(len(mol_array))
        nn = -1 * np.ones(len(mol_array))
        for i, m in enumerate(mol_array):
            if re.sub(r"\d+", "", m) == self.branch_operator:
                ff[i] = int(re.sub("[^0-9]", "", m))
            elif re.sub(r"\d+", "", m) == self.ring_operator:
                ff[i] = -int(re.sub("[^0-9]", "", m))
            else:
                nn[i] = n
                n += 1
        return ff.astype(int), nn.astype(int)

    # def build_string_array(self, molecule):
    def build_string_array(self, atom_names, funs):
        def builder(x):
            if x > 0:
                x = self.branch_operator + str(int(np.abs(x)))
            elif x < 0:
                x = self.ring_operator + str(int(np.abs(x)))
            return x

        # dummy = molecule.f.copy().astype(str)
        dummy = funs.copy().astype(str)
        dummy[funs == 0] = atom_names  # molecule.atom_names
        # dummy[ molecule.f!=0]  = np.vectorize(builder)( molecule.f[ molecule.f!=0] )
        dummy[funs != 0] = np.vectorize(builder, otypes=[str])(funs[funs != 0])
        return dummy

    # def build_string(self,molecule):
    def build_string(self, atom_names, funs):
        mol_array = self.build_string_array(atom_names, funs)
        return self.stringer(mol_array)

    def splitter(self, molstring):
        a = len(self.start)
        b = len(self.end)
        return np.array(molstring[a:-b].split(self.split))

    def stringer(self, mol_array):
        return self.start + self.split.join(mol_array) + self.end


def get_fun_arrays_set_main(graph, bond_list, names, main_path, bond_types=[]):
    """
    generates a graphstring from a bond list and atom names
    uses the longest path from source to an end as main path

    Args:
        graph:
            - networkx graph object
        bond_list:
            - np.array, bond list
        names:
            - np.array, atom names
        main_path:
            - np.array, main path to build graph from
    Returns:
        str, graphstring to use with moleculegraph
    """
    funs = np.zeros(main_path.shape)
    fun_ranges = np.zeros(main_path.shape)

    main_path_bond_list = bond_list_from_simple_path(main_path)
    remaining_bonds = get_diff_in_bond_lists(bond_list, main_path_bond_list)

    if remaining_bonds.size > 0:
        while True:
            subgraph = graph_from_bonds(remaining_bonds)

            idx = get_next_index(main_path, remaining_bonds)

            subpath = get_longest_path(subgraph, source=idx)
            subpath_bond_list = bond_list_from_simple_path(subpath)
            match = np.intersect1d(main_path, subpath)

            if len(match) == 2 and len(subpath) == 2:
                # print("ring")
                subpath = get_shortest_nontrivial_path(graph, match[0], match[1])
                i = np.squeeze(np.where(main_path == match[0]))
                j = np.squeeze(np.where(main_path == match[1]))
                iinsert = np.max((i, j)) + 1

                main_path = np.insert(main_path, iinsert, [-1])
                fun_ranges = np.insert(fun_ranges, iinsert, [len(subpath)])
                funs = np.insert(funs, iinsert, [-1])

            elif len(match) == 1:
                # print("branch")
                i = np.squeeze(match)
                if subpath[0] != i:
                    subpath = subpath[::-1]
                subpath = subpath[1:]

                subfuns = np.concatenate([[1], np.zeros(subpath.shape)])
                subfun_ranges = np.concatenate(
                    [[len(subpath)], np.zeros(subpath.shape)]
                )
                subpath = np.concatenate([[-1], subpath])

                iinsert = np.squeeze(np.where(main_path == i)) + 1

                main_path = np.insert(main_path, iinsert, subpath)
                fun_ranges = np.insert(fun_ranges, iinsert, subfun_ranges)
                funs = np.insert(funs, iinsert, subfuns)

            else:
                # print("ERROR")
                return None, None

            main_path_bond_list = np.concatenate(
                [main_path_bond_list, subpath_bond_list]
            )
            remaining_bonds = get_diff_in_bond_lists(bond_list, main_path_bond_list)
            if len(remaining_bonds) == 0:
                break

    return funs * fun_ranges, main_path


#### Own utils


def filter_bonds_by_elements(
    bond_list: List[List[int]], atom_types: List[str], element: str
):
    """Filter bonds where the bonded atoms include specific elements."""
    cleaned_bond_list = []
    cleaned_atomtypes = []

    for b0, b1 in bond_list:
        a0, a1 = atom_types[[b0, b1]]
        # If element in present in this bond, skip it.
        if element in {a0, a1}:
            continue
        cleaned_bond_list.append((b0, b1))
        cleaned_atomtypes.append((a0, a1))

    return np.array(cleaned_bond_list), np.array(cleaned_atomtypes)


def adjust_bond_list_indexes(bond_list):
    """Adjust bond list indexes to a continuous range starting from zero."""
    old_indexes = np.unique(bond_list.flatten())
    new_indexes = np.arange(old_indexes.size)
    new_list = np.full(bond_list.shape, -1)

    for i, j in zip(old_indexes, new_indexes):
        new_list[bond_list == i] = j

    return new_list, old_indexes, new_indexes


def get_mol_from_bond_list(atom_types, bond_list):
    graph = graph_from_bonds(bond_list)

    main_path = get_longest_path(graph, source=0)

    a, b = get_fun_arrays_set_main(graph, bond_list, atom_types, main_path)
    b = b.astype(int)

    aatom_names = atom_types[b[b >= 0]]

    ms = moleculegraph_syntax().build_string(aatom_names, a)

    mol = moleculegraph.molecule(ms)

    return mol

def get_x_hop_neighboorhood(atom: Chem.Atom, molecule: Chem.Mol, depth: int=1, visited_atoms=None):
    if visited_atoms is None:
        visited_atoms = set()
        
    # Mark this atom as visited
    visited_atoms.add(atom.GetIdx())

    substructure = f"[{atom.GetSymbol()}]"
    
    if depth > 0:
        for bond in atom.GetBonds():
            neighbor_atom = bond.GetOtherAtom(atom)
            neighbor_idx = neighbor_atom.GetIdx()
            
            if neighbor_idx in visited_atoms:
                continue  # Skip already visited atoms to avoid cycles

            bond_type = bond.GetBondTypeAsDouble()
            bond_str = ""
            if bond_type == 1.0:
                bond_str = "-"
            elif bond_type == 2.0:
                bond_str = "="
            elif bond_type == 3.0:
                bond_str = "#"
                
            neighbor_string = get_x_hop_neighboorhood(neighbor_atom, molecule, depth-1, visited_atoms.copy())
            substructure += f"({bond_str}{neighbor_string})"

    return substructure


def get_forcefield_molecule_from_smiles(
    smiles: str,
    substructure_smarts: Dict[str, str],
    depth: int = 2,
    UA_flag: bool = False,
    verbose: bool = False,
):

    # Get molecule via rdkit
    molecule = Chem.MolFromSmiles(smiles)

    # Add H's to the atom
    molecule = Chem.AddHs(molecule)

    # Get 3D molecule
    AllChem.EmbedMolecule(molecule)

    # Add H-C to smarts to detect hydrogens that are bonded to carbon.
    # This is needed as filter in case united atoms are wanted.
    carbon_h_key = [
        key for key, value in substructure_smarts.items() if value == "[H]([C])"
    ]

    # If no carbon H is provided in force field add one manually
    # Else extract the carbon H key, to remove it later one
    if len(carbon_h_key) == 0:
        carbon_h_key = "carbon_h"
        substructure_smarts[carbon_h_key] = "[H]([C])"
    else:
        carbon_h_key = carbon_h_key[0]

    atom_symbols = []
    atom_types = []
    atom_nums = []
    atom_xyz = []

    if verbose:
        print(f"\nMatching atoms to force field for SMILES: {smiles}\n")

    # Loop through all atoms in the molecule and match them with provided substructers
    for i,atom in enumerate(molecule.GetAtoms()):
        substructure = get_x_hop_neighboorhood(atom, molecule, depth=depth)
        # If sanitize is activated some non valid sub neighborhoods
        # seem to not work
        substructure = Chem.MolFromSmiles(substructure, sanitize=False)

        for label, smarts in substructure_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            matches = substructure.GetSubstructMatches(pattern)

            # Check if main atom (atom_idx=0) of substructure
            # is matched as first item.
            if any(match[0]==0 for match in matches):
                matched_ff_type = label
                break
        
        # Add atom force field type
        atom_types.append(matched_ff_type)

        # Add atom symbol
        atom_symbols.append(atom.GetSymbol())

        # Add atomic number
        atom_nums.append(atom.GetAtomicNum())
        
        # Add cordinates
        positions = molecule.GetConformer().GetAtomPosition(i)
        atom_xyz.append([positions.x, positions.y, positions.z])
    
    # Create bond list
    bond_list = []
    for bond in molecule.GetBonds():
        bond_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if verbose:
        for atom_idx, ff_type in enumerate(atom_types):
            print(f"Atom {atom_idx} ({atom_symbols[atom_idx]}) is matched to ff type: {ff_type}")

    # Make all to numpy arrays
    atom_symbols = np.array(atom_symbols)
    atom_types = np.array(atom_types)
    atom_nums = np.array(atom_nums)
    atom_xyz = np.array(atom_xyz)
    bond_list = np.array(bond_list)

    # In case UA is wanted, clean all 'carbon_h' from the list
    UA_filter = carbon_h_key if UA_flag else ""

    cleaned_bond_list, _ = filter_bonds_by_elements(bond_list, atom_types, UA_filter)
    cleaned_bond_list, old_indexes, _ = adjust_bond_list_indexes(cleaned_bond_list)

    # Assuming atom_xyz and atom_names are defined elsewhere:
    cleaned_atom_symbols = atom_symbols[old_indexes]
    cleaned_atom_types = atom_types[old_indexes]
    cleaned_atom_nums = atom_nums[old_indexes]
    cleaned_atom_xyz = atom_xyz[old_indexes]

    if verbose and UA_flag:
        print("\nFinal molecule after removing carbon h's\n")
        for i, (cas, cat) in enumerate(zip(cleaned_atom_symbols, cleaned_atom_types)):
            print(f"Atom {i} ({cas}): {cat}")

    # Get moleculegraph representation of the molecule
    mol = get_mol_from_bond_list(cleaned_atom_types, cleaned_bond_list)

    # Save coordinates in molecule
    mol.coordinates = cleaned_atom_xyz

    # Save atom symbol in molecule
    mol.atomsymbols = cleaned_atom_symbols

    # Save atom nos in molecule
    mol.atomnos = cleaned_atom_nums

    return mol
