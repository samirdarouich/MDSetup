Module pyLMP.tools.molecule_utils
=================================

Functions
---------

    
`adjust_bond_list_indexes(bond_list: numpy.ndarray)`
:   Function that takes a bond list with atom indices that are not consecutive (e.g.: after removing Hydrogens for a united atom approach)
    
    Args:
        bond_list (np.ndarray): Old bond list containing not consecutive atom indices
    
    Returns:
        (np.ndarray): Same bond list, just with consecutive atom indices

    
`assign_coos_via_distance_mat_local(coos_list: numpy.ndarray, distance_matrix: numpy.ndarray, atom_names: List[str], reference_matrix: numpy.ndarray, reference_names: List[str])`
:   Assigns coos to suit a reference based on a distance matrix relying to the coos.
    Reference and distance matrix/ coos belong to the same molecule type but are sorted in
    different ways.
    
    Args:
        coos_list: 
            - list of coordinates.
        distance_matrix:
            - distance matrix which belongs to the coos_list.
        atom_names:
            - names of the atoms corresponding to coos_list.
        reference_matrix:
            - distance matrix which belongs to the reference you want to apply the coos to.
        reference_names:
            - names of the reference molecule.
            
    Returns:
        new_coos_list:
            - list of coordinates fitting the reference.
        idx:
            - indexes to translate sth. to reference.

    
`get_molecule_coordinates(molecule_name_list: List[str], molecule_graph_list: List[str], molecule_smiles_list: List[str], verbose: bool = False) ‑> Tuple[List[numpy.ndarray], List[numpy.ndarray], List[numpy.ndarray], List[numpy.ndarray]]`
:   

    
`vis_mol(bond_list, bond_atomtypes)`
: