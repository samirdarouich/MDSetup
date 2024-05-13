Module MDSetup.forcefield.molecule
==================================

Functions
---------

    
`adjust_bond_list_indexes(bond_list)`
:   Adjust bond list indexes to a continuous range starting from zero.

    
`clean_mollist(atom_names, atoms_xyz, bond_list, filter: Set[str] = {'H', 'C'})`
:   

    
`filter_bonds_by_elements(bond_list, atom_names, elements)`
:   Filter bonds where the bonded atoms include specific elements.

    
`get_fun_arrays_set_main(graph, bond_list, names, main_path, bond_types=[])`
:   generates a graphstring from a bond list and atom names
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

    
`get_mol_from_bond_list(atom_types, bond_list)`
:   

    
`get_molecule_from_smiles(smiles: str, forcefieldtypes: List[str])`
:   

    
`get_mollist_from_smiles(smiles: str)`
:   

Classes
-------

`moleculegraph_syntax(split='][', start='[', end=']', branch_operator='b', ring_operator='r')`
:   

    ### Methods

    `build_string(self, atom_names, funs)`
    :

    `build_string_array(self, atom_names, funs)`
    :

    `get_from_array(self, mol_array)`
    :   - gets syntactic elemnts from a splitted molstring
        - rings and brances are marked with letters followed
          by a number giving the size

    `get_from_string(self, molstring)`
    :

    `splitter(self, molstring)`
    :

    `stringer(self, mol_array)`
    :