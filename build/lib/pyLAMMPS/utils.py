#!/usr/bin/env python

import os
import moleculegraph
import numpy as np
import networkx as nx
import pubchempy as pcp
import matplotlib.pyplot as plt
from jinja2 import Template
from typing import List, Dict


options = {
    "node_size": 2000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
    "alpha":.3,
    "font_size":16,
}

def vis_mol(bond_list,bond_atomtypes):
    graph  = nx.Graph()
    for (b0,b1),(bt0,bt1) in zip(bond_list,bond_atomtypes):
        graph.add_edge( "%s%d"%(bt0,b0), "%s%d"%(bt1,b1) )
        
    labels = {}
    nx.draw_networkx(graph, **options)
    nx.draw_networkx_labels(graph, nx.spring_layout(graph) , labels, font_size=12, font_color="black")

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    return

def adjust_bond_list_indexes(bond_list:np.ndarray):
    """Function that takes a bond list with atom indices that are not consecutive (e.g.: after removing Hydrogens for a united atom approach)

    Args:
        bond_list (np.ndarray): Old bond list containing not consecutive atom indices

    Returns:
        (np.ndarray): Same bond list, just with consecutive atom indices
    """
    old = np.unique(bond_list)
    new = np.arange(old.size)
    new_list = np.ones( bond_list.shape )*-1
    for i,j in zip(old,new):
        new_list[ bond_list == i ] = j
    return new_list,old


def assign_coos_via_distance_mat_local( coos_list: np.ndarray, distance_matrix: np.ndarray, atom_names: List[str],reference_matrix: np.ndarray, reference_names: List[str]):
    """
    Assigns coos to suit a reference based on a distance matrix relying to the coos.
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
    """

    
    distance_matrix_sort = np.sort(distance_matrix,axis=1)
    reference_sort = np.sort(reference_matrix,axis=1)
    idx = []
    
    for rm,rn in zip(reference_sort,reference_names):
        for i,(row,nn) in enumerate(zip(distance_matrix_sort,atom_names)):
            if np.array_equal(rm, row) and i not in idx and nn in rn:
                idx.append(i)
                break
            
    if not bool(idx):
        raise KeyError("Molecules could not be matched!\n")
    
    idx = np.array(idx)

    #print("""\n\nWARNING:
    #Assign_coos_via_distance_mat is not mature yet.
    #Double-check your results!!! \n \n""")
    
    return coos_list[idx], idx 

def get_molecule_coordinates( molecule_name_list: List[str], molecule_graph_list: List[str], molecule_smiles_list: List[str], 
                              xyz_destinations: List[str], template_xyz: str, UA: bool=True, verbose: bool=False ) -> None:

    # Get molecule objects via PupChem and visualize them
    mol_list  = [ pcp.get_compounds(smiles, "smiles", record_type='3d')[0] for smiles in molecule_smiles_list ]

    # Extract the coordinates, atom names and bond lists of each molecule
    atoms          = [ np.array([[a.x,a.y,a.z] for a in molecule.atoms]) for molecule in mol_list ]
    atom_names     = [ np.array([a.element for a in molecule.atoms]) for molecule in mol_list ]
    bond_list      = [ np.array([[b.aid1-1,b.aid2-1] for b in molecule.bonds]) for molecule in mol_list ]
    bond_atomtypes = [ np.array([[mol.atoms[bb[0]].element, mol.atoms[bb[1]].element] for bb in b]) for mol,b in zip(mol_list,bond_list)]

    if verbose:
        print("\nPubChem representation\n")
        for i,(n,bl,bat) in enumerate( zip(molecule_name_list,bond_list,bond_atomtypes) ):
            tmp = [ "%s%d"%(a.element,a.aid-1) for a in mol_list[i].atoms ]
            print("Molecule: %s"%n)
            print("Coordinates:\n" + "\n".join( ["  %s: %.3f %.3f %.3f"%(an, ax, ay, az) for an, (ax, ay, az) in zip( tmp, atoms[i] ) ] ) + "\n")
            vis_mol(bl,bat)

    # Filter out hydrogens bonded to C atoms for an united atom approach or just take every atom for all-atom molecules.
    cleaned_bond_list = []
    cleaned_atomtypes = []

    for atom,bonds in zip( bond_atomtypes, bond_list ):
        dummy1 = []
        dummy2 = []
        for (b0,b1),(a0,a1) in zip( bonds, atom ):
            if UA:
                if all([x in (a0,a1) for x in ["H","C"]]):
                    hydrogen = np.array([b0, b1])[ np.array([a0,a1]) == "H" ]
                    if verbose: print("C + H detected")                                   
                    if bonds[ bonds == hydrogen ].size == 1:          
                        if verbose: print("continue")                                     
                        continue
                if verbose: print( "keep",b0,b1,"->",a0,"--",a1 )   
            dummy1.append((b0,b1))
            dummy2.append((a0,a1))
        cleaned_bond_list.append(np.array(dummy1))
        cleaned_atomtypes.append(np.array(dummy2))
        if UA and verbose: print("\n")

    if UA and verbose:
        print("\nUnited atom representation\n")
        for n,bl,bat in zip(molecule_name_list,cleaned_bond_list,cleaned_atomtypes):
            print("Molecule: %s"%n)
            vis_mol(bl,bat)

    # Get moleculegraph representation of the molecules
    raw_mol_list  = [ moleculegraph.molecule(molecule_graph) for molecule_graph in molecule_graph_list ]

    if verbose:
        print("\nMoleculegraph representation\n")
        for name,raw_mol in zip( molecule_name_list, raw_mol_list) :
            print("Molecule: %s"%name)
            raw_mol.visualize()

    # Correct the atom indices in the bond list, after hydrogen atoms are removed for an united atom approach
    new_cleaned_bond_list = []
    new_idx = []

    for cbl in cleaned_bond_list:
        ncbl,p = adjust_bond_list_indexes(cbl)
        new_cleaned_bond_list.append(ncbl)
        new_idx.append(p)

    cleaned_coordinates = [ atom[idx] for atom,idx in zip(atoms, new_idx) ]
    cleaned_atomtyps    = [ atom_name[idx] for atom_name,idx in zip(atom_names, new_idx) ]

    # Get the correct name used for Playmol (force field type + running number of atom in the system)
    add_atom          = [1] + [ sum(mol.atom_number for mol in raw_mol_list[:(i+1)]) + 1 for i in range( len(raw_mol_list[1:]) ) ]
    raw_atom_numbers  = [ mol.atom_numbers + add_atom[i] for i,mol in enumerate(raw_mol_list) ]
    raw_atom_types    = [ mol.atom_names for mol in raw_mol_list ]  
    final_atomtyps    = [ ["%s%d"%(a,i) for a,i in zip( atn, idx ) ] for atn,idx in zip( raw_atom_types, raw_atom_numbers ) ]

    # Get the correct coordinates from the PubChem coordinates
    final_coordinates = []

    # Create distance matrix of the PubChem molecule. This is a method to match the PubChem molecule description to the moleculegraph description of the same molecule
    distance_matrix = [ moleculegraph.molecule_utils.get_distance_matrix(ncbl,np.unique(ncbl).size)[0] for ncbl in new_cleaned_bond_list ]

    # Match the PubChem distance matrix and atom names to the moleculegraph distance matrix and atom names to exctract the corret atom coordinates
    for mol,ca,cc,dm in zip( raw_mol_list, cleaned_atomtyps, cleaned_coordinates, distance_matrix ):

        # extract moleculegraph names of atoms
        reference_names = [n.split("_")[0] for n in mol.atom_names]

        # This function matches the distance matrix and thus reorder the cleaned coordinates to match the moleculegraph description
        fc,idx = assign_coos_via_distance_mat_local(  cc, dm, ca, mol.distance_matrix, reference_names )

        final_coordinates.append( fc )
        

    # Write coordinates to xyz files. These coordinates are sorted in the way the molecule is defined in 
    # the moleculegraph. This is important for step 2 of this workflow
    for xyz_destination, raw_atom_number, final_atomtyp, final_coordinate in zip( xyz_destinations, raw_atom_numbers, final_atomtyps, final_coordinates ):
        # Make folder if not already done
        os.makedirs( os.path.dirname(xyz_destination), exist_ok = True)

        # Write template for xyz file
        with open(template_xyz) as file:
            template = Template(file.read())

        rendered = template.render( atno  = len(raw_atom_number), 
                                    atoms = zip( final_atomtyp, final_coordinate ) )

        with open(xyz_destination, "w") as fh:
            fh.write( rendered )
    

## External functions for LAMMPS input ##

def write_pair_ff(settings: Dict[str, str | List | Dict], atom_numbers_ges: List[int], nonbonded: List[Dict[str,str]], 
                  ff_template: str="", lammps_ff_path: str="", relative_lammps_ff_path: str=""):
    """
    This function prepares LAMMPS understandable van der Waals pair interactions. If wanted these are written to a seperated force field file, or are added to the settings input dictionary.

    Args:
        settings (Dict[str, str  |  List  |  Dict]): Settings dictionary which will be used to render the LAMMPS input template. 
        atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
        nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: sigma, epsilon, name, m 
        ff_template (str, optional): Path to the force field template if it should be written to an external file instead into the LAMMPS input file. Defaults to "".
        lammps_ff_path (str, optional): Destination of the external force field file. Defaults to "".
        relative_lammps_ff_path (str, optional): Relative path to the external force field file from the lammps input file. Defaults to "".

    Raises:
        KeyError: If the settings dictionary do not contain the subdictionary "style".
    """

    if not "style" in settings.keys():
        raise KeyError("Settings dictionary do not contain the style subdictionary!")
    
    # Van der Waals pair interactions
    pair_interactions = []

    for i,iatom in zip(atom_numbers_ges, nonbonded):
        for j,jatom in zip(atom_numbers_ges[i-1:], nonbonded[i-1:]):
            
            name_ij   = f"{iatom['name']}  {jatom['name']}"

            if settings["style"]["mixing"] == "arithmetic": 
                sigma_ij   = ( iatom["sigma"] + jatom["sigma"] ) / 2
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif settings["style"]["mixing"] ==  "geometric":
                sigma_ij   = np.sqrt( iatom["sigma"] * jatom["sigma"] )
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif settings["style"]["mixing"] ==  "sixthpower": 
                sigma_ij   = ( 0.5 * ( iatom["sigma"]**6 + jatom["sigma"]**6 ) )**( 1 / 6 ) 
                epsilon_ij = 2 * np.sqrt( iatom["epsilon"] * jatom["epsilon"] ) * iatom["sigma"]**3 * jatom["sigma"]**3 / ( iatom["sigma"]**6 + jatom["sigma"]**6 )

            n_ij  = ( iatom["m"] + jatom["m"] ) / 2
            
            pair_interactions.append( { "i": i, "j": j, "sigma": round( sigma_ij, 4 ) , "epsilon": round( epsilon_ij, 4 ),  "m": n_ij, "name": name_ij } ) 

    # If provided write pair interactions as external LAMMPS force field file.
    if ff_template and lammps_ff_path and relative_lammps_ff_path:
        with open(ff_template) as file_:
            template = Template(file_.read())
        
        rendered = template.render( pair_interactions = pair_interactions )

        with open(lammps_ff_path, "w") as fh:
            fh.write(rendered) 

        settings["style"]["pairs_path"] = relative_lammps_ff_path

    else:
        # Otherwise add pair interactions in the style section
        settings["style"]["pairs"] = pair_interactions