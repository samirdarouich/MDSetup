#!/usr/bin/env python

import os
import moleculegraph
import numpy as np
import networkx as nx
import pubchempy as pcp
import matplotlib.pyplot as plt
from jinja2 import Template
from typing import List, Dict
from scipy.constants import Avogadro, epsilon_0, e


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
            
    idx = np.array(idx)
    
    print("""\n\nWARNING:
    Assign_coos_via_distance_mat is not mature yet.
    Double-check your results!!! \n \n""")
    
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

        rendered = template.render( atno  = raw_atom_number[-1], 
                                    atoms = zip( final_atomtyp, final_coordinate ) )

        with open(xyz_destination, "w") as fh:
            fh.write( rendered )


def get_local_dipol( molecule: moleculegraph.molecule, force_field: Dict[str,Dict]) -> List[List]:
    """
    Function to get the local dipols of a molecule

    Args:
        molecule (moleculegraph.molecule): Moleculegraph object of the molecule under investigation.
        force_field (dict[dict]): Dictionary contains the force field types of the molecule under investigation.

    Returns:
        dipol_list (List[List]): List with sublists for each local dipol with the corresponding atom types in the dipol.
    """

    flag       = 0
    dipol_list = []
    dipol      = []

    # Search through all bonds and check if an atom is charged. If thats the case, start a local dipol list.
    # Check the neighboring atom (in the bond), and if its also charged, append it to the list and add up the local dipol charge. (if one atom is already in the local dipol list, skip the charge evaluation)
    # Once the charge is zero, a local dipol is identified and appended to the overall dipol list

    for bond_types in molecule.bond_list:

        for atom_type in bond_types:
            if atom_type in dipol: 
                continue

            elif force_field["atoms"][molecule.atom_names[atom_type]]["charge"] != 0 and flag == 0:
                dipol = [ atom_type ]
                chrg  = force_field["atoms"][molecule.atom_names[atom_type]]["charge"]
                flag  = 1
            
            elif force_field["atoms"][molecule.atom_names[atom_type]]["charge"] != 0 and flag == 1 :
                chrg += force_field["atoms"][molecule.atom_names[atom_type]]["charge"]
                dipol.append(atom_type)

                if np.round(chrg,3) == 0:
                    dipol_list.append(dipol)
                    flag = 0
            
    return dipol_list

def calc_mie(r: np.ndarray, sigma: float, eps: float, n: int, m: int=6, key: str= "energy"):
    """
    This function computes the energy / force of the Mie potential for a given sigma, epsilon, distance, and repuslive, attrative exponent.

    Args:
        r (np.ndarray): Distance of the two atoms. Given in the same unit as sigma.
        sigma (float): Mixed sigma of the two atoms. Given in the same unit as the distance.
        eps (float): Mixed epsilon of the two atoms. Given in the energy unit.
        n (int): Repulsive exponent.
        m (int): Attractive exponent. Defaults to 6.
        key (str, optional): Key specifing if the energy (enery) or the force (force) should be computed. Defaults to "energy".

    Returns:
        mf (float): Either the energy or the force acting between the two atoms.
    """

    if eps > 0:

        # Mie constant
        c0 = n /  ( n - m ) * ( n / m )**( m / ( n - m ) )

        # energy U(r)
        if key == "energy":
            mf = c0 * eps * ( ( sigma / r ) ** n - ( sigma / r ) ** m) 
        
        # force = -dU(r)/dr
        elif key == "force":
            mf = -c0 * eps * ( m / r * ( sigma / r )**m - n / r * ( sigma/ r )**n )

    else:
        mf = np.zeros(len(r))

    return mf

def calc_coulomb(r: np.ndarray | np.ndarray, q1: float, q2: float, key: str= "energy"):
    """
    This function computes the energy / force of the Coulomb potential for given distance, and partial charges of the two atoms.

    Args:
        r (np.ndarray): Distance of the two atoms. Given in Angstrom.
        q1 (float): Partial charge of atom one. Given in portions of eletron charge.
        q2 (float): Partial charge of atom two. Given in portions of eletron charge.
        key (str, optional): Key specifing if the energy (enery) or the force (force) should be computed. Defaults to "energy".

    Returns:
        qf (float): Either the energy or the force acting between the two atoms. Unit is either kcal/(mol*AA) or kcal/(mol*AA^2)
    """
    # epsilon_0: As/Vm = C^2/(Nm^2)
    # distance: m = 10^-10AA
    # electron charge: C
    # q1*q2/(4*pi*eps_0*r) = const * q1*q2/r --> Unit: C^2 / ( C^2/(Nm^2) * m) --> Nm = J
    # * Na (1/mol) / 4184 kcal/J --> kcal/mol

    if q1 != 0.0 and q2!= 0.0:
        
        const = e**2 / ( 4 * np.pi * epsilon_0 * 1e-10 )  *  Avogadro / 4184

        # Energy U(r) = const*q1q2/r
        if key == "energy":
            qf = q1 * q2 * const / r 
        
        # Force = -dU(r)/dr = -d/dr (const*q1*q2*r^(-1)) = - (-const*q1*q2/r^2) = const*q1*q2/r^2
        elif key == "force":
            qf = q1 * q2 * const / r**2
    else:
        qf = np.zeros(len(r))

    return qf

def calc_bond(r: np.ndarray, r0: float, K: float, key = "energy"):
    """
    This function computes the energy / force of the harmonic bond potential for given distance, spring constant and equilibrium length.

    Args:
        r (np.ndarray): Distance of the two atoms. Given in the same unit as sigma.
        r0 (float): Equilibrium length of the bond.
        K (float): Harmonic spring constant of the bond.
        key (str, optional): Key specifing if the energy (enery) or the force (force) should be computed. Defaults to "energy".

    Returns:
        bf (float): Either the energy or the force acting between the two atoms. 
    """
    # Energy U(r) = K * (r-r0)^2
    if key == "energy":
        bf = K * ( r - r0 )**2
    
    # Force -dU(r)/dr = -K * 2 * (r-r0)
    elif key == "force":
        bf = -K * 2 * ( r - r0 )
        
    return bf
        