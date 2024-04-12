import os
import toml
import json
import yaml
import numpy as np
import moleculegraph
from jinja2 import Template
from typing import Dict, List, Any
from scipy.constants import Avogadro

## LAMMPS molecule class

class LAMMPS_molecules():
    """
    This class writes LAMMPS data input for arbitrary mixtures using moleculegraph.
    """
    def __init__(self, mol_str: List[str], force_field_path: str ):

        # Save moleclue graphs of both components class wide
        self.mol_str    = mol_str
        self.mol_list   = [ moleculegraph.molecule(mol) for mol in self.mol_str ]

        # Read in force field file
        if ".yaml" in force_field_path:
            self.ff = yaml.load( open(force_field_path) )
        elif ".json" in force_field_path:
            self.ff = json.load( open(force_field_path) )
        elif ".toml" in force_field_path:
            self.ff = toml.load( open(force_field_path) )
        else:
            raise KeyError("Force field file is not supported. Please provide 'YAML', 'JSON', or 'TOML' file.")
            

        ## Map force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions) ##

        # Get (unique) atom types and parameters
        self.nonbonded = [j for sub in [molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) bond types and parameters
        self.bonds     = [j for sub in [molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) angle types and parameters
        self.angles    = [j for sub in [molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) torsion types and parameters 
        self.torsions  = [j for sub in [molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in self.mol_list] for j in sub]
        
        if not all( [ all(self.nonbonded), all(self.bonds), all(self.angles), all(self.torsions) ] ):
            txt = "nonbonded" if not all(self.nonbonded) else "bonds" if not all(self.bonds) else "angles" if not all(self.angles) else "torsions"
            raise ValueError("Something went wrong during the force field mapping for key: %s"%txt)
        
        # Get nonbonded force field for all atom types not only the unique one. This is later used to extract the charge of each atom in the system while writing the LAMMPS data file.
        self.ff_all    = np.array([j for sub in [molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) for molecule in self.mol_list] for j in sub])


    def prepare_lammps_force_field(self):
        """
        Save force field parameters (atoms, bonds, angles, torsions) used for LAMMPS input.
        """

        #### Define general settings that are not system size dependent

        ## Definitions for atoms in the system

        # This is a helper list, that higher the unique atom indexes in such a way that the atom indexes of the 2nd component 
        # start after the indexes of the first component (e.g.: Component 1: 1 cH_alcohol, 2 OH_alcohol 3 CH2_alcohol --> Component 2: 4 CH3_alkane, ...)
        # As molecule graph just gives for each component the indexes starting from 0 --> Component 1: 0 cH_alcohol, 1 OH_alcohol 2 CH2_alcohol; Component 2: 0 CH3_alkane, ... 
        # one needs to add the index of all preceding molecules.
        add_atoms                    = [1] + [ sum(len(mol.unique_atom_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        # Get the force field type identifiers for each atom. This is used in the #Atoms section in the data file.
        # Therefore take the identifier number of every atom in each moleculegraph and add the preceeding number of atom types.
        # 1 1 1 0.404 -10.69874 -8.710742 1.59434903 # cH_alcohol1 --> First index is a running LAMMPS number for the atom. The 2nd index is the running numner for the molecule.
        # The third index is the force field type identifier (here it is 1, thus it is type the cH_alcohol type, as defined in the input file header)
        # The forth number is the partial charge. The following 3 numbers are the x, y and z coordinate of the atom in the system.
        self.atoms_running_number    = np.concatenate( [mol.unique_atom_inverse + add_atoms[i] for i,mol in enumerate(self.mol_list)], axis=0 )
        
        # These are the !unique! atom force field types from "atoms_running_number". This is used in the atom definition section of the data file.
        # As just the unique atom force field types are defined in LAMMPS. 
        self.atom_numbers_ges        = np.unique( self.atoms_running_number )

        # Get the number of atoms per component. This will later be multiplied with the number of molecules per component to get the total number of atoms in the system
        self.number_of_atoms         = [ mol.atom_number for mol in self.mol_list ]

        # Define the total number of atom tpyes
        self.atom_type_number = len( self.atom_numbers_ges )

        ## Definitions for bonds in the system ##

        # This is the same as for the unique atom types, just for the unique bond types. One need to add the amount of all preceeding unique
        # bond types to every bond type of each component.
        add_bonds = [1] + [ sum(len(mol.unique_bond_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        # Get the identifiers for each bond with there corresponding force field type. 
        # (e.g.: Ethanediol: 1 2 3 2 1: where 1 is the cH_alc - OH_alc bond, 2 is the OH_alc - CH2_alc bond, and 3 the CH2_alc - CH2_alc bond)
        # These types will be written in the # Bonds section. Where every bond for every atom in the system is defined. 
        # (e.g.: 1 1 1 5 # cH_alc OH_alc --> the first number is a running index for LAMMPS, the 2nd is the force field bond type index (as defined in bonds_running_number),
        # and the 3th and 4th are the atoms in this bond (as defined in self.bond_numbers)
        self.bonds_running_number    = np.concatenate( [mol.unique_bond_inverse + add_bonds[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")

        # These are the !unique! bond force field types from "bonds_running_number". This is used in the bonds definition section of the data file.
        # As just the unique bond force field types are defined in LAMMPS. 
        self.bond_numbers_ges        = np.unique( self.bonds_running_number )
        
        # This identify the atoms of each bond in the molecule. This is used in #Bonds section, where each Atom is assigned a bond, as well as the bond force field type. 
        # To these indicies the number of preceeding atoms in the system will be added (in the write_lammps_data function).
        # Thus the direct list from molecule graph can be used, without further refinement. 
        # (E.g.: Bond_list for ethanediol: [ [1,2], [2,3], [3,4], [4,5], [5,6] ] )
        self.bond_numbers            = np.concatenate( [mol.bond_list + 1 for mol in self.mol_list], axis=0 ).astype("int")

        # This is just the name of each the bonds defined above. This is written as information that one knows what which bond is defined in the data file.
        self.bond_names              = np.concatenate( [[[mol.atom_names[i] for i in bl] for bl in mol.bond_list] for mol in self.mol_list], axis=0 )

        # Get the number of bonds per component. This will later be multiplied with the number of molecules per component to get the total number of bonds in the system
        self.number_of_bonds         = [ len(mol.bond_keys) for mol in self.mol_list ]

        # If several bond styles are used, these needs to be added in the data file, as well as the "hybrid" style.
        self.bond_styles             = list( np.unique( [ p["style"] for p in self.bonds ] ) )

        # Defines the total number of bond types
        self.bond_type_number        = len( self.bond_numbers_ges )


        ## Definitions for angles in the system --> (all elements here have the same meaning as above for bonds just for angles) ##
        
        add_angles = [1] + [ sum(len(mol.unique_angle_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        self.angles_running_number   = np.concatenate( [mol.unique_angle_inverse + add_angles[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")
        self.angle_numbers_ges       = np.unique( self.angles_running_number )
        self.angle_numbers           = np.concatenate( [mol.angle_list + 1 for mol in self.mol_list], axis=0 ).astype("int")
        self.angle_names             = np.concatenate( [[[mol.atom_names[i] for i in al] for al in mol.angle_list] for mol in self.mol_list], axis=0 )
        self.number_of_angles        = [ len(mol.angle_keys) for mol in self.mol_list ]

        self.angle_styles            = list( np.unique( [ p["style"] for p in self.angles] ) )
        self.angle_type_number       = len(self.angle_numbers_ges)


        ## Definitions for torsions in the system --> (all elements here have the same meaning as above for bonds just for torsions) ##
        
        add_torsions = [1] + [ sum(len(mol.unique_torsion_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]
        
        self.torsions_running_number = np.concatenate( [mol.unique_torsion_inverse + add_torsions[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")
        self.torsion_numbers_ges     = np.unique( self.torsions_running_number )
        self.torsion_numbers         = np.concatenate( [mol.torsion_list + 1 for mol in self.mol_list], axis=0 ).astype("int")
        self.torsion_names           = np.concatenate( [[[mol.atom_names[i] for i in tl] for tl in mol.torsion_list] for mol in self.mol_list], axis=0 )
        self.number_of_torsions      = [ len(mol.torsion_keys) for mol in self.mol_list ]

        self.torsion_styles          = list( np.unique( [ p["style"] for p in self.torsions ] ) )
        self.torsion_type_number     = len(self.torsion_numbers_ges)
        
        return

    def write_lammps_data( self, xyz_path: str, data_template: str, data_path: str,
                           nmol_list: List[int], density: float):
        """
        Function that generates a LAMMPS data file.

        Args:
            xyz_path (str): Path to the xyz file for this system.
            data_template (str): Path to the jinja2 template for the LAMMPS data file.
            data_path (str): Path where the LAMMPS data file should be generated.
        """

        if not os.path.exists(data_template):
            raise FileExistsError(f"Data template does not exists:\n   {data_template}")
        
        if not os.path.exists(xyz_path):
            raise FileExistsError(f"Coordinate file does not exists:\n   {xyz_path}")

        #### System specific settings ####

        # This is used to write the header of the data file 
        self.atom_paras = zip(self.atom_numbers_ges, self.nonbonded)

        # Total atoms in system (summation of the number of atoms of each component times the number of molecules of each component)
        self.total_number_of_atoms    = np.dot( self.number_of_atoms, nmol_list )

        # Total bonds in system 
        self.total_number_of_bonds    = np.dot( self.number_of_bonds, nmol_list )
        
        # Total angles in system 
        self.total_number_of_angles   = np.dot( self.number_of_angles, nmol_list) 
        
        # Total torsions in system 
        self.total_number_of_torsions = np.dot( self.number_of_torsions, nmol_list )


        ## Mass, mol, volume and box size of the system ##

        # Molar masses of each species [g/mol]
        Mol_masses = np.array( [ np.sum( [ a["mass"] for a in molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) ] ) for molecule in self.mol_list ] )

        # Account for mixture density --> in case of pure component this will not alter anything

        # mole fraction of mixture (== numberfraction)
        x = np.array( nmol_list ) / np.sum( nmol_list )

        # Average molar weight of mixture [g/mol]
        M_avg = np.dot( x, Mol_masses )

        # Total mole n = N/NA [mol] #
        n = np.sum( nmol_list ) / Avogadro

        # Total mass m = n*M [kg]
        mass = n * M_avg / 1000

        # Compute box volume V=m/rho and with it the box lenght L (in Angstrom) --> assuming orthogonal box
        # With mass (kg) and rho (kg/m^3 --> convert in g/A^3 necessary as lammps input)

        # Volume = mass / mass_density = mol / mol_density [A^3]
        volume = mass / density * 1e30

        boxlen = volume**(1/3) / 2

        box = [ -boxlen, boxlen ]
    
        # Running counts of atoms, bonds, angles, and torsions.
        atom_count       = 0
        bond_count       = 0
        angle_count      = 0
        torsion_count    = 0

        # Introduce a count for the number of molecules of each component. This will be updated during the loop and at the end have the same numbers as the nmol_list.
        mol_count        = np.zeros(len(self.mol_list)).astype("int")

        # Lists containing all the lines for atoms, bonds, angles, and torsions that will be written into the #Atoms, #Bonds, #Angles and #Dihedrals section in the data file.
        lmp_atom_list    = []
        lmp_bond_list    = []
        lmp_angle_list   = []
        lmp_torsion_list = []

        # Read in the specified coordinates --> the atom names are also read in (these are the atom names given in playmol setup. Double check if they match the force field type you expect!)
        coordinates      = moleculegraph.general_utils.read_xyz(xyz_path)
        
        # Get the number of atoms per component as list
        component_atom_numbers = [mol.atom_number for mol in self.mol_list] 

        for m,mol in enumerate(self.mol_list):

            # All the lists produced yet are one flat 1D list, containing the information of every atom of every component.
            # As we here loop through every component individually, the correct entries of all the lists need to be taken.
            # (E.g.: For the correct atoms_running_number of component 1, one needs to take the list entries from 0 to n° of atoms of component 1
            # The same needs to be done for component 2: from n° of atoms of component 1 to n° of atoms of component 1 + component 2, and so on...)
            # The same is true for the bonds, angles and torsions
            idx  = mol.atom_numbers + sum( mole.atom_number for mole in self.mol_list[:m] )
            idx1 = np.arange( len(mol.bond_keys) ) + sum( len(mole.bond_keys) for mole in self.mol_list[:m] )
            idx2 = np.arange( len(mol.angle_keys) ) + sum( len(mole.angle_keys) for mole in self.mol_list[:m] )
            idx3 = np.arange( len(mol.torsion_keys) ) + sum( len(mole.torsion_keys) for mole in self.mol_list[:m] )
            

            ## Now write LAMMPS input for every molecule of each component ##

            for mn in range(nmol_list[m]):
                
                # The atom index in the bond, angle, torsion list starts always at 1. Thus to write the correct atoms for each bond, angle, torsion one need to 
                # know the indcies of the atoms of the current molecule. To do so, add the dot product of all atom numbers and molecules of each component
                # to the atom index in the current bond, angle, torsion.
                add_atom_count = np.dot( mol_count, component_atom_numbers ).astype("int")

                # Define atoms
                for atomtype,ff_atom in zip( self.atoms_running_number[idx], self.ff_all[idx] ):
                    
                    atom_count +=1

                    # LAMMPS INPUT: total n° of atom in system, mol n° in system, atomtype, partial charges,coordinates
                    line = [ atom_count, sum(mol_count)+1, atomtype, ff_atom["charge"],*coordinates[atom_count-1]["xyz"], coordinates[atom_count-1]["atom"] ]

                    lmp_atom_list.append(line)


                # Define bonds 
                for bondtype,bond,bond_name in zip( self.bonds_running_number[idx1], self.bond_numbers[idx1], self.bond_names[idx1] ):

                    bond_count += 1

                    # Higher the indices of the bond to match the atoms of the current molecule.
                    dummy = bond + add_atom_count
      
                    # LAMMPS INPUT: total n° of bond in system, bond force field type, atom n° in this bond
                    line  = [ bond_count, bondtype, *dummy, " ".join(bond_name) ]

                    lmp_bond_list.append(line)

                # Define angles
                for angletype,angle,angle_name in zip (self.angles_running_number[idx2], self.angle_numbers[idx2], self.angle_names[idx2] ):

                    angle_count += 1

                    # Higher the indices of the angle to match the atoms of the current molecule.
                    dummy = angle + add_atom_count

                    # LAMMPS INPUT: total n° of angles in system, angle force field type, atom n° in this angle
                    line = [ angle_count, angletype, *dummy, " ".join(angle_name) ]

                    lmp_angle_list.append(line)

                # Define torsions
                for torsiontype,torsion,torsion_name in zip( self.torsions_running_number[idx3], self.torsion_numbers[idx3], self.torsion_names[idx3] ):

                    torsion_count += 1

                    # Higher the indices of the dihedral to match the atoms of the current molecule.
                    dummy = torsion + add_atom_count

                    # LAMMPS INPUT: total n° of torsions in system, torsion force field type, atom n° in this torsion
                    line = [ torsion_count, torsiontype, *dummy, " ".join(torsion_name) ]

                    lmp_torsion_list.append(line)

                # Increase the molecule count of the current component by one.
                mol_count[m] += 1

        renderdict = { "box_x": box, "box_y": box, "box_z": box,
                       "atom_type_number": self.atom_type_number,
                       "bond_type_number": self.bond_type_number,
                       "angle_type_number": self.angle_type_number,
                       "torsion_type_number": self.torsion_type_number,
                       "atom_paras": self.atom_paras  }

        renderdict["atoms"]    = lmp_atom_list
        renderdict["bonds"]    = lmp_bond_list
        renderdict["angles"]   = lmp_angle_list
        renderdict["torsions"] = lmp_torsion_list

        renderdict["atom_number"]    = atom_count
        renderdict["bond_number"]    = bond_count
        renderdict["angle_number"]   = angle_count
        renderdict["torsion_number"] = torsion_count
            
        # Write LAMMPS data file
        os.makedirs( os.path.dirname(data_path), exist_ok=True )

        with open(data_template) as file_:
            template = Template(file_.read())
            
        rendered = template.render( renderdict )

        with open(data_path, "w") as fh:
            fh.write(rendered)

        return
    
    def get_shake_indices(self, shake_dict: Dict[str,List[str]]={"atoms":[],"bonds":[],"angles":[]}):
        """
        Function that get the unique type identifier for atoms, bonds or angles that should be constrained using SHAKE.

        Args:
            shake_dict (Dict[str,List[str]], optional): Dictionary with force field keys that should be constrained. Defaults to {"atoms":[],"bonds":[],"angles":[]}.

        Returns:
            {"t":[], "b":[], "a":[]}: LAMMPS compatible dict with constraint types per section (t: types, b: bonds, a:angles)
        """
        atom_paras  = zip(self.atom_numbers_ges, self.nonbonded)
        bond_paras  = zip(self.bond_numbers_ges, self.bonds)
        angle_paras = zip(self.angle_numbers_ges, self.angles)

        # Search the index of the given force field types
        key_at  = [item for sublist in [[a[0] for a in atom_paras if a_key == a[1]["name"]] for a_key in shake_dict["atoms"]] for item in sublist]
        key_b   = [item for sublist in [[a[0] for a in bond_paras if a_key == a[1]["list"]] for a_key in shake_dict["bonds"]] for item in sublist]
        key_an  = [item for sublist in [[a[0] for a in angle_paras if a_key == a[1]["list"]] for a_key in shake_dict["angles"]] for item in sublist]

        return {"t":key_at, "b":key_b, "a":key_an}
    
## External functions for LAMMPS 

def write_lammps_ff( ff_template: str, lammps_ff_path: str, potential_kwargs: Dict[str,List[str]],
                     atom_numbers_ges: List[int], nonbonded: List[Dict[str,str]], 
                     bond_numbers_ges: List[int], bonds: List[Dict[str,str]],
                     angle_numbers_ges: List[int], angles: List[Dict[str,str]],
                     torsion_numbers_ges: List[int], torsions: List[Dict[str,str]],
                     only_self_interactions: bool=True, mixing_rule: str = "arithmetic",
                     ff_kwargs: Dict[str,Any]={}, n_eval: int=1000) -> str:
    """
    This function prepares LAMMPS understandable force field interactions and writes them to a file.

    Parameters:
     - ff_template (str, optional): Path to the force field template for LAMMPS
     - lammps_ff_path (str, optional): Destination of the external force field file. 
     - potential_kwargs (Dict[str,List[str]]): Dictionary that contains the LAMMPS arguments for every pair style that is used.
     - atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
     - nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: vdw_style, coul_style, sigma, epsilon, name, m 
     - bond_numbers_ges (List[int]): List with the unique bond type identifiers used in LAMMPS
     - bonds (List[Dict[str,str]]): List with the unique bonds dictionaries containing
     - angle_numbers_ges (List[int]): List with the unique angle type identifiers used in LAMMPS
     - angles (List[Dict[str,str]]): List with the unique angles dictionaries containing
     - torsion_numbers_ges (List[int]): List with the unique dihedral type identifiers used in LAMMPS
     - torsions (List[Dict[str,str]]): List with the unique dihedral dictionaries containing
     - only_self_interactions (bool, optional): If only self interactions should be written and LAMMPS does the mixing. Defaults to "True".
     - mixing_rule (str, optional): In case this function should do the mixing, the used mixing rule. Defaultst to "arithmetic"
     - ff_kwargs (Dict[str,Any], optional): Parameter kwargs, which are directly parsed to the template. Defaults to {}.
     - n_eval (int, optional): Number of spline interpolations saved if tabled bond is used. Defaults to 1000.

    Returns:
     -  lammps_ff_path (str): Destination of the external force field file.
    
    Raises:
        FileExistsError: If the force field template do not exist.
    """

    # Define arugment mapping 
    ARGUMENT_MAP = { }

    if not os.path.exists(ff_template):
        raise FileExistsError(f"Template file does not exists:\n  {ff_template}")
    
    # Dictionary to render the template. Pass are keyword arguments directly in the template.
    renderdict = { **ff_kwargs, "charged": not all( charge==0 for charge in np.unique([p["charge"] for p in nonbonded]) ) }

    # Van der Waals and Coulomb pair interactions
    vdw_interactions     = []
    coulomb_interactions = []
    pair_hybrid_flag = ( len( np.unique( [a["vdw_style"] for a in nonbonded ] ) ) + len( np.unique( [a["coulomb_style"] for a in nonbonded if a["coulomb_style"]] ) )) > 1

    for i,iatom in zip(atom_numbers_ges, nonbonded):
        for j,jatom in zip(atom_numbers_ges[i-1:], nonbonded[i-1:]):
            
            # Skip mixing pair interactions in case self only interactions are wanted
            if only_self_interactions and i != j:
                continue

            name_ij   = f"{iatom['name']}  {jatom['name']}"

            if mixing_rule == "arithmetic": 
                sigma_ij   = ( iatom["sigma"] + jatom["sigma"] ) / 2
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif mixing_rule ==  "geometric":
                sigma_ij   = np.sqrt( iatom["sigma"] * jatom["sigma"] )
                epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

            elif mixing_rule ==  "sixthpower": 
                sigma_ij   = ( 0.5 * ( iatom["sigma"]**6 + jatom["sigma"]**6 ) )**( 1 / 6 ) 
                epsilon_ij = 2 * np.sqrt( iatom["epsilon"] * jatom["epsilon"] ) * iatom["sigma"]**3 * jatom["sigma"]**3 / ( iatom["sigma"]**6 + jatom["sigma"]**6 )
            
            else:
                raise KeyError(f"Specified mixing rule is not implemented: '{mixing_rule}'. Valid options are: 'arithmetic', 'geometric', and 'sixthpower'")
            
            n_ij  = ( iatom["n"] + jatom["n"] ) / 2
            m_ij  = ( iatom["m"] + jatom["m"] ) / 2

            ARGUMENT_MAP["sigma"]   = np.round(sigma_ij, 4)
            ARGUMENT_MAP["epsilon"] = np.round(epsilon_ij, 4)
            ARGUMENT_MAP["n"]       = n_ij
            ARGUMENT_MAP["m"]       = m_ij

            if iatom["vdw_style"] != jatom["vdw_style"]:
                raise KeyError(f"Atom '{i}' and atom '{j}' has different vdW pair style:\n  {i}: {iatom['vdw_style']}\n  {j}: {jatom['vdw_style']}")
            
            if pair_hybrid_flag:
                vdw_interactions.append( [ i, j, iatom["vdw_style"] ] + [ ARGUMENT_MAP[arg] for arg in potential_kwargs[iatom["vdw_style"]] ] + [ "#", name_ij ] ) 
            else:
                vdw_interactions.append( [ i, j ] + [ ARGUMENT_MAP[arg] for arg in potential_kwargs[iatom["vdw_style"]] ] + [ "#", name_ij ] ) 

            # If the system is charged, add Coulomb interactions
            if renderdict["charged"]:
                if iatom["charge"] == 0.0 or jatom["charge"] == 0.0:
                    continue
                if iatom["coulomb_style"] != jatom["coulomb_style"]:
                    raise KeyError(f"Atom '{i}' and atom '{j}' has different Coulomb pair style:\n  {i}: {iatom['coulomb_style']}\n  {j}: {jatom['coulomb_style']}")
                
                if pair_hybrid_flag:
                    coulomb_interactions.append( [ i, j, iatom["coulomb_style"] ] + [ ARGUMENT_MAP[arg] for arg in potential_kwargs[iatom["coulomb_style"]] ] + [ "#", name_ij ] )
                else:
                    coulomb_interactions.append( [ i, j ] + [ ARGUMENT_MAP[arg] for arg in potential_kwargs[iatom["coulomb_style"]] ] + [ "#", name_ij ] )

    renderdict["vdw_interactions"] = vdw_interactions

    # Overwrite Coulomb pair interactions if there is only one style
    if len( np.unique( [a["coulomb_style"] for a in nonbonded if a["coulomb_style"]] ) ) == 1:
        if pair_hybrid_flag:
            coulomb_interactions = [ [ "*", "*", np.unique( [a["coulomb_style"] for a in nonbonded if a["coulomb_style"] ] )[0] ] ]
        else:
            coulomb_interactions = [ [ "*", "*" ] ]
    
    renderdict["coulomb_interactions"] = coulomb_interactions

    # Bonded interactions
    bond_paras = []
    bond_styles = np.unique( [a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"] for a in bonds ] ).tolist()
    bond_hybrid_flag = len( np.unique( [a["style"] for a in bonds ] ) ) > 1

    for bond_type, bond in zip(bond_numbers_ges, bonds):
        if bond_hybrid_flag:
            bond_paras.append( [ bond_type, bond["style"], *bond["p"], "#", *bond["list"] ] )
        else:
            bond_paras.append( [ bond_type, *bond["p"], "#", *bond["list"] ] )

    renderdict["bond_paras"]  = bond_paras 
    renderdict["bond_styles"] = ["hybrid"] + bond_styles if bond_hybrid_flag else bond_styles

    # Angle interactions
    angle_paras = []
    angle_styles = np.unique( [a["style"] for a in angles] ).tolist()
    angle_hybrid_flag = len( np.unique( [a["style"] for a in angles ] ) ) > 1

    for angle_type, angle in zip(angle_numbers_ges, angles):
        if angle_hybrid_flag:
            angle_paras.append( [ angle_type, angle["style"], *angle["p"], "#", *angle["list"] ] )
        else:
            angle_paras.append( [ angle_type, *angle["p"], "#", *angle["list"] ] )

    renderdict["angle_paras"]  = angle_paras 
    renderdict["angle_styles"] = ["hybrid"] + angle_styles if angle_hybrid_flag else angle_styles

    # Dihedral interactions
    torsion_paras = []
    torsion_styles = np.unique( [a["style"] for a in torsions] ).tolist()
    torsion_hybrid_flag = len( np.unique( [a["style"] for a in torsions ] ) ) > 1

    for torsion_type, torsion in zip(torsion_numbers_ges, torsions):
        if torsion_hybrid_flag:
            torsion_paras.append( [ torsion_type, torsion["style"], *torsion["p"], "#", *torsion["list"] ] )
        else:
            torsion_paras.append( [ torsion_type, *torsion["p"], "#", *torsion["list"] ] )

    renderdict["torsion_paras"]  = torsion_paras 
    renderdict["torsion_styles"] = ["hybrid"] + torsion_styles if torsion_hybrid_flag else torsion_styles

    # If provided write pair interactions as external LAMMPS force field file.
    with open(ff_template) as file_:
        template = Template( file_.read() )
    
    rendered = template.render( renderdict )

    os.makedirs( os.path.dirname( lammps_ff_path ), exist_ok = True  )

    with open(lammps_ff_path, "w") as fh:
        fh.write(rendered) 

    return lammps_ff_path