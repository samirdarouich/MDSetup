import os
import toml
import json
import yaml
import numpy as np
import moleculegraph
from jinja2 import Template
from typing import Dict, List, Any

from ..tools.general import merge_nested_dicts, flatten_list, get_system_volume

## LAMMPS molecule class

#### to do:
# include check if playmol defined atom name matches expected atom ff type 

class LAMMPS_molecules():
    """
    This class writes LAMMPS data input for arbitrary mixtures using moleculegraph.
    """
    def __init__(self, mol_str: List[str], force_field_paths: List[str] ):

        # Save moleclue graphs of both components class wide
        self.mol_str    = mol_str
        self.mol_list   = [ moleculegraph.molecule(mol) for mol in self.mol_str ]

        # Read in force field files
        self.ff = {}
        for force_field_path in force_field_paths:
            if ".yaml" in force_field_path:
                data = yaml.safe_load( open(force_field_path) )
            elif ".json" in force_field_path:
                data = json.load( open(force_field_path) )
            elif ".toml" in force_field_path:
                data = toml.load( open(force_field_path) )
            else:
                raise KeyError(f"Force field file is not supported: '{force_field_path}'. Please provide 'YAML', 'JSON', or 'TOML' file.")
            # Update overall dict
            merge_nested_dicts( self.ff, data.copy() )

        ## Map force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions) ##

        # Get (unique) atom types and parameters
        self.nonbonded =  flatten_list( molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in self.mol_list ) 
        
        # Get (unique) bond types and parameters
        self.bonds     =  flatten_list( molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in self.mol_list ) 
        
        # Get (unique) angle types and parameters
        self.angles    =  flatten_list( molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in self.mol_list ) 
        
        # Get (unique) torsion types and parameters 
        self.torsions  =  flatten_list( molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in self.mol_list ) 
        
        if not all( [ all(self.nonbonded), all(self.bonds), all(self.angles), all(self.torsions) ] ):
            txt = "nonbonded" if not all(self.nonbonded) else "bonds" if not all(self.bonds) else "angles" if not all(self.angles) else "torsions"
            raise ValueError("Something went wrong during the force field mapping for key: %s"%txt)
        
        # Get nonbonded force field for all atom types not only the unique one. This is later used to extract the charge of each atom in the system while writing the LAMMPS data file.
        self.ff_all    = [ molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) for molecule in self.mol_list ]


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
        add_atoms = [1] + [ sum(len(mol.unique_atom_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        # Get the force field type identifiers for each atom. This is used in the #Atoms section in the data file.
        # Therefore take the identifier number of every atom in each moleculegraph and add the preceeding number of atom types.
        # 1 1 1 0.404 -10.69874 -8.710742 1.59434903 # cH_alcohol1 --> First index is a running LAMMPS number for the atom. The 2nd index is the running numner for the molecule.
        # The third index is the force field type identifier (here it is 1, thus it is type the cH_alcohol type, as defined in the input file header)
        # The forth number is the partial charge. The following 3 numbers are the x, y and z coordinate of the atom in the system.
        self.atoms_running_number = [ mol.unique_atom_inverse + add_atoms[i] for i,mol in enumerate(self.mol_list) ]
        
        # These are the !unique! atom force field types from "atoms_running_number". This is used in the atom definition section of the data file.
        # As just the unique atom force field types are defined in LAMMPS. 
        self.atom_numbers_ges = np.unique( flatten_list( self.atoms_running_number ) )

        # Get the number of atoms per component. This will later be multiplied with the number of molecules per component to get the total number of atoms in the system
        self.number_of_atoms = [ mol.atom_number for mol in self.mol_list ]

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
        self.bonds_running_number = [ mol.unique_bond_inverse + add_bonds[i] for i,mol in enumerate(self.mol_list) ]

        # These are the !unique! bond force field types from "bonds_running_number". This is used in the bonds definition section of the data file.
        # As just the unique bond force field types are defined in LAMMPS. 
        self.bond_numbers_ges = np.unique( flatten_list( self.bonds_running_number ) )
        
        # This identify the atoms of each bond in the molecule. This is used in #Bonds section, where each Atom is assigned a bond, as well as the bond force field type. 
        # To these indicies the number of preceeding atoms in the system will be added (in the write_lammps_data function).
        # Thus the direct list from molecule graph can be used, without further refinement. 
        # (E.g.: Bond_list for ethanediol: [ [1,2], [2,3], [3,4], [4,5], [5,6] ] )
        self.bond_numbers = [ mol.bond_list + 1 for mol in self.mol_list ]

        # This is just the name of each the bonds defined above. This is written as information that one knows what which bond is defined in the data file.
        self.bond_names = [ [ [mol.atom_names[i] for i in bl] for bl in mol.bond_list ] for mol in self.mol_list ]

        # Get the number of bonds per component. This will later be multiplied with the number of molecules per component to get the total number of bonds in the system
        self.number_of_bonds = [ len(mol.bond_keys) for mol in self.mol_list ]


        ## Definitions for angles in the system --> (all elements here have the same meaning as above for bonds just for angles) ##
        
        add_angles = [1] + [ sum(len(mol.unique_angle_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        self.angles_running_number = [ mol.unique_angle_inverse + add_angles[i] for i,mol in enumerate(self.mol_list) ]
        self.angle_numbers_ges = np.unique( flatten_list( self.angles_running_number ) )
        self.angle_numbers = [ mol.angle_list + 1 for mol in self.mol_list ]
        self.angle_names  = [ [ [mol.atom_names[i] for i in al] for al in mol.angle_list ] for mol in self.mol_list ]
        self.number_of_angles = [ len(mol.angle_keys) for mol in self.mol_list ]


        ## Definitions for torsions in the system --> (all elements here have the same meaning as above for bonds just for torsions) ##
        
        add_torsions = [1] + [ sum(len(mol.unique_torsion_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]
        
        self.torsions_running_number = [ mol.unique_torsion_inverse + add_torsions[i] for i,mol in enumerate(self.mol_list) ]
        self.torsion_numbers_ges = np.unique( flatten_list( self.torsions_running_number ) )
        self.torsion_numbers = [ mol.torsion_list + 1 for mol in self.mol_list ]
        self.torsion_names = [ [ [mol.atom_names[i] for i in tl] for tl in mol.torsion_list ] for mol in self.mol_list ]
        self.number_of_torsions = [ len(mol.torsion_keys) for mol in self.mol_list ]
        
        return

    def write_lammps_data( self, xyz_path: str, data_template: str, data_path: str,
                           nmol_list: List[int], density: float, box_type: str="cubic",
                           z_x_relation: float=1.0, z_y_relation: float=1.0 ):
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
        self.total_number_of_atoms = np.dot( self.number_of_atoms, nmol_list )

        # Total bonds in system 
        self.total_number_of_bonds = np.dot( self.number_of_bonds, nmol_list )
        
        # Total angles in system 
        self.total_number_of_angles = np.dot( self.number_of_angles, nmol_list) 
        
        # Total torsions in system 
        self.total_number_of_torsions = np.dot( self.number_of_torsions, nmol_list )

        
        # Get the box dimensions
        molar_masses = [ sum( a["mass"] for a in mol_nb ) for mol_nb in self.ff_all ]

        box_dimensions = get_system_volume( molar_masses = molar_masses, molecule_numbers = nmol_list, 
                                            density = density, box_type = box_type, 
                                            z_x_relation = z_x_relation, z_y_relation = z_y_relation )

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

        for m,mol in enumerate(self.mol_list):

            ## Now write LAMMPS input for every molecule of each component ##

            for mn in range(nmol_list[m]):
                
                # The atom index in the bond, angle, torsion list starts always at 1. Thus to write the correct atoms for each bond, angle, torsion one need to 
                # know the indcies of the atoms of the current molecule. To do so, add the dot product of all atom numbers and molecules of each component
                # to the atom index in the current bond, angle, torsion.
                add_atom_count = np.dot( mol_count, self.number_of_atoms ).astype("int")

                # Define atoms
                for atomtype,ff_atom in zip( self.atoms_running_number[m], self.ff_all[m] ):
                    
                    atom_count +=1

                    # LAMMPS INPUT: total n° of atom in system, mol n° in system, atomtype, partial charges,coordinates
                    line = [ atom_count, sum(mol_count)+1, atomtype, ff_atom["charge"], *coordinates[atom_count-1]["xyz"], "#", coordinates[atom_count-1]["atom"] ]

                    lmp_atom_list.append(line)


                # Define bonds 
                for bondtype,bond,bond_name in zip( self.bonds_running_number[m], self.bond_numbers[m], self.bond_names[m] ):

                    bond_count += 1

                    # Higher the indices of the bond to match the atoms of the current molecule.
                    dummy = bond + add_atom_count
      
                    # LAMMPS INPUT: total n° of bond in system, bond force field type, atom n° in this bond
                    line  = [ bond_count, bondtype, *dummy, "#", " ".join(bond_name) ]

                    lmp_bond_list.append(line)

                # Define angles
                for angletype,angle,angle_name in zip (self.angles_running_number[m], self.angle_numbers[m], self.angle_names[m] ):

                    angle_count += 1

                    # Higher the indices of the angle to match the atoms of the current molecule.
                    dummy = angle + add_atom_count

                    # LAMMPS INPUT: total n° of angles in system, angle force field type, atom n° in this angle
                    line = [ angle_count, angletype, *dummy, "#", " ".join(angle_name) ]

                    lmp_angle_list.append(line)

                # Define torsions
                for torsiontype,torsion,torsion_name in zip( self.torsions_running_number[m], self.torsion_numbers[m], self.torsion_names[m] ):

                    torsion_count += 1

                    # Higher the indices of the dihedral to match the atoms of the current molecule.
                    dummy = torsion + add_atom_count

                    # LAMMPS INPUT: total n° of torsions in system, torsion force field type, atom n° in this torsion
                    line = [ torsion_count, torsiontype, *dummy, "#", " ".join(torsion_name) ]

                    lmp_torsion_list.append(line)

                # Increase the molecule count of the current component by one.
                mol_count[m] += 1

        renderdict = { **box_dimensions,
                       "atom_type_number": len( self.atom_numbers_ges ),
                       "bond_type_number": len( self.bond_numbers_ges ),
                       "angle_type_number": len( self.angle_numbers_ges ),
                       "torsion_type_number": len( self.torsion_numbers_ges ),
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
    
    def get_shake_indices(self, shake_dict: Dict[str,List[List[str]]]={"atoms":[],"bonds":[],"angles":[]}):
        """
        Function that get the unique type identifier for atoms, bonds or angles that should be constrained using SHAKE.

        Args:
            shake_dict (Dict[str,List[str]], optional): Dictionary with force field keys that should be constrained. Defaults to {"atoms":[],"bonds":[],"angles":[]}.

        Returns:
            {"t":[], "b":[], "a":[]}: LAMMPS compatible dict with constraint types per section (t: types, b: bonds, a:angles)
        """
        atom_dict = { f["name"]: n for n,f in zip(self.atom_numbers_ges, self.nonbonded) }
        bond_dict = { "_".join(f["list"]): n for n,f in zip(self.bond_numbers_ges, self.bonds) }
        angle_dict = { "_".join(f["list"]): n for n,f in zip(self.angle_numbers_ges, self.angles) }

        # Search the index of the given force field types
        key_tt  = sorted( flatten_list( atom_dict.get( "".join(t_key), [] ) for t_key in shake_dict["atoms"] ) )
        key_bt  = sorted( flatten_list( bond_dict.get( "_".join(b_key), [] ) for b_key in shake_dict["bonds"] ) )
        key_at  = sorted( flatten_list( angle_dict.get( "_".join(a_key), [] ) for a_key in shake_dict["angles"] ) )

        return {"t":key_tt, "b":key_bt, "a":key_at}



## External functions for LAMMPS 

def get_pair_style( local_attributes: Dict[str,Any], vdw_pair_styles: List[str], 
                    coul_pair_styles: List[str], pair_style_kwargs: Dict[str,str] ):
    """
    This function takes in several parameters and returns a string representing the combined pair style for a molecular simulation.

    Parameters:
    - local_attributes (Dict[str,Any]): A dictionary containing local attributes for the pair style.
    - vdw_pair_styles (List[str]): A list of strings representing the Van der Waals pair styles to be used.
    - coul_pair_styles (List[str]): A list of strings representing the Coulombic pair styles to be used.
    - pair_style_kwargs (Dict[str,str]): A dictionary mapping pair styles to their corresponding arguments.

    Returns:
    - A string representing the combined pair style for the simulation.

    The function iterates over the unique Van der Waals pair styles and Coulombic pair styles provided. For each pair style, it constructs a substring 
    by concatenating the pair style name with the corresponding arguments from the local_attributes dictionary. These substrings are then appended to the combined_pair_style list.

    If multiple pair styles are used, the function inserts the "hybrid/overlay" style at the beginning of the combined_pair_style list.

    Finally, the function returns the combined pair style as a string, with each pair style separated by two spaces.

    Note: The function assumes that the local_attributes dictionary contains all the necessary arguments for each pair style specified 
          in vdw_pair_styles and coul_pair_styles. If an argument is missing, a KeyError will be raised.
    """
    combined_pair_style = []

    for vdw_pair_style in set(vdw_pair_styles):
        sub_string = f"{vdw_pair_style} " + ' '.join( [ str(local_attributes[arg]) for arg in pair_style_kwargs[vdw_pair_style] ] )
        combined_pair_style.append( sub_string )

    for coul_pair_style in set(coul_pair_styles):
        sub_string = f"{coul_pair_style} " + ' '.join( [ str(local_attributes[arg]) for arg in pair_style_kwargs[coul_pair_style] ] )
        combined_pair_style.append( sub_string )

    # Add hybrid/overlay style in case several styles are used
    if len(combined_pair_style) > 1:
        combined_pair_style.insert( 0, "hybrid/overlay" )

    return "  ".join( combined_pair_style )

def get_mixed_parameter( sigma_i: float, sigma_j: float, epsilon_i: float, epsilon_j: float, 
                         mixing_rule: str="arithmetic", precision: int=4 ):
    """
    Calculate the mixed parameters for a pair of interacting particles.

    Parameters:
        sigma_i (float): The sigma parameter of particle i.
        sigma_j (float): The sigma parameter of particle j.
        epsilon_i (float): The epsilon parameter of particle i.
        epsilon_j (float): The epsilon parameter of particle j.
        mixing_rule (str, optional): The mixing rule to use. Valid options are "arithmetic", "geometric", and "sixthpower". Defaults to "arithmetic".
        precision (int, optional): The number of decimal places to round the results to. Defaults to 4.

    Returns:
        tuple: A tuple containing the mixed sigma and epsilon parameters.

    Raises:
        KeyError: If the specified mixing rule is not implemented.

    Example:
        >>> get_mixed_parameter(3.5, 2.5, 0.5, 0.8, mixing_rule="arithmetic", precision=3)
        (3.0, 0.632)
    """
    
    if mixing_rule == "arithmetic": 
        sigma_ij   = ( sigma_i + sigma_j ) / 2
        epsilon_ij = np.sqrt( epsilon_i * epsilon_j )

    elif mixing_rule ==  "geometric":
        sigma_ij   = np.sqrt( sigma_i * sigma_j )
        epsilon_ij = np.sqrt( epsilon_i * epsilon_j )

    elif mixing_rule ==  "sixthpower": 
        sigma_ij   = ( 0.5 * ( sigma_i**6 + sigma_j**6 ) )**( 1 / 6 ) 
        epsilon_ij = 2 * np.sqrt( epsilon_i * epsilon_j ) * sigma_i**3 * sigma_j**3 / ( sigma_i**6 + sigma_j**6 )
    
    else:
        raise KeyError(f"Specified mixing rule is not implemented: '{mixing_rule}'. Valid options are: 'arithmetic', 'geometric', and 'sixthpower'")
    
    return np.round(sigma_ij, precision), np.round(epsilon_ij, precision)

def get_bonded_style( bonded_numbers: List[int], bonded_dict: List[Dict[str,Any]], n_eval: int=1000 ):
    """
    Get bonded styles and parameters for a given list of bonded numbers and bonded dictionary for LAMMPS ff input. 
    This can be used for bonds, angles, dihedrals, etc...

    Parameters:
    - bonded_numbers (List[int]): A list of integers representing the bonded numbers.
    - bonded_dict (List[Dict[str, Any]]): A list with dictionaries containing the bonded styles and parameters.
    - n_eval (int, optional): The number of evaluations. Default is 1000.

    Returns:
    - bonded_styles (List[str]): A list of unique bonded styles.
    - bonded_paras (List[List]): A list of bonded parameters.

    """
    bonded_styles = list( { a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"] for a in bonded_dict } )
    hybrid_flag = len(bonded_styles) > 1

    bonded_paras = [ ]

    for bonded_type, bonded in zip(bonded_numbers, bonded_dict):
        bonded_paras.append( [ bonded_type ] + 
                             ( [ bonded["style"]] if hybrid_flag else [] ) +
                             [ *bonded["p"], "#", *bonded["list"] ] 
                            )

    if hybrid_flag:
        bonded_styles.insert( 0, "hybrid" )

    return bonded_styles, bonded_paras


