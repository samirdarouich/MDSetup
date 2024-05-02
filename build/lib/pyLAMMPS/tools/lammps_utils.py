import os
import toml
import json
import yaml
import numpy as np
import moleculegraph
from jinja2 import Template
from typing import Dict, List, Any

from .general_utils import merge_nested_dicts, flatten_list, get_system_volume

## LAMMPS molecule class

#### to do:
# include check if playmol defined atom name matches expected atom ff type 
# data add '#' and change template to use join

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
                    line = [ atom_count, sum(mol_count)+1, atomtype, ff_atom["charge"], *coordinates[atom_count-1]["xyz"], coordinates[atom_count-1]["atom"] ]

                    lmp_atom_list.append(line)


                # Define bonds 
                for bondtype,bond,bond_name in zip( self.bonds_running_number[m], self.bond_numbers[m], self.bond_names[m] ):

                    bond_count += 1

                    # Higher the indices of the bond to match the atoms of the current molecule.
                    dummy = bond + add_atom_count
      
                    # LAMMPS INPUT: total n° of bond in system, bond force field type, atom n° in this bond
                    line  = [ bond_count, bondtype, *dummy, " ".join(bond_name) ]

                    lmp_bond_list.append(line)

                # Define angles
                for angletype,angle,angle_name in zip (self.angles_running_number[m], self.angle_numbers[m], self.angle_names[m] ):

                    angle_count += 1

                    # Higher the indices of the angle to match the atoms of the current molecule.
                    dummy = angle + add_atom_count

                    # LAMMPS INPUT: total n° of angles in system, angle force field type, atom n° in this angle
                    line = [ angle_count, angletype, *dummy, " ".join(angle_name) ]

                    lmp_angle_list.append(line)

                # Define torsions
                for torsiontype,torsion,torsion_name in zip( self.torsions_running_number[m], self.torsion_numbers[m], self.torsion_names[m] ):

                    torsion_count += 1

                    # Higher the indices of the dihedral to match the atoms of the current molecule.
                    dummy = torsion + add_atom_count

                    # LAMMPS INPUT: total n° of torsions in system, torsion force field type, atom n° in this torsion
                    line = [ torsion_count, torsiontype, *dummy, " ".join(torsion_name) ]

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

def get_coupling_lambdas( combined_lambdas: List[float], coupling: bool = True, precision: int=3 ):
    """
    Calculate the van der Waals (vdW) and Coulomb coupling lambdas.

    Parameters:
    - combined_lambdas (List[float]): A list of combined lambdas.
    - coupling (bool, optional): Whether to calculate coupling or decoupling lambdas. Defaults to true.
    - precision (int, optional): The precision of the lambdas (default is 3).

    Returns:
    - vdw_lambdas (List[str]): A list of formatted vdW lambdas.
    - coul_lambdas (List[str]): A list of formatted Coulomb lambdas.

    The vdW lambdas are calculated as the minimum of each combined lambda and 1.0, rounded to the specified precision.
    The Coulomb lambdas are calculated as the maximum of each combined lambda minus 1.0 and 0.0, rounded to the specified precision.
    If coupling is False, the vdW lambdas are calculated as 1 minus the maximum of each combined lambda minus 1.0, rounded to the specified precision.
    If coupling is False, the Coulomb lambdas are calculated as 1 minus the minimum of each combined lambda and 1.0, rounded to the specified precision.
    The Coulomb lambdas are adjusted to be at least 1e-9 to avoid division by 0.
    """

    # Coulomb lambdas need to be at least 1e-9 to avoid division by 0
    if coupling:
        vdw_lambdas = [ f"{min(l,1.0):.{precision}f}" for l in combined_lambdas] 
        coul_lambdas = [ f"{max(l-1,0):.{precision}f}".replace(f"{0:.{precision}f}","1e-9") for l in combined_lambdas ] 
    else:
        vdw_lambdas = [ f"{1-max(l-1,0.0):.{precision}f}" for l in combined_lambdas] 
        coul_lambdas = [ f"{1-min(l,1.0):.{precision}f}".replace(f"{0:.{precision}f}","1e-9") for l in combined_lambdas ]
    
    return vdw_lambdas, coul_lambdas

def write_fep_sampling( fep_template: str, fep_outfile: str, combined_lambdas: List[float], 
                        charge_list: List[List[int|float]], current_state: int, 
                        precision: int=3, coupling: bool=True, kwargs: Dict[str,Any]={} ):
    """
    Write FEP sampling.

    This function takes in various parameters to generate a free energy sampling file based on a provided FEP sampling template.

    Parameters:
    - fep_template (str): The path to the FEP sampling template file.
    - fep_outfile (str): The path to the output file where the generated sampling file will be saved.
    - combined_lambdas (List[float]): A list of combined lambda values.
    - charge_list (List[List[int|float]]): A list of charge lists, where each charge list contains the charges for each atom in the system.
    - current_state (int): The index of the current state in the combined_lambdas list.
    - precision (int, optional): The precision of the lambda values. Defaults to 3.
    - coupling (bool, optional): Whether to use coupling or decoupling lambdas. Defaults to true.
    - kwargs (Dict[str,Any], optional): Additional keyword arguments to be passed to the template rendering. Defaults to {}.

    Returns:
    - str: The path to the generated FEP sampling file.

    Raises:
    - FileExistsError: If the provided FEP sampling template file does not exist.

    """

    if not os.path.exists( fep_template ):
        raise FileExistsError(f"Provided FEP sampling template does not exist!: '{fep_template}'")
    
    # Produce free energy sampling file
    with open( fep_template ) as f:
        template = Template( f.read() )
    
    lambda_vdw, lambda_coul = get_coupling_lambdas( combined_lambdas, coupling, precision )

    lambda_states = [ (float(l_vdw),float(l_coul)) for l_vdw,l_coul in zip( lambda_vdw, lambda_coul ) ]
    lambda_state = lambda_states[current_state]

    renderdict = { "no_intermediates": len(combined_lambdas),
                   "charge_list": charge_list,
                   "init_lambda_state": current_state + 1,
                   "charged": any(np.array(charge_list)[:,1]),
                   "lambda_states": lambda_states,
                   "current_lambda_state": lambda_state,
                    **kwargs  
                    }

    os.makedirs( os.path.dirname( fep_outfile ), exist_ok = True )
    
    with open( fep_outfile, "w" ) as f:
        f.write( template.render( renderdict ) )
    
    return fep_outfile

def write_lammps_ff( ff_template: str, lammps_ff_path: str, potential_kwargs: Dict[str,List[str]],
                     atom_numbers_ges: List[int], nonbonded: List[Dict[str,str]], 
                     bond_numbers_ges: List[int], bonds: List[Dict[str,str]],
                     angle_numbers_ges: List[int], angles: List[Dict[str,str]],
                     torsion_numbers_ges: List[int], torsions: List[Dict[str,str]],
                     shake_dict: Dict[str,List[str|int]]={"t":[], "b":[], "a":[]},
                     only_self_interactions: bool=True, mixing_rule: str = "arithmetic",
                     ff_kwargs: Dict[str,Any]={}, n_eval: int=1000) -> str:
    """
    This function prepares LAMMPS understandable force field interactions and writes them to a file.

    Parameters:
     - ff_template (str, optional): Path to the force field template for LAMMPS
     - lammps_ff_path (str, optional): Destination of the external force field file. 
     - potential_kwargs (Dict[str,List[str]]): Dictionary that contains the LAMMPS arguments for every pair style that is used. 
                                               Should contain 'pair_style', 'vdw_style' and 'coulomb_style' key.
     - atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
     - nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: vdw_style, coul_style, sigma, epsilon, name, m 
     - bond_numbers_ges (List[int]): List with the unique bond type identifiers used in LAMMPS
     - bonds (List[Dict[str,str]]): List with the unique bonds dictionaries containing
     - angle_numbers_ges (List[int]): List with the unique angle type identifiers used in LAMMPS
     - angles (List[Dict[str,str]]): List with the unique angles dictionaries containing
     - torsion_numbers_ges (List[int]): List with the unique dihedral type identifiers used in LAMMPS
     - torsions (List[Dict[str,str]]): List with the unique dihedral dictionaries containing
     - shake_dict (Dict[str,List[str|int]]): Shake dictionary containing the unique atom,bond,angle identifier for SHAKE algorithm.
                                             Defaults to {"t":[],"b":[],"a":[]}.
     - only_self_interactions (bool, optional): If only self interactions should be written and LAMMPS does the mixing. Defaults to "True".
     - mixing_rule (str, optional): In case this function should do the mixing, the used mixing rule. Defaultst to "arithmetic"
     - ff_kwargs (Dict[str,Any], optional): Parameter kwargs, which are directly parsed to the template. Defaults to {}.
     - n_eval (int, optional): Number of spline interpolations saved if tabled bond is used. Defaults to 1000.

    Returns:
     -  lammps_ff_path (str): Destination of the external force field file.
    
    Raises:
        FileExistsError: If the force field template do not exist.
    """

    if not os.path.exists(ff_template):
        raise FileExistsError(f"Template file does not exists:\n  {ff_template}")
    
    if not all(key in potential_kwargs for key in ["pair_style", "vdw_style", "coulomb_style"]):
        key = 'pair_style' if not 'pair_style' in potential_kwargs else 'vdw_style' if not 'vdw_style' in potential_kwargs else 'coulomb_style'
        raise KeyError(f"{key} not in keys of the potential kwargs! Check nonbonded input!")
    
    # Dictionary to render the template. Pass ff keyword arguments directly in the template.
    renderdict = { **ff_kwargs, "charged": any( p["charge"] != 0 for p in nonbonded ),
                   "shake_dict": shake_dict 
                 }

    # Define cut of radius (scanning the force field and taking the highest value)
    rcut = max( map( lambda p: p["cut"], nonbonded ) )

    # Get pair style defined in force field
    local_variables = locals()

    # Get pair style defined in force field
    vdw_pair_styles = list( { p["vdw_style"] for p in nonbonded if p["vdw_style"] } )
    coul_pair_styles = list( { p["coulomb_style"] for p in nonbonded if p["coulomb_style"] } )
    
    pair_style = get_pair_style( local_variables, vdw_pair_styles,
                                 coul_pair_styles, potential_kwargs["pair_style"] )

    renderdict["pair_style"] = pair_style

    pair_hybrid_flag = "hybrid" in pair_style

    # Van der Waals and Coulomb pair interactions
    vdw_interactions     = []
    coulomb_interactions = []
    
    for i,iatom in zip(atom_numbers_ges, nonbonded):
        for j,jatom in zip(atom_numbers_ges[i-1:], nonbonded[i-1:]):
            
            # Skip mixing pair interactions in case self only interactions are wanted
            if only_self_interactions and i != j:
                continue
            
            if iatom["vdw_style"] != jatom["vdw_style"]:
                raise KeyError(f"Atom '{i}' and atom '{j}' has different vdW pair style:\n  {i}: {iatom['vdw_style']}\n  {j}: {jatom['vdw_style']}")
            
            # Get mixed parameters
            name_ij = f"{iatom['name']}  {jatom['name']}"
            sigma_ij, epsilon_ij = get_mixed_parameter( sigma_i=iatom["sigma"], sigma_j=jatom["sigma"],
                                                        epsilon_i=iatom["epsilon"], epsilon_j=jatom["epsilon"],
                                                        mixing_rule=mixing_rule, precision=4 )
            
            # In case n and m are present get mixed exponents
            if "n" in iatom.keys() and "n" in jatom.keys():
                n_ij  = ( iatom["n"] + jatom["n"] ) / 2
                m_ij  = ( iatom["m"] + jatom["m"] ) / 2

            # Get local variables and map them with potential kwargs 
            local_variables = locals()
            
            vdw_interactions.append( [ i, j ] + 
                                     ( [ iatom["vdw_style"] ] if pair_hybrid_flag else [] ) + 
                                     [ local_variables[arg] for arg in potential_kwargs["vdw_style"][iatom["vdw_style"]] ] +
                                     [ "#", name_ij ] 
                                    ) 

            # If the system is charged, add Coulomb interactions
            if renderdict["charged"]:
                if iatom["charge"] == 0.0 or jatom["charge"] == 0.0:
                    continue
                if iatom["coulomb_style"] != jatom["coulomb_style"]:
                    raise KeyError(f"Atom '{i}' and atom '{j}' has different Coulomb pair style:\n  {i}: {iatom['coulomb_style']}\n  {j}: {jatom['coulomb_style']}")
                

                coulomb_interactions.append( [ i, j ] + 
                                             ( [ iatom["coulomb_style"] ] if pair_hybrid_flag else [] ) + 
                                             [ local_variables[arg] for arg in potential_kwargs["coulomb_style"][iatom["coulomb_style"]] ] +
                                             [ "#", name_ij ] 
                                            )
                
    # Overwrite Coulomb pair interactions if there is only one style
    if len( coul_pair_styles ) == 1:
        coulomb_interactions = [ [ "*", "*"] +
                                 ( coul_pair_styles if pair_hybrid_flag else [] )
                                ]

    renderdict["vdw_interactions"] = vdw_interactions
    renderdict["coulomb_interactions"] = coulomb_interactions

    ## Add all kind of bonded interactions

    # Bonded interactions
    bond_styles, bond_paras   = get_bonded_style( bond_numbers_ges, bonds, n_eval = n_eval )
    renderdict["bond_paras"]  = bond_paras 
    renderdict["bond_styles"] = bond_styles

    # Angle interactions
    angle_styles, angle_paras = get_bonded_style( angle_numbers_ges, angles )

    renderdict["angle_paras"]  = angle_paras 
    renderdict["angle_styles"] = angle_styles

    # Dihedral interactions
    torsion_styles, torsion_paras = get_bonded_style( torsion_numbers_ges, torsions )

    renderdict["torsion_paras"]  = torsion_paras 
    renderdict["torsion_styles"] = torsion_styles
    
    # If provided write pair interactions as external LAMMPS force field file.
    with open(ff_template) as file_:
        template = Template( file_.read() )
    
    rendered = template.render( renderdict )

    os.makedirs( os.path.dirname( lammps_ff_path ), exist_ok = True  )

    with open(lammps_ff_path, "w") as fh:
        fh.write(rendered) 

    return lammps_ff_path

def write_coupled_lammps_ff(ff_template: str, lammps_ff_path: str, potential_kwargs: Dict[str,List[str]],
                            solute_numbers: List[int], combined_lambdas: List[float],
                            coupling_potential: Dict[str,Any], coupling_soft_core: Dict[str,float],
                            atom_numbers_ges: List[int], nonbonded: List[Dict[str,str]], 
                            bond_numbers_ges: List[int], bonds: List[Dict[str,str]],
                            angle_numbers_ges: List[int], angles: List[Dict[str,str]],
                            torsion_numbers_ges: List[int], torsions: List[Dict[str,str]],
                            mixing_rule: str, 
                            shake_dict: Dict[str,List[str|int]]={"t":[], "b":[], "a":[]},
                            coupling: bool=True, precision: int=3, 
                            ff_kwargs: Dict[str,Any]={}, n_eval: int=1000) -> str:
    """
    This function prepares LAMMPS understandable force field interactions for coupling simulations and writes them to a file.

    Parameters:
     - ff_template (str, optional): Path to the force field template for LAMMPS
     - lammps_ff_path (str, optional): Destination of the external force field file. 
     - potential_kwargs (Dict[str,List[str]]): Dictionary that contains the LAMMPS arguments for every pair style that is used. 
                                               Should contain 'pair_style', 'vdw_style' and 'coulomb_style' key.
     - coupling_potential (Dict[str,Any]): Define the coupling potential that is used. Should contain 'vdw' and 'coulomb' key.
     - coupling_soft_core (Dict[str,float]): Soft core potential parameters.
     - solute_numbers (List[int]): List with unique force field type identifiers for the solute which is coupled/decoupled.
     - combined_lambdas (List[float]): Combined lambdas. Check "coupling" parameter for description.
     - atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
     - nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: vdw_style, coul_style, sigma, epsilon, name, m 
     - bond_numbers_ges (List[int]): List with the unique bond type identifiers used in LAMMPS
     - bonds (List[Dict[str,str]]): List with the unique bonds dictionaries containing
     - angle_numbers_ges (List[int]): List with the unique angle type identifiers used in LAMMPS
     - angles (List[Dict[str,str]]): List with the unique angles dictionaries containing
     - torsion_numbers_ges (List[int]): List with the unique dihedral type identifiers used in LAMMPS
     - torsions (List[Dict[str,str]]): List with the unique dihedral dictionaries containing
     - mixing_rule (str): Provide mixing rule. Defaults to "arithmetic"
     - shake_dict (Dict[str,List[str|int]]): Shake dictionary containing the unique atom,bond,angle identifier for SHAKE algorithm.
                                             Defaults to {"t":[],"b":[],"a":[]}.
     - coupling (bool, optional): If True, coupling is used (lambdas between 0 and 1 are vdW, 1 to 2 are Coulomb). 
                                  For decoupling (lambdas between 0 and 1 are Coulomb, 1 to 2 are vdW). Defaults to "True".
     - precision (int, optional): Precision of coupling lambdas. Defaults to 3.
     - ff_kwargs (Dict[str,Any], optional): Parameter kwargs, which are directly parsed to the template. Defaults to {}.
     - n_eval (int, optional): Number of spline interpolations saved if tabled bond is used. Defaults to 1000.

    Returns:
     -  lammps_ff_path (str): Destination of the external force field file.
    
    Raises:
        FileExistsError: If the force field template do not exist.
    """
    if not os.path.exists(ff_template):
        raise FileExistsError(f"Template file does not exists:\n  {ff_template}")
    
    if not all(key in potential_kwargs for key in ["pair_style", "vdw_style", "coulomb_style"]):
        key = 'pair_style' if not 'pair_style' in potential_kwargs else 'vdw_style' if not 'vdw_style' in potential_kwargs else 'coulomb_style'
        raise KeyError(f"{key} not in keys of the potential kwargs! Check nonbonded input!")
    
    if not all(key in coupling_potential for key in ["vdw", "coulomb"]):
        key = 'vdw' if not 'vdw' in coupling_potential else 'coulomb'
        raise KeyError(f"{key} not in keys of the coupling potential! Check coupling input!")
    

    # Dictionary to render the template. Pass all keyword arguments directly in the template.
    renderdict = { **ff_kwargs, "charged": any( p["charge"] != 0 for p in nonbonded ), 
                   "shake_dict": shake_dict 
                 }
    
    # Define coupling lambdas
    vdw_lambdas, coul_lambdas = get_coupling_lambdas( combined_lambdas, coupling, precision )

    renderdict.update( { "vdw_lambdas": vdw_lambdas, "coul_lambdas": coul_lambdas } )
    
    # Define charge of coupled molecule
    renderdict["charge_list"] = [ [i, iatom["charge"]] for i,iatom in zip(solute_numbers, nonbonded) ]

    # Define cut of radius (scanning the force field and taking the highest value)
    rcut = max( p["cut"] for p in nonbonded if p["cut"] )

    # Update local variables with soft core settings
    local_variables = { **locals(), **coupling_soft_core }
    
    # Get pair style defined in force field
    
    vdw_pair_styles = list( { p["vdw_style"] for p in nonbonded if p["vdw_style"] } ) + [ coupling_potential["vdw"] ]
    coul_pair_styles = list( { p["coulomb_style"] for p in nonbonded if p["coulomb_style"] } ) + \
                        ( [ coupling_potential["coulomb"] ] if any(np.array(renderdict["charge_list"])[:,1]) else [] )
    
    pair_style = get_pair_style( local_variables, vdw_pair_styles,
                                 coul_pair_styles, potential_kwargs["pair_style"] )

    renderdict["pair_style"] = pair_style
    
    # Van der Waals and Coulomb pair interactions
    vdw_interactions     = { "solute_solute": [], 
                             "solution_solution": [], 
                             "solute_solution": [] 
                            }
    
    coulomb_interactions = { "all": [], 
                             "solute_solute": [ [ f"{solute_numbers[0]}*{solute_numbers[-1]} {solute_numbers[0]}*{solute_numbers[-1]}",  
                                                  coupling_potential['coulomb'], "${init_scaling_lambda}" ] 
                                              ]
                            }

    for i,iatom in zip(atom_numbers_ges, nonbonded):
        for j,jatom in zip(atom_numbers_ges[i-1:], nonbonded[i-1:]):
            
            if iatom["vdw_style"] != jatom["vdw_style"]:
                raise KeyError(f"Atom '{i}' and atom '{j}' has different vdW pair style:\n  {i}: {iatom['vdw_style']}\n  {j}: {jatom['vdw_style']}")
            
            # Get mixed parameters
            name_ij = f"{iatom['name']}  {jatom['name']}"
            sigma_ij, epsilon_ij = get_mixed_parameter( sigma_i=iatom["sigma"], sigma_j=jatom["sigma"],
                                                        epsilon_i=iatom["epsilon"], epsilon_j=jatom["epsilon"],
                                                        mixing_rule=mixing_rule, precision=4 )
            
            # In case n and m are present get mixed exponents
            if "n" in iatom.keys() and "n" in jatom.keys():
                n_ij  = ( iatom["n"] + jatom["n"] ) / 2
                m_ij  = ( iatom["m"] + jatom["m"] ) / 2

            # Check interaction type (solute_solute, solution_solution, solute_solution)
            key = "solute_solute" if (i in solute_numbers and j in solute_numbers) else \
                  "solution_solution" if (not i in solute_numbers and not j in solute_numbers) else \
                  "solute_solution"

            vdw_style = iatom["vdw_style"] if (i in solute_numbers and j in solute_numbers) or \
                                            (not i in solute_numbers and not j in solute_numbers) else \
                        coupling_potential['vdw']

            l_vdw_args = [] if (i in solute_numbers and j in solute_numbers) or \
                               (not i in solute_numbers and not j in solute_numbers) else \
                        [ "${init_vdw_lambda}" ]

            # Get local variables and map them with potential kwargs 
            local_variables = locals()

            vdw_interactions[key].append( [ i, j, vdw_style ] + 
                                          [ local_variables[arg] for arg in potential_kwargs["vdw_style"][iatom["vdw_style"]] ] +
                                          l_vdw_args + 
                                          [ "#", name_ij ] ) 

            # If the system is charged, add Coulomb interactions
            if renderdict["charged"]:
                if iatom["charge"] == 0.0 or jatom["charge"] == 0.0:
                    continue
                if iatom["coulomb_style"] != jatom["coulomb_style"]:
                    raise KeyError(f"Atom '{i}' and atom '{j}' has different Coulomb pair style:\n  {i}: {iatom['coulomb_style']}\n  {j}: {jatom['coulomb_style']}")
                
                coulomb_interactions["all"].append( [ i, j, iatom["coulomb_style"] ] + 
                                                    [ local_variables[arg] for arg in potential_kwargs["coulomb_style"][iatom["coulomb_style"]] ] +
                                                    [ "#", name_ij ] )

    # Overwrite Coulomb pair interactions if there is only one style
    if len( { p["coulomb_style"] for p in nonbonded if p["coulomb_style"] } ) == 1:
        coulomb_interactions["all"] = [ [ "*", "*", *{ p["coulomb_style"] for p in nonbonded if p["coulomb_style"] } ] ]

    renderdict["vdw_interactions"] = vdw_interactions
    renderdict["coulomb_interactions"] = coulomb_interactions

    ## Add all kind of bonded interactions

    # Bonded interactions
    bond_styles, bond_paras   = get_bonded_style( bond_numbers_ges, bonds, n_eval = n_eval )
    renderdict["bond_paras"]  = bond_paras 
    renderdict["bond_styles"] = bond_styles

    # Angle interactions
    angle_styles, angle_paras = get_bonded_style( angle_numbers_ges, angles )

    renderdict["angle_paras"]  = angle_paras 
    renderdict["angle_styles"] = angle_styles

    # Dihedral interactions
    torsion_styles, torsion_paras = get_bonded_style( torsion_numbers_ges, torsions )

    renderdict["torsion_paras"]  = torsion_paras 
    renderdict["torsion_styles"] = torsion_styles

            
    # If provided write pair interactions as external LAMMPS force field file.
    with open(ff_template) as file_:
        template = Template( file_.read() )
    
    rendered = template.render( renderdict )

    os.makedirs( os.path.dirname( lammps_ff_path ), exist_ok = True  )

    with open(lammps_ff_path, "w") as fh:
        fh.write(rendered) 

    return lammps_ff_path