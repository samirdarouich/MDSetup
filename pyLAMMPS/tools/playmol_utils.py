import os
import numpy as np
import subprocess
import moleculegraph
from jinja2 import Template
from typing import List, Dict
from .submission_utils import submit_and_wait

def prepare_playmol_input(mol_str: List[str], ff: Dict[str,Dict[str,float]], playmol_template: str, playmol_ff_path: str):
    """
    Function that writes playmol force field using a jinja2 template given a list of moleculegraph strings

    Parameters:
        - mol_str (List[str]): List of moleculegraph interpretable strings
        - ff (Dict[str,Dict[str,float]]): Dictinoary with moleculegraph interpretable keys for the force field.
        - playmol_template (str): Path to playmol template for system building.
        - playmol_ff_path (str): Path were the new playmol force field file should be writen to.
    """

    # Get moleculegraph representation for each molecule
    mol_list  =  [ moleculegraph.molecule(mol) for mol in mol_str ]

    # Get (unique) atom types and parameters 
    nonbonded = [j for sub in [molecule.map_molecule( molecule.unique_atom_keys, ff["atoms"] ) for molecule in mol_list] for j in sub]
    
    # Get (unique) bond types and parameters
    bonds     = [j for sub in [molecule.map_molecule( molecule.unique_bond_keys, ff["bonds"] ) for molecule in mol_list] for j in sub]
    
    # Get (unique) angle types and parameters
    angles    = [j for sub in [molecule.map_molecule( molecule.unique_angle_keys, ff["angles"] ) for molecule in mol_list] for j in sub]
    
    # Get (unique) torsion types and parameters
    torsions  = [j for sub in [molecule.map_molecule( molecule.unique_torsion_keys, ff["torsions"] ) for molecule in mol_list] for j in sub]
    

    # Prepare dictionary for jinja2 template to write force field input for Playmol 
    renderdict              = {}
    renderdict["nonbonded"] = list( zip( [j for sub in [molecule.unique_atom_keys for molecule in mol_list] for j in sub], nonbonded ) )
    renderdict["bonds"]     = list( zip( [j for sub in [molecule.unique_bond_names for molecule in mol_list] for j in sub], bonds ) )
    renderdict["angles"]    = list( zip( [j for sub in [molecule.unique_angle_names for molecule in mol_list] for j in sub], angles ) )
    renderdict["torsions"]  = list( zip( [j for sub in [molecule.unique_torsion_names for molecule in mol_list] for j in sub], torsions ) )
    
    # Generate force field file for playmol using jinja2 template
    os.makedirs( os.path.dirname(playmol_ff_path), exist_ok=True )

    with open(playmol_template) as file_:
        template = Template(file_.read())
    
    rendered = template.render( rd=renderdict )

    with open(playmol_ff_path, "w") as fh:
        fh.write(rendered) 

    return


def write_playmol_input( mol_str: List[str], molecule_numbers: List[int], density: float, 
                         nb_all: List[Dict[str,str|float]], playmol_template: str, 
                         playmol_path: str, playmol_ff_path: str, xyz_paths: List[str], 
                         playmol_execute_template: str, submission_command: str,
                         on_cluster: bool=False):
    """
    Function that generates input file for playmol to build the specified system, as well as execute playmol to build the system

    Parameters:
     - mol_str (List[str]): List of moleculegraph interpretable strings
     - molecule_numbers (List[int]): List of molecule numbers per component
     - density (float): Estimated density of the system.
     - nb_all (List[Dict[str,str|float]]): List with nonbonded information of every atomtype in the system (not only the unique ones)
     - playmol_template (str): Path to playmol input template.
     - playmol_path (str): Path where the playmol .mol file is writen and executed.
     - playmol_ff_path (str): Path to the playmol force field file.
     - xyz_paths (List[str]): List with the path(s) to the xyz file(s) for each component.
     - playmol_executeable_template (str): Path to bash template to execute playmol.
     - submission_command (str, optional): The command used to submit the job files.
     - on_cluster (bool, optional): If the PLAYMOL build should be submited to the cluster. Defaults to "False".
    
    Returns:
     - xyz_path (str): Path to build coordinate file.
    """

    # Get moleculegraph representation for each molecule
    mol_list      =  [ moleculegraph.molecule(mol) for mol in mol_str ]
                
    moldict       = {}

    # Get running atom numbers --> These are the numbers of all atoms in each molecule. Here the index of each atom needs to be 
    # lifted by the index of all preceeding atoms.
    add_atom      = [1] + [ sum(mol.atom_number for mol in mol_list[:(i+1)]) + 1 for i in range( len(mol_list[1:]) ) ]
    atom_numbers  = list( np.concatenate( [ mol.atom_numbers + add_atom[i] for i,mol in enumerate(mol_list) ] ) )
    
    # Get running bond numbers --> These are the corresponding atoms of each bond in each molecule. (In order to use the right atom index, the add_atom list is used)
    bond_numbers  = list( np.concatenate( [mol.bond_list + add_atom[i] for i,mol in enumerate(mol_list)], axis=0 ) )

    # Get the force field type of each atom in each molecule.
    atom_names    = [j for sub in [molecule.atom_names for molecule in mol_list] for j in sub]

    # Get the names of each atom in each bond of each molecule. This is done to explain playmol which bond type they should use for this bond
    playmol_bond_names = list(np.concatenate( [ [ [mol.atom_names[i] for i in bl] for bl in mol.bond_list ] for mol in mol_list], axis=0 ))

    # Playmol uses as atom input: atom_name force_field_type charge --> the atom_name is "force_field_type+atom_index"
    moldict["atoms"]   = list(zip( atom_numbers, atom_names, [nb_all[i]["charge"] for i,_ in enumerate(atom_names)] ) )

    # Playmol uses as bond input: atom_name atom_name --> the atom_name is "force_field_type+atom_index"
    moldict["bonds"]   = list(zip( bond_numbers, playmol_bond_names))

    # Provide the number of molecules per component, as well as the starting atom of each molecule (e.g.: molecule1 = {C1, C2, C3}, molecule2 = {C4, C5, C6} --> Provide C1 and C4 )
    molecule_indices   = [a-1 for a in add_atom]
    moldict["mol"]     = list( zip( molecule_numbers, [ str(moldict["atoms"][i][1])+str(moldict["atoms"][i][0]) for i in molecule_indices ] ) )

    # Add path to force field
    moldict["force_field"] = playmol_ff_path

    # Add path to xyz of one molecule of each component
    moldict["xyz"]         = xyz_paths

    # Add name of the final xyz file and log file
    moldict["final_xyz"]   = "inital.xyz"
    moldict["final_log"]   = "build.log"


    ## Write playmol input file to build the system with specified number of molecules for each component ##

    with open(playmol_template) as file:
        template = Template(file.read())

    # Playmol template needs density in g/cm^3; rho given in kg/m^3 to convert in g/cm^3 divide by 1000
    rendered = template.render( rd   = moldict,
                                rho  = str(density / 1000),
                                seed = np.random.randint(1,1e6) )
    
    os.makedirs( os.path.dirname(playmol_path), exist_ok = True)

    with open(playmol_path, "w") as fh:
        fh.write(rendered) 
    
    # Write bash template 
    with open(playmol_execute_template) as file:
        template = Template(file.read())

    rendered = template.render( folder = os.path.dirname(playmol_path),
                                file   = playmol_path
                             )
    
    bash_file = f"{os.path.dirname(playmol_path)}/build_system.sh"
    with open(bash_file, "w") as fh:
        fh.write(rendered) 

    # Execute bash file
    if on_cluster:
        print("\nSubmit build to cluster and wait untils it is finished.\n")
        submit_and_wait( job_files = [ bash_file ], submission_command = submission_command )
    else:
        print("\nBuild system locally! Wait until it is finished.\n")
        # Call the bash to build the box. Write GROMACS output to file.
        with open(f"{os.path.dirname(playmol_path)}/build_output.txt", "w") as f:
            subprocess.run(["bash", f"{os.path.dirname(playmol_path)}/build_system.sh"], stdout=f, stderr=f)

    # Check if the system is build 
    if not os.path.isfile( f"{os.path.dirname(playmol_path)}/{moldict['final_xyz']}"  ):
        raise FileNotFoundError(f"Something went wrong during the box building! '{os.path.dirname(playmol_path)}/{moldict['final_xyz']}' not found.")
    print("Build successful\n")
    
    return f"{os.path.dirname(playmol_path)}/{moldict['final_xyz']}"