import os
import numpy as np
from jinja2 import Template
from typing import List, Dict, Any
from ..forcefield.lammps import LAMMPS_molecules
from ..tools.general import flatten_list
from ..tools.molecule import get_molecule_coordinates
from ..forcefield.playmol import prepare_playmol_input, write_playmol_input
from ..forcefield.lammps import get_bonded_style, get_mixed_parameter, get_pair_style

def generate_initial_configuration( lammps_molecules: LAMMPS_molecules, destination_folder: str,
                                    molecules_dict_list: List[Dict[str,str|float]], density: float,
                                    template_xyz: str, playmol_ff_template: str, 
                                    playmol_input_template: str, playmol_bash_file: str,
                                    lammps_data_template: str,  box_type: str="cubic",
                                    z_x_relation: float=1.0, z_y_relation: float=1.0,
                                    submission_command: str="qsub", on_cluster: bool=False ):   
                
    # Get molecule xyz from pubchem
    xyz_destinations = [ f'{destination_folder}/build/{mol["name"]}.xyz' for mol in molecules_dict_list ]
    (raw_atom_numbers, final_atomtyps, 
    final_atomsymbols, final_coordinates) = get_molecule_coordinates( molecule_name_list = [ mol["name"] for mol in molecules_dict_list ], 
                                                                      molecule_graph_list = [ mol["graph"] for mol in molecules_dict_list ],
                                                                      molecule_smiles_list = [ mol["smiles"] for mol in molecules_dict_list ],
                                                                      verbose = False
                                                                    )

    # Write coordinates to xyz files.
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

    # Build system with PLAYMOL
    playmol_ff  = f'{destination_folder}/build/force_field.playmol'
    playmol_mol = f'{destination_folder}/build/build_script.mol'

    prepare_playmol_input( mol_str = lammps_molecules.mol_str, 
                            ff = lammps_molecules.ff, 
                            playmol_template = playmol_ff_template,
                            playmol_ff_path = playmol_ff
                        )

    playmol_relative_ff_path  = os.path.relpath(playmol_ff, os.path.dirname(playmol_mol))
    playmol_relative_xyz_path = [ os.path.relpath(xyz, os.path.dirname(playmol_mol)) for xyz in xyz_destinations ]

    playmol_xyz = write_playmol_input( mol_str = lammps_molecules.mol_str, 
                                        molecule_numbers = [ mol["number"] for mol in molecules_dict_list ], 
                                        density = density, 
                                        nb_all = flatten_list(lammps_molecules.ff_all), 
                                        playmol_template = playmol_input_template,
                                        playmol_path = playmol_mol, 
                                        playmol_ff_path = playmol_relative_ff_path, 
                                        xyz_paths = playmol_relative_xyz_path, 
                                        playmol_execute_template = playmol_bash_file,
                                        submission_command = submission_command, 
                                        on_cluster = on_cluster
                                    )
                    

    # Write LAMMPS data file 
    
    lammps_data_file = f"{destination_folder}/build/system.data"
    lammps_molecules.write_lammps_data( xyz_path = playmol_xyz, 
                                        data_template = lammps_data_template,
                                        data_path = lammps_data_file,
                                        nmol_list = [ mol["number"] for mol in molecules_dict_list ],
                                        density = density,
                                        box_type = box_type,
                                        z_x_relation = z_x_relation,
                                        z_y_relation = z_y_relation
                                    )
    
    if not os.path.exists(lammps_data_file):
        raise FileExistsError("Something went wrong during the production of the LAMMPS data!\n")
    
    return lammps_data_file


def generate_input_files( destination_folder: str, input_template: str, ensembles: List[str], 
                          temperature: float, pressure: float, data_file: str, ff_file: str,
                          ensemble_definition: Dict[str, Any|Dict[str, str|float]], 
                          simulation_times: List[float], dt: float, kwargs: Dict[str, Any]={}, 
                          off_set: int=0 ):
    
    """
    Generate input files for simulation pipeline.

    Parameters:
     - destination_folder (str): The destination folder where the input files will be saved. Will be saved under destination_folder/0x_ensebmle/ensemble.input
     - input_template (str): The path to the LAMMPS input template file.
     - ensembles (List[str]): A list of ensembles to generate input files for.
     - temperature (float): The temperature for the simulation.
     - pressure (float): The pressure for the simulation.
     - data_file (str): Path to LAMMPS data or restart file.
     - ff_file (str): Path to LAMMPS ff file.
     - ensemble_definition (Dict[str, Any|Dict[str, str|float]]): Dictionary containing the ensemble settings for each ensemble.
     - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
     - dt (float): The time step for the simulation.
     - kwargs (Dict[str, Any], optional): Additional keyword arguments for the input file. That should contain all default values. Defaults to {}.
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
    
    Raises:
     - KeyError: If an invalid ensemble is specified.
     - FileNotFoundError: If any input file does not exists.

    Returns:
     - input_files (List[str]): List with paths of the input files

    """

    # Check if input template file exists
    if not os.path.isfile( input_template ):
        raise FileNotFoundError(f"Input template file { input_template } not found.")
    
    # Check if datafile exists
    if not os.path.isfile( data_file ):
        raise FileNotFoundError(f"Data file { data_file } not found.")
    
    # Check if force field file exists
    if not os.path.isfile( ff_file ):
        raise FileNotFoundError(f"Force field file { ff_file } not found.")
        
    # Save ensemble names
    ensemble_names    = [ f"{'0' if (j+off_set) < 10 else ''}{j+off_set}_{step}" for j,step in enumerate(ensembles) ]

    # Produce input files for simulation pipeline
    input_files = []
    for j,(ensemble,time) in enumerate( zip( ensembles, simulation_times ) ):
        
        try:
            ensemble_settings = ensemble_definition[ensemble]
        except:
            raise KeyError(f"Wrong ensemple specified: {ensemble}. Valid options are: {', '.join(ensemble_definition.keys())} ")
        
        # Add ensebmle variables
        values = []
        for v in ensemble_settings["variables"]:
            if v == "temperature":
                values.append( temperature )
            elif v == "pressure":
                values.append( round( pressure / 1.01325, 3 ) )
            else:
                raise KeyError(f"Variable is not implemented: '{v}'. Currently implemented are 'temperature' or 'pressure'. ")


        # Simulation time is provided in nano seconds and dt in fs seconds, hence multiply with factor 1e6
        kwargs["system"]["nsteps"] = int( 1e6 * time / dt ) if not ensemble == "em" else int(time)
        kwargs["system"]["dt"]     = dt
 
        # Overwrite the ensemble settings
        kwargs["ensemble"]        = { "var_val": zip(ensemble_settings["variables"],values), "command": ensemble_settings["command"] }

        # Provide a seed for tempearture generating:
        kwargs["seed"] = np.random.randint(0,1e5)
        
        # Ensemble name
        kwargs["ensemble_name"] = ensemble

        # Write template
        input_out = f"{destination_folder}/{'0' if (j+off_set) < 10 else ''}{j+off_set}_{ensemble}/{ensemble}.input"

        kwargs["force_field_file"] = os.path.relpath( ff_file, os.path.dirname(input_out) )

        # If its the first ensemble use provided data path, otherwise use the previous restart file. Hence set restart flag
        if j == 0:
            kwargs["data_file"] = os.path.relpath( data_file, os.path.dirname(input_out) )
        else:
            kwargs["data_file"] = f"../{ensemble_names[j-1]}/{ensembles[j-1]}.restart"
            kwargs["restart_flag"] = True

        # Open and fill template
        with open( input_template ) as f:
            template = Template( f.read() )

        rendered = template.render( ** kwargs ) 

        # Create the destination folder
        os.makedirs( os.path.dirname( input_out ), exist_ok = True )

        with open( input_out, "w" ) as f:
            f.write( rendered )
            
        input_files.append( input_out )

    return input_files



def generate_job_file( destination_folder: str, job_template: str, input_files: List[str],
                       ensembles: List[str], job_name: str, job_out: str="job.sh", 
                       off_set: int=0 ):
    
    """
    Generate initial job file for a set of simulation ensemble

    Parameters:
     - destination_folder (str): Path to the destination folder where the job file will be created.
     - job_template (str): Path to the job template file.
     - input_files (List[str]): List of lists containing the paths to the input files for each simulation phase.
     - ensembles (List[str]): List of simulation ensembles
     - job_name (str): Name of the job.
     - job_out (str, optional): Name of the job file. Defaults to "job.sh".
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.

    Returns:
     - job_file (str): Path of job file

    Raises:
     - FileNotFoundError: If the job template file does not exist.
     - FileNotFoundError: If any of the MDP files does not exist.
     - FileNotFoundError: If the initial coordinate file does not exist.
     - FileNotFoundError: If the initial topology file does not exist.
     - FileNotFoundError: If the initial checkpoint file does not exist.
    """

    # Check if job template file exists
    if not os.path.isfile( job_template ):
        raise FileNotFoundError(f"Job template file { job_template } not found.")

    # Check for input files
    for file in input_files:
        if not os.path.isfile( file ):
            raise FileNotFoundError(f"Input file { file  } not found.")
    
    with open(job_template) as f:
        template = Template(f.read())

    job_file_settings = { "ensembles": { f"{'0' if (j+off_set) < 10 else ''}{j+off_set}_{step}": {} for j,step in enumerate(ensembles)} }
    ensemble_names    = list(job_file_settings["ensembles"].keys())

    # Create the simulation folder
    os.makedirs( destination_folder, exist_ok = True )

    # Relative paths for each input file for each simulation phase
    for j,step in enumerate(ensemble_names):
        job_file_settings["ensembles"][step]["mdrun"] = os.path.relpath( input_files[j], f"{destination_folder}/{step}" )

    # Define LOG output
    log_path   = f"{destination_folder}/LOG"

    # Add to job file settings
    job_file_settings.update( { "job_name": job_name, "log_path": log_path, "working_path": destination_folder } )

    rendered = template.render( **job_file_settings )

    # Create the job folder
    job_file = f"{destination_folder}/{job_out}"

    os.makedirs( os.path.dirname( job_file ), exist_ok = True )

    # Write new job file
    with open( job_file, "w") as f:
        f.write( rendered )

    return job_file


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