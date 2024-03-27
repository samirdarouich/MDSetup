import os
import numpy as np
from jinja2 import Template
from typing import List, Dict, Any
from .lammps_utils import LAMMPS_molecules
from .playmol_utils import prepare_playmol_input, write_playmol_input
from .molecule_utils import get_molecule_coordinates

def generate_initial_configuration( lammps_molecules: LAMMPS_molecules, destination_folder: str,
                                    molecules_dict_list: List[Dict[str,str|float]], density: float,
                                    template_xyz: str, playmol_ff_template: str, 
                                    playmol_input_template: str, playmol_bash_file: str,
                                    lammps_data_template: str,
                                    submission_command: str="qsub", on_cluster: bool=False ):   
                
    # Get molecule xyz from pubchem
    xyz_destinations = [ f'{destination_folder}/build/{mol["name"]}.xyz' for mol in molecules_dict_list ]
    get_molecule_coordinates( molecule_name_list = [ mol["name"] for mol in molecules_dict_list ], 
                                molecule_graph_list = [ mol["graph"] for mol in molecules_dict_list ],
                                molecule_smiles_list = [ mol["smiles"] for mol in molecules_dict_list ],
                                xyz_destinations = xyz_destinations, 
                                template_xyz = template_xyz,
                                verbose = False
                            )
    
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
                                        nb_all = lammps_molecules.ff_all, 
                                        playmol_template = playmol_input_template,
                                        playmol_path = playmol_mol, 
                                        playmol_ff_path = playmol_relative_ff_path, 
                                        xyz_paths = playmol_relative_xyz_path, 
                                        playmol_execute_template = playmol_bash_file,
                                        submission_command = submission_command, 
                                        on_cluster = on_cluster
                                    )
                    

    # Write LAMMPS data file 
    ####################### (possible to do: extract from playmol the box dimension, so no need to compute them ourself)
    lammps_data_file = f"{destination_folder}/build/system.data"
    lammps_molecules.write_lammps_data( xyz_path = playmol_xyz, 
                                        data_template = lammps_data_template,
                                        data_path = lammps_data_file,
                                        nmol_list = [ mol["number"] for mol in molecules_dict_list ],
                                        density = density 
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