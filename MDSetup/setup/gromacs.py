import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
from jinja2 import Template
from typing import List, Dict, Any
from ..tools.submission import submit_and_wait

def generate_initial_configuration( build_template: str, destination_folder: str, coordinate_paths: List[str], 
                                    molecules_list: List[Dict[str, str|int]], box_lenghts: List[float], 
                                    on_cluster: bool=False, initial_system: str="",
                                    n_try: int=10000, submission_command: str="qsub" ):
    """
    Generate initial configuration for molecular dynamics simulation with GROMACS.

    Parameters:
     - build_template (str): Template for system building.
     - destination_folder (str): The destination folder where the initial configurations will be saved.
     - coordinate_paths (List[str]): List of paths to coordinate files (GRO format) for each ensemble.
     - molecules_list (List[Dict[str, str|int]]): List with dictionaries with numbers and names of the molecules.
     - box_lengths (List[float]): List of box lengths for each ensemble. Provide [] if build_intial_box is false.
     - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
     - initial_system (str, optional): Path to initial system, if initial system should be used to add molecules rather than new box. Defaults to "".
     - n_try (int, optional): Number of attempts to insert molecules. Defaults to 10000.
     - submission_command (str, optional): Command to submit jobs for cluster

    Returns:
     - intial_coord (str): Path of inital configuration

    """
    # Define box folder
    box_folder = f"{destination_folder}/box"

    # Create and the output folder of the box
    os.makedirs( box_folder, exist_ok = True )

    # Check if job template file exists
    if not os.path.isfile( build_template ):
        raise FileNotFoundError(f"Build template file { build_template } not found.")
    else:
        with open( build_template ) as f:
            template = Template( f.read() )
    
    # Sort out molecules that are zero
    non_zero_coord_mol_no = [ (coord, value["name"], value["number"]) for coord,value in zip(coordinate_paths,molecules_list) if value["number"] > 0 ]

    # Define template settings
    template_settings = { "coord_mol_no": non_zero_coord_mol_no, 
                          "box_lengths": box_lenghts,
                          "initial_system": initial_system,
                          "n_try": n_try,
                          "folder": box_folder }

    # Define output file
    bash_file = f"{box_folder}/build_box.sh"

    # Write bash file
    with open( bash_file, "w" ) as f:
        f.write( template.render( template_settings ) )

    if on_cluster:
        print("\nSubmit build to cluster and wait untils it is finished.\n")
        submit_and_wait( job_files = [ bash_file ], submission_command = submission_command )
    else:
        print("\nBuild system locally! Wait until it is finished.\n")
        # Call the bash to build the box. Write GROMACS output to file.
        with open(f"{box_folder}/build_output.txt", "w") as f:
            subprocess.run(["bash", f"{box_folder}/build_box.sh"], stdout=f, stderr=f)
        
    intial_coord = f"{box_folder}/init_conf.gro"

    # Check if the system is build 
    if not os.path.isfile( intial_coord ):
        raise FileNotFoundError(f"Something went wrong during the box building! { intial_coord } not found.")
    print("Build successful\n")

    return intial_coord


def generate_mdp_files( destination_folder: str, mdp_template: str, ensembles: List[str], temperature: float, pressure: float, 
                        compressibility: float, ensemble_definition: Dict[str, Any|Dict[str, str|float]], 
                        simulation_times: List[float], dt: float, kwargs: Dict[str, Any]={}, off_set: int=0,
                        init_step: int=0 ):
    """
    Generate MDP files for simulation pipeline.

    Parameters:
     - destination_folder (str): The destination folder where the MDP files will be saved. Will be saved under destination_folder/0x_ensebmle/ensemble.mdp
     - mdp_template (str): The path to the MDP template file.
     - ensembles (List[str]): A list of ensembles to generate MDP files for.
     - temperature (float): The temperature for the simulation.
     - pressure (float): The pressure for the simulation.
     - compressibility (float): The compressibility for the simulation.
     - ensemble_definition (Dict[str, Any|Dict[str, str|float]]): Dictionary containing the ensemble settings for each ensemble.
     - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
     - dt (float): The time step for the simulation.
     - kwargs (Dict[str, Any], optional): Additional keyword arguments for the mdp. That should contain all default values. Defaults to {}.
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - init_step (int, optional): Starting step if first simulation should be extended. Defaults to 0.
    
    Raises:
     - KeyError: If an invalid ensemble is specified.

    Returns:
     - mdp_files (List[str]): List with paths of the mdp files

    """

    # Produce mdp files for simulation pipeline
    mdp_files = []
    for j,(ensemble,time) in enumerate( zip( ensembles, simulation_times ) ):
        
        try:
            ensemble_settings = ensemble_definition[ensemble]
        except:
            raise KeyError(f"Wrong ensemple specified: {ensemble}. Valid options are: {', '.join(ensemble_definition.keys())} ")
        
        # Add temperature of sim to ensemble settings
        if "t" in ensemble_settings.keys():
            ensemble_settings["t"]["ref_t"] = temperature
        
        # Add pressure and compressibility to ensemble settings
        if "p" in ensemble_settings.keys():
            ensemble_settings["p"].update( { "ref_p": pressure, "compressibility": compressibility } )

        # Simulation time is provided in nano seconds and dt in pico seconds, hence multiply with factor 1000
        kwargs["system"]["nsteps"] = int( 1000 * time / dt ) if not ensemble == "em" else int(time)
        kwargs["system"]["dt"]     = dt
        kwargs["restart"]          = "no" if ensemble == "em" or ensembles[j-1] == "em" else "yes"

        # Overwrite the ensemble settings
        kwargs["ensemble"]        = ensemble_settings

        # Provide a seed for tempearture generating:
        kwargs["seed"] = np.random.randint(0,1e5)
        
        # Add extension to first system (if wanted)
        if j == 0 and init_step > 0:
            kwargs["system"]["init_step"] = init_step

        # Open and fill template
        with open( mdp_template ) as f:
            template = Template( f.read() )

        rendered = template.render( ** kwargs ) 

        # Write template
        mdp_out = f"{destination_folder}/{'0' if (j+off_set) < 10 else ''}{j+off_set}_{ensemble}/{ensemble}.mdp"

        # Create the destination folder
        os.makedirs( os.path.dirname( mdp_out ), exist_ok = True )

        with open( mdp_out, "w" ) as f:
            f.write( rendered )
            
        mdp_files.append( mdp_out )

    return mdp_files

def generate_job_file( destination_folder: str, job_template: str, mdp_files: List[str], intial_coord: str, initial_topo: str, 
                       job_name: str, job_out: str="job.sh", ensembles: List[str]=[ "em", "nvt", "npt", "prod" ],
                       initial_cpt: str="", off_set: int=0, extend_sim: bool=False ):
    """
    Generate initial job file for a set of simulation ensemble

    Parameters:
     - destination_folder (str): Path to the destination folder where the job file will be created.
     - job_template (str): Path to the job template file.
     - mdp_files (List[List[str]]): List of lists containing the paths to the MDP files for each simulation phase.
     - intial_coord (str): Path to the initial coordinate file.
     - initial_topo (str): Path to the initial topology file.
     - job_name (str): Name of the job.
     - job_out (str, optional): Name of the job file. Defaults to "job.sh".
     - ensembles (List[str], optional): List of simulation ensembles. Defaults to ["em", "nvt", "npt", "prod"].
     - initial_cpt (str, optional): Path to the inital checkpoint file. Defaults to "".
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - extend_time (int, optional): If cpt file is provided, extend this simulation by this amount of steps.

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

    # Check for mdp files
    for file in mdp_files:
        if not os.path.isfile( file ):
            raise FileNotFoundError(f"Mdp file { file  } not found.")
    
    # Check if topology file exists
    if not os.path.isfile( initial_topo ):
        raise FileNotFoundError(f"Topology file { initial_topo } not found.")

    # Check if coordinate file exists
    if not os.path.isfile( intial_coord ):
        raise FileNotFoundError(f"Coordinate file { intial_coord } not found.")
    
    # Check if checkpoint file exists
    if initial_cpt and not os.path.isfile( initial_cpt ):
        raise FileNotFoundError(f"Checkpoint file { initial_cpt } not found.")
    
    with open(job_template) as f:
        template = Template(f.read())

    job_file_settings = { "ensembles": { f"{'0' if (j+off_set) < 10 else ''}{j+off_set}_{step}": {} for j,step in enumerate(ensembles)} }
    ensemble_names    = list(job_file_settings["ensembles"].keys())

    # Create the simulation folder
    os.makedirs( destination_folder, exist_ok = True )

    # Relative paths for each mdp file for each simulation phase
    mdp_relative  = [ os.path.relpath( mdp_files[j], f"{destination_folder}/{step}" ) for j,step in enumerate(ensemble_names) ]

    # Relative paths for each coordinate file (for energy minimization use initial coodinates, otherwise use the preceeding output)
    cord_relative = [ f"../{ensemble_names[j-1]}/{ensembles[j-1]}.gro" if j > 0 else os.path.relpath( intial_coord, f"{destination_folder}/{step}" ) for j,step in enumerate(job_file_settings["ensembles"].keys()) ]

    # Relative paths for each checkpoint file 
    cpt_relative  = [ f"../{ensemble_names[j-1]}/{ensembles[j-1]}.cpt" if j > 0 else os.path.relpath( initial_cpt, f"{destination_folder}/{step}" ) if initial_cpt and not ensembles[j] == "em" else "" for j,step in enumerate(ensemble_names) ]

    # Relative paths for topology
    topo_relative = [ os.path.relpath( initial_topo, f"{destination_folder}/{step}" ) for j,step in enumerate(ensemble_names) ]

    # output file 
    out_relative  = [ f"{step}.tpr -maxwarn 10" for step in ensembles]

    for j,step in enumerate(ensemble_names):

        # If first or preceeding step is energy minimization, or if there is no cpt file to read in
        if ensembles[j-1]  == "em" or ensembles[j] == "em" or not cpt_relative[j]:
            job_file_settings["ensembles"][step]["grompp"] = f"grompp -f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -o {out_relative[j]}"
        else:
            job_file_settings["ensembles"][step]["grompp"] = f"grompp -f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -t {cpt_relative[j]} -o {out_relative[j]}"
        
        # Define mdrun command
        if j == 0 and initial_cpt and extend_sim:
            # In case extension of the first simulation in the pipeline is wanted
            job_file_settings["ensembles"][step]["grompp"] = f"grompp -f {mdp_relative[j]} -c {ensembles[j]}.gro -p {topo_relative[j]} -t {ensembles[j]}.cpt -o {out_relative[j]}"
            job_file_settings["ensembles"][step]["mdrun"] = f"mdrun -deffnm {ensembles[j]} -cpi {ensembles[j]}.cpt" 
        else: 
            job_file_settings["ensembles"][step]["mdrun"] = f"mdrun -deffnm {ensembles[j]}" 

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


def change_topo( topology_path: str, topology_out: str, molecules_list: List[Dict[str, str|int]], system_name: str ):
    """
    Change the number of molecules in a topology file.

    Parameters:
    - topology_path (str): The path to the topology file.
    - topology_out (str): The output path of the changed topology file.
    - molecules_list (List[Dict[str, str|int]]): List with dictionaries with numbers and names of the molecules. If a molecule has the number "-1", 
                                                 then it will increase the initial topology number of that molecule by one.
    - system_name (str): Name of the new system.

    Returns:
    - topology_out (str): Destination of new topology file

    Description:
    This function reads the content of the topology file specified by 'topology_path' and searches for the section containing the number of molecules. 
    It then finds the line containing the molecule and changes the number to the specified one.

    Example:
    change_topo('topology.txt', "topology_new.txt", {'water':5}, "pure_water")
    """
    
    with open(topology_path) as f: 
        lines = [line for line in f]
    
    # Change system name accordingly
    system_str_idx = [ i for i,line in enumerate(lines) if "[ system ]" in line and not line.startswith(";")][0] + 1
    system_end_idx = [ i for i,line in enumerate(lines) if (line.startswith("[") or i == len(lines)-1) and i > system_str_idx ][0]

    for i,line in enumerate(lines[ system_str_idx: system_end_idx ]):
        if line and not line.startswith(";"):
            lines[i+system_str_idx] = f"{system_name}\n\n"
            break
    
    molecule_str_idx = [ i for i,line in enumerate(lines) if "[ molecules ]" in line and not line.startswith(";")][0] + 1
    molecule_end_idx = [ i for i,line in enumerate(lines) if (line.startswith("[") or i == len(lines)-1) and i > molecule_str_idx ][0] + 1

    for i,line in enumerate(lines[ molecule_str_idx: molecule_end_idx ]):
        for molecule in molecules_list:
            if line.split("\n")[0] and line.split()[0].replace(";","") == molecule["name"]:
                # Get current (old) number of molecule
                old = line.split()[1]

                # In case a None is provided, simply increase the current number by one. Otherwise replace it
                new = str( int(old) + 1 ) if molecule["number"] == -1 else str( molecule["number"] )

                # Make sure that line is not commented
                lines[i+molecule_str_idx] = lines[i+molecule_str_idx].replace( old, new ).replace( ";", "" )

        # Check if the number of any molecules is zero, then comment it out
        if lines[i+molecule_str_idx].split("\n")[0] and not lines[i+molecule_str_idx].startswith(";"):
            # Get current number of molecule
            mol_number = lines[i+molecule_str_idx].split()[1]
            if int(mol_number) == 0:
                lines[i+molecule_str_idx] = ";" + lines[i+molecule_str_idx]

    # Write new topology file
    os.makedirs( os.path.dirname(topology_out), exist_ok = True )

    with open(topology_out,"w") as f:
        f.writelines(lines)

    return topology_out