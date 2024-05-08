Module pyLMP.tools.utils
========================

Functions
---------

    
`generate_initial_configuration(lammps_molecules: pyLMP.tools.lammps_utils.LAMMPS_molecules, destination_folder: str, molecules_dict_list: List[Dict[str, str | float]], density: float, template_xyz: str, playmol_ff_template: str, playmol_input_template: str, playmol_bash_file: str, lammps_data_template: str, box_type: str = 'cubic', z_x_relation: float = 1.0, z_y_relation: float = 1.0, submission_command: str = 'qsub', on_cluster: bool = False)`
:   

    
`generate_input_files(destination_folder: str, input_template: str, ensembles: List[str], temperature: float, pressure: float, data_file: str, ff_file: str, ensemble_definition: Dict[str, Union[Any, Dict[str, str | float]]], simulation_times: List[float], dt: float, kwargs: Dict[str, Any] = {}, off_set: int = 0)`
:   Generate input files for simulation pipeline.
    
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

    
`generate_job_file(destination_folder: str, job_template: str, input_files: List[str], ensembles: List[str], job_name: str, job_out: str = 'job.sh', off_set: int = 0)`
:   Generate initial job file for a set of simulation ensemble
    
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