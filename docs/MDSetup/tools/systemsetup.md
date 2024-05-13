Module MDSetup.tools.systemsetup
================================

Functions
---------

    
`generate_initial_configuration(destination_folder: str, build_template: str, software: str, coordinate_paths: List[str], molecules_list: List[Dict[str, str | int]], box: Dict[str, float], on_cluster: bool = False, initial_system: str = '', n_try: int = 10000, submission_command: str = 'qsub', **kwargs)`
:   Generate initial configuration for molecular dynamics simulation with GROMACS.
    
    Parameters:
     - destination_folder (str): The destination folder where the initial configurations will be saved.
     - build_template (str): Template for system building.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - coordinate_paths (List[str]): List of paths to coordinate files (GRO format) for each ensemble.
     - molecules_list (List[Dict[str, str|int]]): List with dictionaries with numbers and names of the molecules.
     - box (Dict[str,float]): List of box lengths for each ensemble. Provide [] if build_intial_box is false.
     - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
     - initial_system (str, optional): Path to initial system, if initial system should be used to add molecules rather than new box. Defaults to "".
     - n_try (int, optional): Number of attempts to insert molecules. Defaults to 10000.
     - submission_command (str, optional): Command to submit jobs for cluster,
     - **kwargs (Any): Arbitrary keyword arguments.
    
    Keyword Args:
     - 
    
    Returns:
     - intial_coord (str): Path of inital configuration

    
`generate_input_files(destination_folder: str, input_template: str, software: str, ensembles: List[str], simulation_times: List[float], ensemble_definition: Dict[str, Union[Any, Dict[str, str | float]]], dt: float, temperature: float, pressure: float, off_set: int = 0, **kwargs)`
:   Generate input files for simulation pipeline.
    
    Parameters:
     - destination_folder (str): The destination folder where the input files will be saved. Will be saved under destination_folder/0x_ensebmle/ensemble.input
     - input_template (str): The path to the input template file.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - ensembles (List[str]): A list of ensembles to generate input files for.
     - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
     - ensemble_definition (Dict[str, Any|Dict[str, str|float]]): Dictionary containing the ensemble settings for each ensemble.
     - dt (float): The time step for the simulation.
     - temperature (float): The temperature for the simulation.
     - pressure (float): The pressure for the simulation.
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - **kwargs (Any): Arbitrary keyword arguments.
    
    Keyword Args:
     - initial_coord (str): Absolute path of LAMMPS data file for LAMMPS.
     - initial_topology (str): Absolute path of LAMMPS force field file for LAMMPS.
     - compressibility (float): Compressibility of the system for GROMACS.
     - init_step (int): Initial step to continue simulation for GROMACS.
    
    Raises:
     - KeyError: If an invalid ensemble is specified.
     - FileNotFoundError: If any input file does not exists.
    
    Returns:
     - input_files (List[str]): List with paths of the input files

    
`generate_job_file(destination_folder: str, job_template: str, software: str, input_files: List[str], ensembles: List[str], job_name: str, job_out: str = 'job.sh', off_set: int = 0, **kwargs)`
:   Generate initial job file for a set of simulation ensembles.
    
    Parameters:
     - destination_folder (str): Path to the destination folder where the job file will be created.
     - job_template (str): Path to the job template file.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - input_files (List[List[str]]): List of lists containing the paths to the input files for each simulation phase.
     - ensembles (List[str], optional): List of simulation ensembles.
     - job_name (str): Name of the job.
     - job_out (str, optional): Name of the job file. Defaults to "job.sh".
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - **kwargs (Any): Arbitrary keyword arguments.
    
    Keyword Args:
     - initial_topology (str): Path to the initial topology file for GROMACS.
     - intial_coord (str): Path to the initial coordinate file for GROMACS.
     - initial_cpt (str): Path to the inital checkpoint file for GROMACS.
     - init_step (int): Initial step to continue simulation for GROMACS.
    
    Returns:
     - job_file (str): Path of job file
    
    Raises:
     - FileNotFoundError: If the job template file does not exist.
     - FileNotFoundError: If any of the input files does not exist.
     - FileNotFoundError: If the initial coordinate file does not exist.
     - FileNotFoundError: If the initial topology file does not exist.
     - FileNotFoundError: If the initial checkpoint file does not exist.

    
`get_system_volume(molar_masses: List[float], molecule_numbers: List[int], density: float, unit_conversion: float, box_type: str = 'cubic', z_x_relation: float = 1.0, z_y_relation: float = 1.0)`
:   Calculate the volume of a system and the dimensions of its bounding box based on molecular masses, numbers and density.
    
    Parameters:
    - molar_masses (List[List[float]]): A list with the molar masses of each molecule in the system.
    - molecule_numbers (List[int]): A list containing the number of molecules of each type in the system.
    - density (float): The density of the mixture in kg/m^3.
    - unit_conversion (float): Unit conversion from Angstrom to xx.
    - box_type (str, optional): The type of box to calculate dimensions for. Currently, only 'cubic' is implemented.
    - z_x_relation (float, optional): Relation of z to x length. z = z_x_relation*x. Defaults to 1.0.
    - z_y_relation (float, optional): Relation of z to y length. z = z_y_relation*y. Defaults to 1.0.
    
    Returns:
    - dict: A dictionary with keys 'box_x', 'box_y', and 'box_z', each containing a list with the negative and positive half-lengths of the box in Angstroms.
    
    Raises:
    - KeyError: If the `box_type` is not 'cubic' or 'orthorhombic', since other box types are not implemented yet.