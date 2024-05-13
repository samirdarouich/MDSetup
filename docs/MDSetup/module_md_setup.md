Module MDSetup.module_md_setup
==============================

Classes
-------

`MDSetup(system_setup: str, simulation_default: str, simulation_ensemble: str, submission_command: str, simulation_sampling: str = '')`
:   This class sets up structured and FAIR molecular dynamic simulations. It also has the capability to build a system based on a list of molecules.
    
    Initialize a new instance of the MDsetup class.
    
    Parameters:
     - system_setup (str): Path to the system setup YAML file. Containing all system settings.
     - simulation_default (str): Path to the simulation default YAML file. Containing all default MD settings.
     - simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all MD ensemble settings.
     - submission_command (str): Command to submit jobs to cluster.
     - simulation_sampling (str,optional): Path to the sampling YAML file. Containing all sampling settings. 
                                           This is only needed for LAMMPS.
    
    Returns:
        None

    ### Methods

    `prepare_simulation(self, folder_name: str, ensembles: List[str], simulation_times: List[float], initial_systems: List[str] = [], copies: int = 0, on_cluster: bool = False, off_set: int = 0, input_kwargs: Dict[str, Any] = {}, **kwargs)`
    :   Prepares the simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
        The method checks if an initial configuration file is provided. 
        If not, it generates the initial configuration based on the provided by the software.
        It then generates input files for each ensemble in a separate folder and creates a job file for each copy of the simulation.
        
        Parameters:
         - folder_name (str, optional): Name of the subfolder where to perform the simulations.
                                        Path structure is as follows: system.folder/system.name/folder_name
         - ensembles (List[str]): A list of ensembles to generate input files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - input_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the input template. Defaults to "{}".
         - on_cluster (bool, optional): If the build should be submited to the cluster. Defaults to "False".
         - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
         - **kwargs: Arbitrary keyword arguments.
        
        Returns:
            None

    `submit_simulation(self)`
    :   Function that submits predefined jobs to the cluster.
        
        Parameters:
            None
        
        Returns:
            None

    `write_topology(self)`
    :