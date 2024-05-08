Module pyLMP.module_lammps_setup
================================

Classes
-------

`LAMMPS_setup(system_setup: str, simulation_default: str, simulation_ensemble: str, simulation_sampling: str, submission_command: str)`
:   This class sets up structured and FAIR LAMMPS simulations. It also has the capability to build a system based on a list of molecules.
    
    Initialize a new instance of the LAMMPS_setup class.
    
    Parameters:
     - system_setup (str): Path to the system setup YAML file. Containing all system settings.
     - simulation_default (str): Path to the simulation default YAML file. Containing all default LAMMPS settings.
     - simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all LAMMPS ensemble settings.
     - simulation_sampling (str): Path to the sampling YAML file. Containing all sampling settings.
     - submission_command (str, optional): Command to submit jobs to cluster. Defaults to "qsub".
    
    Returns:
        None

    ### Methods

    `analysis_extract_properties(self, analysis_folder: str, ensemble: str, extracted_properties: List[str], output_suffix: str, fraction: float = 0.0, header: int = 2, header_delimiter: str = ',')`
    :   Extracts properties from LAMMPS output files for a specific ensemble.
        
        Parameters:
         - analysis_folder (str): The name of the folder where the analysis will be performed.
         - ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
         - extracted_properties (List[str]): A list of properties to be extracted from the LAMMPS output files.
         - output_suffix (str): Suffix of the LAMMPS output file to be analyzed.
         - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
         - header (int, optional): The number of header lines from which to extract the keys for the reported values. Defaults to 2.
         - header_delimiter (str, optional): The delimiter used in the header line. Defaults to ",".
        
        Returns:
         - None
        
        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using the specified suffix and properties list.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.

    `analysis_free_energy(self, analysis_folder: str, solute: str, ensemble: str, method: str = 'MBAR', fraction: float = 0.0, decorrelate: bool = True, visualize: bool = False, coupling: bool = True)`
    :   Extracts free energy difference for a specified folder and solute and ensemble.
        
        Parameters:
        - analysis_folder (str): The name of the folder where the analysis will be performed.
        - solute (str): Solute under investigation
        - ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
        - method (str, optional): The free energy method that should be used. Defaults to "MBAR".
        - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
        - decorrelate (bool, optional): Whether to decorrelate the data before estimating the free energy difference. Defaults to True.
        - coupling (bool, optional): If coupling (True) or decoupling (False) is performed. If decoupling, 
                                     multiply free energy results *-1 to get solvation free energy. Defaults to True.
        
        Returns:
            None
        
        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using alchempy.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.

    `prepare_coupling_simulation(self, folder_name: str, solute: str, combined_lambdas: List[float], coupling_settings_file: str, ensembles: List[str], simulation_times: List[float], initial_systems: List[str] = [], copies: int = 0, input_kwargs: Dict[str, Any] = {}, on_cluster: bool = False, off_set: int = 0, lammps_ff_callable: Callable[..., str] = None, ff_argument_map: Dict[str, Any] = {})`
    :   Prepares the coupling simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
        In each state, an own folder for each lambda is created. The method checks if an initial configuration file is provided. 
        If not, it generates the initial configuration based on the provided molecule numbers and PLAYMOL. 
        It then generates input files for each ensemble in a separate folder and creates a job file for each copy of the simulation.
        
        Parameters:
        - folder_name (str, optional): Name of the subfolder where to perform the simulations.
                                        Path structure is as follows: system.folder/system.name/folder_name
        - solute (str): Name of the solute that is coupled to the system. Species should be listed in system setup.
        - combined_lambdas (List[float]): Combined lambdas. If coupling is used (lambdas between 0 and 1 are vdW, 1 to 2 are Coulomb). 
                                        For decoupling (lambdas between 0 and 1 are Coulomb, 1 to 2 are vdW)
        - coupling_settings_file (str): Path to YAML file specifing certain coupling settings.
        - ensembles (List[str]): A list of ensembles to generate input files for. Definitions of each ensemble is provided in self.simulation_ensemble.
        - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
        - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
        - copies (int, optional): Number of copies for the specified system. Defaults to 0.
        - input_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the input template. Defaults to "{}".
        - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
        - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
        - lammps_ff_callable (Callable, optional): Callable to write LAMMPS force field. Defaults to None.
        - ff_argument_map (Dict[str,Any], optional): Argument mapping from local and class variables to the lammps_ff_callable function. 
        
        Returns:
            None

    `prepare_simulation(self, folder_name: str, ensembles: List[str], simulation_times: List[float], initial_systems: List[str] = [], copies: int = 0, input_kwargs: Dict[str, Any] = {}, ff_file: str = '', on_cluster: bool = False, off_set: int = 0, lammps_ff_callable: Callable[..., str] = None, ff_argument_map: Dict[str, Any] = {})`
    :   Prepares the simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
        The method checks if an initial configuration file is provided. 
        If not, it generates the initial configuration based on the provided molecule numbers and PLAYMOL. 
        It then generates input files for each ensemble in a separate folder and creates a job file for each copy of the simulation.
        
        Parameters:
         - folder_name (str, optional): Name of the subfolder where to perform the simulations.
                                        Path structure is as follows: system.folder/system.name/folder_name
         - ensembles (List[str]): A list of ensembles to generate input files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - input_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the input template. Defaults to "{}".
         - ff_file (str, optional): Path to LAMMPS force field file. Defaults to "".
         - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
         - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
         - lammps_ff_callable (Callable, optional): Callable to write LAMMPS force field. Defaults to None.
         - ff_argument_map (Dict[str,Any], optional): Argument mapping from local and class variables to the lammps_ff_callable function. 
        
        Returns:
            None

    `submit_simulation(self)`
    :   Function that submits predefined jobs to the cluster.
        
        Parameters:
            None
        
        Returns:
            None