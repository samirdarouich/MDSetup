import re
import os
import yaml
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from itertools import groupby
from .analysis import read_lammps_output
from typing import Any, List, Dict, Callable
from .tools.general_utils import work_json, merge_nested_dicts, map_function_input, flatten_list
from pyLAMMPS.analysis.solvation_free_energy import ( get_free_energy_difference, visualize_dudl, 
                                                      extract_combined_states )
from .tools import ( LAMMPS_molecules, generate_initial_configuration, 
                     generate_input_files, generate_job_file, 
                     write_lammps_ff, write_coupled_lammps_ff, 
                     write_fep_sampling )

## to do:
# add decorrelation in postprocessing

class LAMMPS_setup():
    """
    This class sets up structured and FAIR LAMMPS simulations. It also has the capability to build a system based on a list of molecules.
    """

    def __init__( self, system_setup: str, simulation_default: str, simulation_ensemble: str, 
                  simulation_sampling : str, submission_command: str):
        """
        Initialize a new instance of the LAMMPS_setup class.

        Parameters:
         - system_setup (str): Path to the system setup YAML file. Containing all system settings.
         - simulation_default (str): Path to the simulation default YAML file. Containing all default LAMMPS settings.
         - simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all LAMMPS ensemble settings.
         - simulation_sampling (str): Path to the sampling YAML file. Containing all sampling settings.
         - submission_command (str, optional): Command to submit jobs to cluster. Defaults to "qsub".
        
        Returns:
            None
        """

        # Open the yaml files and extract the necessary information
        with open( system_setup ) as file: 
            self.system_setup        = yaml.safe_load(file)
        
        with open( simulation_default ) as file:
            self.simulation_default  = yaml.safe_load(file)

        with open( simulation_ensemble ) as file:
            self.simulation_ensemble = yaml.safe_load(file)

        with open( simulation_sampling ) as file:
            self.simulation_sampling = yaml.safe_load(file)

        self.submission_command      = submission_command

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

    def prepare_simulation( self, folder_name: str, ensembles: List[str], simulation_times: List[float],
                            initial_systems: List[str]=[], copies: int=0, input_kwargs: Dict[str, Any]={}, 
                            ff_file: str="", on_cluster: bool=False, off_set: int=0, 
                            lammps_ff_callable: Callable[...,str]=None, ff_argument_map : Dict[str, Any]={} ):
        """
        Prepares the simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
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
        """
        self.job_files = []

        # Predefine the force field argument map
        if not ff_argument_map:
            ff_argument_map = { "ff_template": "system_setup.paths.template.lammps_ff_file",
                                "lammps_ff_path": "lammps_ff_path",
                                "potential_kwargs": "simulation_default.nonbonded",
                                "atom_numbers_ges": "lammps_molecules.atom_numbers_ges",
                                "nonbonded": "lammps_molecules.nonbonded", 
                                "bond_numbers_ges": "lammps_molecules.bond_numbers_ges", 
                                "bonds": "lammps_molecules.bonds",
                                "angle_numbers_ges": "lammps_molecules.angle_numbers_ges", 
                                "angles": "lammps_molecules.angles",
                                "torsion_numbers_ges": "lammps_molecules.torsion_numbers_ges", 
                                "torsions": "lammps_molecules.torsions",
                                "only_self_interactions": "simulation_default.nonbonded.lammps_mixing", 
                                "mixing_rule": "simulation_default.nonbonded.mixing",
                                "ff_kwargs": "simulation_default.nonbonded",
                                "shake_dict": "shake_dict"
                            }
        
        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}'

        # Write LAMMPS force field file
        if not ff_file:
            # Call the LAMMPS molecule class
            # Sort out molecules that are not present in system
            system_molecules = [ mol for mol in self.system_setup["molecules"] if mol["number"] > 0 ]

            lammps_molecules = LAMMPS_molecules( mol_str = [ mol["graph"] for mol in system_molecules ],
                                                 force_field_paths = self.system_setup["paths"]["force_field_paths"] 
                                                ) 
            
            # Prepare the LAMMPS force field
            lammps_molecules.prepare_lammps_force_field()

            # Get shake dictionary
            shake_dict = lammps_molecules.get_shake_indices( self.simulation_default["shake_dict"] )
            
            # Write lammps ff file. Either using the write_lammps_ff or any external provided function
            lammps_ff_path = f"{sim_folder}/force_field.params"
            
            # Get all class attributes and local defined variables
            all_attributes =  { **vars(self), **locals() }
            
            # Get force field input arguments
            ff_input = map_function_input( all_attributes = all_attributes, argument_map = ff_argument_map )

            if callable(lammps_ff_callable):
                print("External function to write LAMMPS force field is provided!\n")
                lammps_ff_file = lammps_ff_callable( **ff_input )
            else:
                lammps_ff_file = write_lammps_ff( **ff_input )
        else:
            #
            if not os.path.exists( ff_file ):
                raise FileExistsError(f"Provided force field file does not exist:\n   '{ff_file}'")
                
            # Copy provided force field file to simulation folder
            os.makedirs( sim_folder, exist_ok = True )
            lammps_ff_file = shutil.copy( ff_file, sim_folder )

            
        for i, (temperature, pressure, density) in enumerate( zip( self.system_setup["temperature"], 
                                                                   self.system_setup["pressure"], 
                                                                   self.system_setup["density"] ) ):
            
            job_files = []
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.1f}_pres_{pressure:.1f}"

            # Build system with PLAYMOL and write LAMMPS data if no initial system is provided
            if not initial_systems:
                
                lammps_data_file = generate_initial_configuration( lammps_molecules = lammps_molecules,
                                                                   destination_folder = state_folder,
                                                                   molecules_dict_list = system_molecules,
                                                                   density = density,
                                                                   template_xyz = self.system_setup["paths"]["template"]["xyz_file"],
                                                                   playmol_ff_template = self.system_setup["paths"]["template"]["playmol_ff_file"],
                                                                   playmol_input_template = self.system_setup["paths"]["template"]["playmol_input_file"],
                                                                   playmol_bash_file = self.system_setup["paths"]["template"]["playmol_bash_file"],
                                                                   lammps_data_template = self.system_setup["paths"]["template"]["lammps_data_file"],
                                                                   submission_command = self.submission_command, 
                                                                   on_cluster = on_cluster
                                                                )
            
                flag_restart = False
            else:
                lammps_data_file = initial_systems[i]
                print(f"\nIntial system provided for at: {lammps_data_file}\n")
                flag_restart = ".restart" in lammps_data_file
                if flag_restart: 
                    print("Restart file is provided. Continue simulation from there!\n")

            # Define folder for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{state_folder}/copy_{copy}"

                # Produce input files (for each ensemble an own folder 0x_ensemble)
                input_files = generate_input_files( destination_folder = copy_folder, 
                                                    input_template = self.system_setup["paths"]["template"]["lammps_input_file"],
                                                    ensembles = ensembles, 
                                                    temperature = temperature, 
                                                    pressure = pressure,
                                                    data_file = lammps_data_file, 
                                                    ff_file = lammps_ff_file,
                                                    simulation_times = simulation_times,
                                                    dt = self.simulation_default["system"]["dt"], 
                                                    kwargs = { **self.simulation_default,
                                                               **self.simulation_sampling, 
                                                               **input_kwargs,
                                                               "restart_flag": flag_restart }, 
                                                    ensemble_definition = self.simulation_ensemble,
                                                    off_set = off_set
                                                    )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, 
                                                     job_template = self.system_setup["paths"]["template"]["job_file"], 
                                                     input_files = input_files, 
                                                     ensembles = ensembles,
                                                     job_name = f'{self.system_setup["name"]}_{temperature:.0f}_{pressure:.0f}',
                                                     job_out = f"job_{temperature:.0f}_{pressure:.0f}.sh", 
                                                     off_set = off_set 
                                                    ) 
                                )
                
            self.job_files.append( job_files )

    def prepare_coupling_simulation( self, folder_name: str, solute: str, combined_lambdas: List[float],
                                    coupling_settings_file: str, ensembles: List[str], simulation_times: List[float],
                                    initial_systems: List[str]=[], copies: int=0, input_kwargs: Dict[str, Any]={}, 
                                    on_cluster: bool=False, off_set: int=0,
                                    lammps_ff_callable: Callable[...,str]=None, ff_argument_map : Dict[str, Any]={} ):
        """
        Prepares the coupling simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
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
        """

        if not ff_argument_map:
            ff_argument_map = { "ff_template": "system_setup.paths.template.lammps_ff_file",
                                "lammps_ff_path": "lammps_ff_path",
                                "potential_kwargs": "simulation_default.nonbonded",
                                "coupling_soft_core": "coupling_settings.soft_core",
                                "solute_numbers": "solute_numbers",
                                "combined_lambdas": "combined_lambdas",
                                "coupling_potential": "coupling_settings.potential",
                                "atom_numbers_ges": "lammps_molecules.atom_numbers_ges",
                                "nonbonded": "lammps_molecules.nonbonded", 
                                "bond_numbers_ges": "lammps_molecules.bond_numbers_ges", 
                                "bonds": "lammps_molecules.bonds",
                                "angle_numbers_ges": "lammps_molecules.angle_numbers_ges", 
                                "angles": "lammps_molecules.angles",
                                "torsion_numbers_ges": "lammps_molecules.torsion_numbers_ges", 
                                "torsions": "lammps_molecules.torsions",
                                "coupling": "coupling_settings.coupling",
                                "precision": "coupling_settings.precision",
                                "mixing_rule": "simulation_default.nonbonded.mixing",
                                "shake_dict": "shake_dict",
                                "ff_kwargs": "simulation_default.nonbonded"  
                            }
        
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}/{solute}'

        # Read in coupling settings
        if not os.path.exists(coupling_settings_file):
            raise FileExistsError(f"Coupling settings YAML could not be found at:\n   {coupling_settings_file}")
        else:
            coupling_settings = yaml.safe_load( open(coupling_settings_file) )
        
        # Add the coupled solute as first species
        current_name_list = [ mol["name"] for mol in self.system_setup["molecules"] ]

        if not solute in current_name_list:
            raise KeyError("Provided solute species is not presented in the system setup! Available species are:\   ",", ".join(current_name_list) )

        idx_solute = current_name_list.index(solute)

        # Change the number of extra solute molecules to 1
        solute_molecule_dict = self.system_setup["molecules"][idx_solute].copy()
        solute_molecule_dict["number"] = 1
        solute_molecule_dict["name"] += "_coupled"

        # Sort out molecules that are not present in system and add the solute as first molecule
        system_molecules = [ solute_molecule_dict ] + [ mol for mol in self.system_setup["molecules"] if mol["number"] > 0 ]

        # Write LAMMPS force field file

        # Call the LAMMPS molecule class
        lammps_molecules = LAMMPS_molecules( mol_str = [ mol["graph"] for mol in system_molecules ],
                                             force_field_paths = self.system_setup["paths"]["force_field_paths"]  
                                            ) 
        
        # Prepare the LAMMPS force field
        lammps_molecules.prepare_lammps_force_field()

        # Get unique type numbers and partial charges of solute
        solute_numbers = np.unique( lammps_molecules.mol_list[0].unique_atom_inverse + 1 )
        charge_list = [ [i, iatom["charge"]] for i,iatom in zip(solute_numbers, lammps_molecules.nonbonded) ]

        # Get shake dictionary
        shake_dict = lammps_molecules.get_shake_indices( self.simulation_default["shake_dict"] )
        
        # Write lammps ff file. Either using the write_lammps_ff or any external provided function
        lammps_ff_path = f"{sim_folder}/force_field.params"
        
        # Get all class attributes and local defined variables
        all_attributes =  { **vars(self), **locals() }
        
        # Get force field input arguments
        ff_input = map_function_input( all_attributes = all_attributes, argument_map = ff_argument_map )
        
        if callable(lammps_ff_callable):
            print("External function to write LAMMPS force field is provided!\n")
            lammps_ff_file = lammps_ff_callable( **ff_input )
        else:
            lammps_ff_file = write_coupled_lammps_ff( **ff_input )
        
        
        for i, (temperature, pressure, density) in enumerate( zip( self.system_setup["temperature"], 
                                                                self.system_setup["pressure"], 
                                                                self.system_setup["density"] ) ):
            
            job_files = []
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.1f}_pres_{pressure:.1f}"

            # Build system with PLAYMOL and write LAMMPS data if no initial system is provided
            if not initial_systems:
                
                lammps_data_file = generate_initial_configuration( lammps_molecules = lammps_molecules,
                                                                    destination_folder = state_folder,
                                                                    molecules_dict_list = system_molecules,
                                                                    density = density,
                                                                    template_xyz = self.system_setup["paths"]["template"]["xyz_file"],
                                                                    playmol_ff_template = self.system_setup["paths"]["template"]["playmol_ff_file"],
                                                                    playmol_input_template = self.system_setup["paths"]["template"]["playmol_input_file"],
                                                                    playmol_bash_file = self.system_setup["paths"]["template"]["playmol_bash_file"],
                                                                    lammps_data_template = self.system_setup["paths"]["template"]["lammps_data_file"],
                                                                    submission_command = self.submission_command, 
                                                                    on_cluster = on_cluster
                                                                    )
            
                flag_restart = False
            else:
                lammps_data_file = initial_systems[i]
                print(f"\nIntial system provided for at: {lammps_data_file}\n")
                flag_restart = ".restart" in lammps_data_file
                if flag_restart: 
                    print("Restart file is provided. Continue simulation from there!\n")

            # Define folder for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{state_folder}/copy_{copy}"

                # Define for each lambda an own folder
                for i,_ in enumerate(combined_lambdas):

                    lambda_folder = f"{copy_folder}/lambda_{i}"

                    fep_sampling_file  = write_fep_sampling( fep_template = coupling_settings["template"]["lammps_fep_sampling"], 
                                                             fep_outfile = f"{lambda_folder}/fep_sampling.lammps",
                                                             combined_lambdas = combined_lambdas, 
                                                             charge_list = charge_list,
                                                             current_state = i,
                                                             coupling = coupling_settings["coupling"],
                                                             precision = coupling_settings["precision"],
                                                             kwargs = coupling_settings 
                                                            )

                    # Produce input files (for each ensemble an own folder 0x_ensemble)
                    input_files = generate_input_files( destination_folder = lambda_folder, 
                                                        input_template = self.system_setup["paths"]["template"]["lammps_input_file"],
                                                        ensembles = ensembles, 
                                                        temperature = temperature, 
                                                        pressure = pressure,
                                                        data_file = lammps_data_file, 
                                                        ff_file = lammps_ff_file,
                                                        simulation_times = simulation_times,
                                                        dt = self.simulation_default["system"]["dt"], 
                                                        kwargs = { **self.simulation_default,
                                                                **self.simulation_sampling, 
                                                                **input_kwargs,
                                                                "restart_flag": flag_restart,
                                                                "init_lambda_state": i+1,
                                                                "fep_sampling_file": f"../{os.path.basename(fep_sampling_file)}"
                                                                }, 
                                                        ensemble_definition = self.simulation_ensemble,
                                                        off_set = off_set
                                                        )
                    
                    # Create job file
                    job_files.append( generate_job_file( destination_folder = lambda_folder, 
                                                        job_template = self.system_setup["paths"]["template"]["job_file"], 
                                                        input_files = input_files, 
                                                        ensembles = ensembles,
                                                        job_name = f'{self.system_setup["name"]}_{solute}_{temperature:.0f}_{pressure:.0f}',
                                                        job_out = f"job_{temperature:.0f}_{pressure:.0f}.sh", 
                                                        off_set = off_set 
                                                        ) 
                                    )
                
            self.job_files.append( job_files )
    
    def submit_simulation(self):
        """
        Function that submits predefined jobs to the cluster.
        
        Parameters:
            None

        Returns:
            None
        """
        for temperature, pressure, job_files in zip( self.system_setup["temperature"], self.system_setup["pressure"], self.job_files ):
            print(f"\nSubmitting simulations at Temperature = {temperature:.0f} K, Pressure = {pressure:.0f} bar\n")

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run( [self.submission_command, job_file] )
                print("\n")


    def analysis_extract_properties( self, analysis_folder: str, ensemble: str, extracted_properties: List[str], 
                                     output_suffix: str, fraction: float=0.0, header: int=2,
                                     header_delimiter: str= "," ):
        """
        Extracts properties from LAMMPS output files for a specific ensemble.

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
        """

        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])

        # sorting patterns
        copy_pattern = re.compile( r'copy_(\d+)')

        # Search output files and sort them after temperature / pressure and then copy
        for i, (temperature, pressure) in enumerate( zip( self.system_setup["temperature"], 
                                                          self.system_setup["pressure"]
                                                    ) ):
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.1f}_pres_{pressure:.1f}"

            # Search for available copies
            files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.{output_suffix}" )
            files.sort( key=lambda x: int(copy_pattern.search(x).group(1)) ) 
        
            if len(files) == 0:
                raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}")

            print(f"Temperature: {temperature}, Pressure: {pressure}\n   "+"\n   ".join(files) + "\n")

            # In case there are multiple CPU's, leave one without task
            num_processes = multiprocessing.cpu_count()
            if num_processes > 1:
                num_processes -= 1

            # Create a pool of processes
            pool = multiprocessing.Pool( processes = multiprocessing.cpu_count()  )

            inputs = [ (file,fraction,header,header_delimiter) for file in files]

            # Execute the tasks in parallel
            data_list = pool.starmap(read_lammps_output, inputs)

            # Close the pool to free up resources
            pool.close()
            pool.join()
        
            if len(data_list) == 0:
                raise KeyError("No data was extracted!")
            
            # Get the columns of one of the extracted data frames
            df_keys = { key.split("(")[0].strip(): i for i,key in enumerate( data_list[0].columns) }

            # Get the index of the keys to extract
            key_idx = flatten_list( df_keys.get( key, [] ) for key in extracted_properties )

            if len(key_idx) == 0:
                raise KeyError(f"Specified keys '{', '.join(extracted_properties) }' could not be extracted! Valid keys are: '{', '.join(df_keys.keys())}'")
            
            # Drop all nonrelevant columns from the data frames
            extracted_df_list = [ df.iloc[:,key_idx] for df in data_list ]

            # Get the mean and std of each property over time
            mean_std_list = []
            for df in extracted_df_list:
                df_new = df.agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'property'})
                df_new['unit'] = df_new['property'].str.extract(r'\((.*?)\)')
                df_new['property'] = [ p.split('(')[0].strip() for p in df_new['property'] ]
                mean_std_list.append(df_new)

            # Concat the copies and group by properties
            grouped_total_df = pd.concat( mean_std_list, axis=0).groupby("property", sort=False)

            # Get the mean over the copies. To get the standard deviation, propagate the std over the copies.
            mean_over_copies = grouped_total_df["mean"].mean()
            std_over_copies = grouped_total_df["std"].apply( lambda p: np.sqrt( sum(p**2) ) / len(p) )

            # Final df has the mean, std and the unit
            final_df = pd.DataFrame([mean_over_copies,std_over_copies]).T.reset_index()
            final_df["unit"] = df_new["unit"]

            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json
            json_data = { f"copy_{i}": { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in df.to_dict(orient="records") } for i,df in enumerate(mean_std_list) }
            json_data["average"] = { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in final_df.to_dict(orient="records") }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"
            
            work_json( json_path, { "temperature": temperature, "pressure": pressure,
                                    ensemble: { "data": json_data, "paths": files, "fraction_discarded": fraction} }, "append" )
        
            # Add the extracted values for the analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, pressure): { analysis_folder: { ensemble: final_df }  } } )

    def analysis_free_energy( self, analysis_folder: str, solute: str, ensemble: str, 
                              method: str="MBAR", fraction: float=0.0, 
                              decorrelate: bool=True, visualize: bool=False  ):
        """
        Extracts free energy difference for a specified folder and solute and ensemble.

        Parameters:
        - analysis_folder (str): The name of the folder where the analysis will be performed.
        - solute (str): Solute under investigation
        - ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
        - method (str, optional): The free energy method that should be used. Defaults to "MBAR".
        - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
        - decorrelate (bool, optional): Whether to decorrelate the data before estimating the free energy difference. Defaults to True.

        Returns:
            None

        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using alchempy.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.
        """
        
        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}'#/{solute}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])
        
        print(f"\nExtract solvation free energy results for solute: {solute}\n")

        # sorting patterns
        copy_pattern = re.compile( r'copy_(\d+)')
        lambda_pattern = re.compile( r'lambda_(\d+)')

        # Loop over each temperature & pressure state
        for temperature, pressure in zip( self.system_setup["temperature"], self.system_setup["pressure"] ):
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.1f}_pres_{pressure:.1f}"

            # Search for available copies
            files = glob.glob( f"{state_folder}/copy_*/lambda_*/{ensemble}/{ensemble_name}.fep" )
            files.sort( key=lambda x: ( int(copy_pattern.search(x).group(1)), 
                                        int(lambda_pattern.search(x).group(1))
                                    )
                    )


            if len(files) == 0:
                raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}")

            print(f"Temperature: {temperature}, Pressure: {pressure}\n   "+"\n   ".join(files) + "\n")
            
            # Sort in copies 
            mean_std_list = [ get_free_energy_difference(list(copy_files), temperature, method, fraction, decorrelate) for 
                            _,copy_files in groupby( files, key=lambda x: int(copy_pattern.search(x).group(1)) ) 
                            ]

            # Visualize dH/dl plots if wanted
            if method in ["TI", "TI_spline"] and visualize:
                for copy,group in groupby( files, key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ):
                    visualize_dudl( fep_files = group, T = temperature, 
                                    fraction = fraction, decorrelate = decorrelate,
                                    save_path = f"{state_folder}/copy_{copy}"  
                                )

            if len(mean_std_list) == 0:
                raise KeyError("No data was extracted!")
            
            # Concat the copies and group by properties
            grouped_total_df = pd.concat( mean_std_list, axis=0).groupby("property", sort=False)

            # Get the mean over the copies. To get the standard deviation, propagate the std over the copies.
            mean_over_copies = grouped_total_df["mean"].mean()
            std_over_copies = grouped_total_df["std"].apply( lambda p: np.sqrt( sum(p**2) ) / len(p) )

            # Final df has the mean, std and the unit
            final_df = pd.DataFrame([mean_over_copies,std_over_copies]).T.reset_index()
            final_df["unit"] = mean_std_list[0]["unit"]

            # Get the combined lambda state list
            combined_states = extract_combined_states( files )

            print(f"\nFollowing combined lambda states were analysed:\n   {', '.join([str(l) for l in combined_states])}")
            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json
            json_data = { f"copy_{i}": { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in df.to_dict(orient="records") } for i,df in enumerate(mean_std_list) }
            json_data["average"] = { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in final_df.to_dict(orient="records") }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"
            
            work_json( json_path, { "temperature": temperature, "pressure": pressure,
                                    ensemble: { method : { "data": json_data, "paths": files, "fraction_discarded": fraction, 
                                                "combined_states": combined_states } } }, "append" )
        
            # Add the extracted values for the analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, pressure): { analysis_folder: { ensemble: final_df }  } } )