import os
import re
import yaml
import glob
import subprocess
import numpy as np
import pandas as pd
import multiprocessing

from rdkit import Chem
from jinja2 import Template
from itertools import groupby
from typing import List, Dict, Any
from .analysis import read_gromacs_xvg
from rdkit.Chem.Descriptors import MolWt
from pyLMP.MDSetup.tools.submission import submit_and_wait
from pyLMP.MDSetup.tools.general import ( work_json, merge_nested_dicts, 
                                  get_system_volume
                                )
from .analysis.solvation_free_energy import ( get_free_energy_difference,
                                              extract_combined_states,
                                              visualize_dudl
                                            )
from .tools import ( GROMACS_molecules, generate_initial_configuration, 
                     generate_mdp_files, generate_job_file, change_topo,
                     write_gro_files
                    )

def extract_function(file):
    subprocess.run(["bash",file], capture_output=True)

# Precision to write folders
FOLDER_PRECISION = 1

# Precision for job names
JOB_PRECISION = 0

class GROMACS_setup:

    def __init__( self, system_setup: str, simulation_default: str, simulation_ensemble: str, submission_command: str="qsub" ) -> None:
        """
        Initialize a new instance of the gromacs class.

        Parameters:
         - system_setup (str): Path to the system setup YAML file. Containing all system settings.
         - simulation_default (str): Path to the simulation default YAML file. Containing all default GROMACS settings.
         - simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all GROMACS ensemble settings (except temperature, pressure, and compressibility).
         - submission_command (str, optional): Command to submit jobs to cluster. Defaults to "qsub".
        
        Returns:
            None
        """
        # Save paths of the yaml files.
        self.system_setup_path        = os.path.abspath( system_setup )
        self.simulation_default_path  = os.path.abspath( simulation_default )
        self.simulation_ensemble_path = os.path.abspath( simulation_ensemble )
        self.submission_command       = submission_command

        with open( system_setup ) as file: 
            self.system_setup    = yaml.safe_load(file)
        
        with open( simulation_default ) as file:
            self.simulation_default  = yaml.safe_load(file)

        with open( simulation_ensemble ) as file:
            self.simulation_ensemble = yaml.safe_load(file)

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

    def write_topology( self ):

        print("\nUtilize moleculegraph to generate itp, topology and initial gro files of every molecule in the system!\n")

        topology_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/topology'

        os.makedirs( topology_folder, exist_ok = True)

        if not any( self.system_setup["paths"]["force_field_paths"] ):
            raise KeyError("No force field paths provided in the system setup yaml file!")            

        gmx_mol = GROMACS_molecules( mol_str = [ mol["graph"] for mol in self.system_setup["molecules"] ],
                                     force_field_paths = self.system_setup["paths"]["force_field_paths"] 
                                    ) 

        itp_files = gmx_mol.write_gromacs_itp( itp_template = self.system_setup["paths"]["template"]["itp_file"],
                                               itp_path = topology_folder, 
                                               residue= [ mol["name"] for mol in self.system_setup["molecules"] ], 
                                               nrexcl = [ mol["nrexcl"] for mol in self.system_setup["molecules"] ] 
                                            )

        topology_file = gmx_mol.write_gromacs_top( top_template = self.system_setup["paths"]["template"]["top_file"], 
                                                   top_path = f'{topology_folder}/{self.system_setup["name"]}.top', 
                                                   comb_rule = self.simulation_default["non_bonded"]["comb_rule"],
                                                   system_name = self.system_setup["name"], 
                                                   itp_files = itp_files,
                                                   residue_dict = { mol["name"]: mol["number"] for mol in self.system_setup["molecules"] }, 
                                                   fudgeLJ = self.simulation_default["non_bonded"]["fudgeLJ"], 
                                                   fudgeQQ = self.simulation_default["non_bonded"]["fudgeQQ"]
                                                 )

        # Generate gro files for every molecule 
        coordinate_paths = write_gro_files( destination = topology_folder,
                                            molecules_dict_list = self.system_setup["molecules"],
                                            gro_template = self.system_setup["paths"]["template"]["gro_file"]
                                          )

        print("Done! Topology paths and molecule coordinates are added within the class.\n")

        # adapt system yaml with new paths
        self.system_setup["paths"]["topology"] = topology_file
        self.system_setup["paths"]["coordinates"] = coordinate_paths

    def prepare_simulation(self, ensembles: List[str], simulation_times: List[float], initial_systems: List[str]=[],
                           copies: int=0, folder_name: str="md", mdp_kwargs: Dict[str, Any]={}, 
                           on_cluster: bool=False, off_set: int=0, init_step: int=0 ):
        """
        Prepares the simulation by generating job files for each temperature, pressure, and compressibility combination specified in the simulation setup.
        The method checks if an initial configuration file is provided. If not, it generates the initial configuration based on the provided molecule numbers and coordinate files. 
        It then generates MDP files for each ensemble in a separate folder and creates a job file for each copy of the simulation.

        Parameters:
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state. Defaults to [].
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - folder_name (str, optional): Name of the subfolder where to perform the simulations. Defaults to "md".
                                        Path structure is as follows: system.folder/system.name/folder_name
         - mdp_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the mdp template. Defaults to "{}".
         - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
         - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
         - init_step (int, optional): Starting step if first simulation should be extended. Defaults to 0.

        Returns:
            None
        """

        if not self.system_setup["paths"]["topology"]:
            raise KeyError("No topology file specified!\n")
        elif not all( self.system_setup["paths"]["coordinates"] ):
            raise KeyError("Not every molecule in the system has a coordinate file specified!\n")
        
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}'

        # Copy tolopoly to the box folder and change it according to system specification
        initial_topo = change_topo( topology_path = self.system_setup["paths"]["topology"], 
                                    topology_out = f'{sim_folder}/toplogy/{self.system_setup["name"]}.top', 
                                    molecules_list = self.system_setup["molecules"],
                                    system_name = self.system_setup["name"]
                                    )

        # Get molecular mass and number for each molecule
        molar_masses = [ MolWt( Chem.MolFromSmiles( mol["smiles"] ) ) for mol in self.system_setup["molecules"] ]
        molecule_numbers = [ mol["number"] for mol in self.system_setup["molecules"] ]

        for i, (temperature, pressure, compressibility, density) in enumerate( zip( self.system_setup["temperature"], 
                                                                                    self.system_setup["pressure"], 
                                                                                    self.system_setup["compressibility"], 
                                                                                    self.system_setup["density"]  ) ):
            
            job_files = []

            print(f"State: T = {temperature:.{FOLDER_PRECISION}f} K, P = {pressure:.{FOLDER_PRECISION}f} bar")

            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            if not initial_systems:
                
                
                # Get intial box lenghts using density estimate
                box_size = get_system_volume( molar_masses = molar_masses, 
                                              molecule_numbers = molecule_numbers, 
                                              density = density, 
                                              box_type = self.system_setup["box"]["type"],
                                              z_x_relation = self.system_setup["box"]["z_x_relation"], 
                                              z_y_relation= self.system_setup["box"]["z_y_relation"]
                                            )
                
                # Box size are given back as -L/2, L/2 in A (convert to nm)
                box_lenghts = [ round(np.abs(value).sum()/10,5) for _,value in box_size.items() ]

                print("\nBuilding system based on provided molecule numbers and coordinate files!\n" )
                initial_coord = generate_initial_configuration( build_template = self.system_setup["paths"]["template"]["build_system_file"],
                                                                destination_folder = state_folder, 
                                                                coordinate_paths = self.system_setup["paths"]["coordinates"], 
                                                                molecules_list = self.system_setup["molecules"], box_lenghts = box_lenghts,
                                                                submission_command = self.submission_command, on_cluster = on_cluster )
                
                # As system is newly build, no checkpoint file is available
                initial_system = initial_coord
                initial_cpt = ""
            else:
                initial_system = initial_systems[i]
                print(f"\nSystem already build and initial configuration is provided at:\n   {initial_system}\n")
                flag_cpt = os.path.exists( initial_system.replace( initial_system.split(".")[-1], "cpt") )
                if flag_cpt:
                    print(f"Checkpoint file (.cpt) is provided in the same folder.\n")
                
                if init_step > 0 and not flag_cpt:
                    raise KeyError("Extension of simulation is intended, but no checkpoint file is provided!")

                # If system is already build assume the cpt file is in the same folder as the coordinates.
                initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""
            
            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{state_folder}/copy_{copy}"

                # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                mdp_files = generate_mdp_files( destination_folder = copy_folder, mdp_template = self.system_setup["paths"]["template"]["mdp_file"],
                                                ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                compressibility = compressibility, simulation_times = simulation_times,
                                                dt = self.simulation_default["system"]["dt"], kwargs = { **mdp_kwargs, **self.simulation_default }, 
                                                ensemble_definition = self.simulation_ensemble, off_set = off_set,
                                                init_step = init_step )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, job_template = self.system_setup["paths"]["template"]["job_file"], mdp_files = mdp_files, 
                                                     intial_coord = initial_system, initial_topo = initial_topo, job_name = f'{self.system_setup["name"]}_{temperature:.{JOB_PRECISION}f}',
                                                     job_out = f"job_{temperature:.{JOB_PRECISION}f}.sh", ensembles = ensembles, initial_cpt = initial_cpt, off_set = off_set,
                                                     extend_sim = init_step > 0 ) )
            self.job_files.append( job_files )
    
    def prepare_free_energy_simulation( self, folder_name: str, simulation_free_energy: str, solute: str, combined_lambdas: List[float], initial_systems: List[str], 
                                        ensembles: List[str], simulation_times: List[float], copies: int=0,
                                        on_cluster: bool=False, precision: int=3, off_set: int=0, init_step: int=0 ):
        """
        Function that prepares free energy simulations for several temperatures and pressure states.

        Parameters:
         - folder_name (str): Name of the subfolder where to perform the simulations.
                              Path structure is as follows: system.folder/system.name/folder_name
         - simulation_free_energy (str): Path to the simulation free energy YAML file. Containing all free energy settings.
         - combined_lambdas (List[float]): List of combined lambdas for the free energy simulations.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
         - precision (int, optional): The number of decimals of the lambdas. Defaults to 3.
         - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
         - init_step (int, optional): Starting step if first simulation should be extended. Defaults to 0.

        Returns:
            None

        The method loads the free energy settings from the provided YAML file and overwrites the provided lambdas with the combined lambdas.
        For each temperature, pressure, and compressibility combination specified in the simulation setup, the method prepares a separate folder for each copy. 
        Within each copy folder, it creates a folder for each lambda value. It then generates MDP files for each ensemble in the lambda folder and creates a job file for each copy of the simulation.

        """
        self.job_files = []

        # Check if solute exists as species
        current_name_list = [ mol["name"] for mol in self.system_setup["molecules"] ]

        if not solute in current_name_list:
            raise KeyError("Provided solute species is not presented in the system setup! Available species are:\   ",", ".join(current_name_list) )

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}/{solute}'

        # Load free energy settings
        with open( simulation_free_energy ) as file: 
            simulation_free_energy = yaml.safe_load(file)

        # Overwrite provided lambdas in free energy settings
        simulation_free_energy.update( { "init_lambda_states": "".join([f"{x:.{JOB_PRECISION}f}" + " "*(precision+2) if x < 10 else f"{x:.{JOB_PRECISION}f}" + " "*(precision+1) for x,_ in enumerate(combined_lambdas)]), 
                                         "vdw_lambdas": " ".join( [ f"{max(l-1,0.0):.{precision}f}" for l in combined_lambdas] ), 
                                         "coul_lambdas": " ".join( [ f"{min(l,1.0):.{precision}f}" for l in combined_lambdas] ),
                                         "couple_moltype": solute } )
        
        # Add free energy settings to overall simulation input
        self.simulation_default["free_energy"] = simulation_free_energy

        # Copy tolopoly to the box folder and change it according to system specification
        initial_topo = change_topo( topology_path = self.system_setup["paths"]["topology"], 
                                    topology_out = f'{sim_folder}/toplogy/{self.system_setup["name"]}.top', 
                                    molecules_list = self.system_setup["molecules"],
                                    system_name = self.system_setup["name"]
                                    )

        # Get molecular mass and number for each molecule
        molar_masses = [ MolWt( Chem.MolFromSmiles( mol["smiles"] ) ) for mol in self.system_setup["molecules"] ]
        molecule_numbers = [ mol["number"] for mol in self.system_setup["molecules"] ]
        
        # Prepare free energy simulations for several temperatures & pressure states
        for i, (temperature, pressure, compressibility, density) in enumerate( zip( self.system_setup["temperature"], 
                                                                                    self.system_setup["pressure"], 
                                                                                    self.system_setup["compressibility"], 
                                                                                    self.system_setup["density"]  ) ):
            
            job_files = []

            print(f"State: T = {temperature:.{FOLDER_PRECISION}f} K, P = {pressure:.{FOLDER_PRECISION}f} bar")

            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            if not initial_systems:

                # Get intial box lenghts using density estimate
                box_size = get_system_volume( molar_masses = molar_masses, 
                                              molecule_numbers = molecule_numbers, 
                                              density = density, 
                                              box_type = self.system_setup["box"]["type"],
                                              z_x_relation = self.system_setup["box"]["z_x_relation"], 
                                              z_y_relation= self.system_setup["box"]["z_y_relation"]
                                            )
                # Box size are given back as -L/2, L/2 in A (convert to nm)
                box_lenghts = [ round(np.abs(value).sum()/10,5) for _,value in box_size.items() ]

                print("\nBuilding system based on provided molecule numbers and coordinate files!\n" )
                initial_coord = generate_initial_configuration( build_template = self.system_setup["paths"]["template"]["build_system_file"],
                                                                destination_folder = state_folder, 
                                                                coordinate_paths = self.system_setup["paths"]["coordinates"], 
                                                                molecules_list = self.system_setup["molecules"], box_lenghts = box_lenghts,
                                                                submission_command = self.submission_command, on_cluster = on_cluster )
                
                # As system is newly build, no checkpoint file is available
                initial_system = initial_coord
                initial_cpt = ""
            else:
                initial_system = initial_systems[i]
                print(f"\nSystem already build and initial configuration is provided at:\n   {initial_system}\n")
                flag_cpt = os.path.exists( initial_system.replace( initial_system.split(".")[-1], "cpt") )
                if flag_cpt:
                    print(f"Checkpoint file (.cpt) is provided in the same folder.\n")
                
                if init_step > 0 and not flag_cpt:
                    raise KeyError("Extension of simulation is intended, but no checkpoint file is provided!")

                # If system is already build assume the cpt file is in the same folder as the coordinates.
                initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""

            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}/copy_{copy}"

                # Define for each lambda an own folder
                for i,_ in enumerate(combined_lambdas):
                    self.simulation_default["free_energy"]["init_lambda_state"] = i

                    lambda_folder = f"{copy_folder}/lambda_{i}"

                    # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                    mdp_files = generate_mdp_files( destination_folder = lambda_folder, mdp_template = self.system_setup["paths"]["template"]["mdp_file"],
                                                    ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                    compressibility = compressibility, simulation_times = simulation_times,
                                                    dt = self.simulation_default["system"]["dt"], kwargs = self.simulation_default, 
                                                    ensemble_definition = self.simulation_ensemble, init_step = init_step,
                                                    off_set = off_set )
                
                    # Create job file
                    job_files.append( generate_job_file( destination_folder = lambda_folder, job_template = self.system_setup["paths"]["template"]["job_file"], 
                                                         mdp_files = mdp_files, intial_coord = initial_system, initial_topo = initial_topo,
                                                         job_name = f'{self.system_setup["name"]}_{temperature:.{JOB_PRECISION}f}_lambda_{i}', job_out = f"job_{temperature:.{JOB_PRECISION}f}_lambda_{i}.sh",
                                                         ensembles = ensembles, initial_cpt = initial_cpt, off_set = off_set, extend_sim = init_step > 0 ) )
            
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
            print(f"\nSubmitting simulations at Temperature = {temperature:.{FOLDER_PRECISION}f} K, Pressure = {pressure:.{FOLDER_PRECISION}f} bar\n")

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run( [self.submission_command, job_file] )
                print("\n")

    def analysis_extract_properties( self, analysis_folder: str, ensemble: str, extracted_properties: List[str], command: str="energy", fraction: float=0.0, 
                                     args: List[str]=[""], output_name: str="properties", on_cluster: bool=False, extract: bool= True ):
        """
        Extracts properties from GROMACS output files for a specific ensemble.

        Parameters:
            analysis_folder (str): The name of the folder where the analysis will be performed.
            ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
            extracted_properties (List[str]): A list of properties to be extracted from the GROMACS output files.
            command (str, optional): The GROMACS command to be used for property extraction. Defaults to "energy".
            fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
            args (List[str], optional): A list of strings that will be added to the gmx command after "-f ... -s ..." and before "-o ...". Defaulst to [""]
            output_name ( str, optional): Name of the resulting xvg output file. Defaults to "properties".
            on_cluster (bool, optional): If the GROMACS analysis should be submited to the cluster. Defaults to "False".
            extract (bool, optional): If the values are already extracted from GROMACS and only the xvg files should be read out, 
                                      this should be set to False. Defaults to "True".

        Returns:
            None

        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using the specified GROMACS command.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.
        """
        
        if not os.path.exists(self.system_setup["paths"]["template"]["extract_property_file"]):
            raise FileExistsError(f'Provided extract template does not exists: {self.system_setup["paths"]["template"]["extract_property_file"]}')
        else:
            with open(self.system_setup["paths"]["template"]["extract_property_file"]) as f:
                template = Template(f.read())
                
        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])

        if extract:
            bash_files = []

            # Search output files and sort them after temperature / pressure and then copy
            for temperature, pressure in zip( self.system_setup["temperature"], 
                                              self.system_setup["pressure"]
                                            ):
                
                # Define folder for specific temp and pressure state
                state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

                # Search for available copies
                files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.edr" )
                files.sort( key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ) 

                if len(files) == 0:
                    raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}")
            
                # Iterate through the files and write extraction bash file
                for path in files:
                    
                    rendered = template.render( folder = os.path.dirname( path ),
                                                extracted_properties = extracted_properties,
                                                gmx_command = f"{command} -f {ensemble_name} -s {ensemble_name} {' '.join(args)} -o {output_name}" )

                    bash_file = f"{os.path.dirname( path )}/extract_properties.sh"

                    with open( bash_file, "w") as f: 
                        f.write( rendered )

                    bash_files.append( bash_file )
        
        if on_cluster and extract:
            print( "Submit extraction to cluster:\n" )
            print( '\n'.join(bash_files), "\n" )
            print( "Wait until extraction is done..." )
            submit_and_wait( bash_files, submission_command = self.submission_command )
        elif not on_cluster and extract:
            print("Extract locally\n")
            print( '\n'.join(bash_files), "\n" )
            print( "Wait until extraction is done..." )
            num_processes = multiprocessing.cpu_count()
            # In case there are multiple CPU's, leave one without task
            if num_processes > 1:
                num_processes -= 1

            # Create a pool of processes
            pool = multiprocessing.Pool( processes = multiprocessing.cpu_count()  )

            # Execute the tasks in parallel
            data_list = pool.map(extract_function, bash_files)

            # Close the pool to free up resources
            pool.close()
            pool.join()

        if extract:
            print( f"Extraction finished!\n\n" )

        for temperature, pressure in zip( self.system_setup["temperature"], 
                                          self.system_setup["pressure"]
                                        ):
            print(f"Temperature: {temperature}, Pressure: {pressure}\n   ")
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{output_name}.xvg" )
            files.sort( key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ) 

            if len(files) == 0:
                raise FileExistsError("Extracting the properties did not work for:\n   " + "\n   ".join(files) + "\n")

            data_list = [ read_gromacs_xvg( file_path = file, fraction = fraction) for file in files ]

            # Get the mean and std of each property over time
            mean_std_list = []
            for df in data_list:
                df = df.drop(columns=['Time (ps)'])
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
            
            work_json( json_path, { "temperature": temperature, "pressure": temperature,
                                    ensemble: { "data": json_data, "paths": files, "fraction_discarded": fraction } }, "append" )
        
            # Add the extracted values for the command, analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, temperature): { analysis_folder: { ensemble: final_df } } } )
    

    def analysis_free_energy( self, analysis_folder: str, solute: str, ensemble: str, 
                              method: str="MBAR", fraction: float=0.0, 
                              decorrelate: bool=True, visualize: bool=False,
                              coupling: bool=True ):
        """
        Extracts free energy difference for a specified folder and solute and ensemble.

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
        """
        
        # Check if solute species is present in the system
        current_name_list = [ mol["name"] for mol in self.system_setup["molecules"] ]

        if not solute in current_name_list:
            raise KeyError("Provided solute species is not presented in the system setup! Available species are:\   ",", ".join(current_name_list) )

        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}/{solute}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])
        
        print(f"\nExtract solvation free energy results for solute: {solute}\n")

        # sorting patterns
        copy_pattern = re.compile( r'copy_(\d+)')
        lambda_pattern = re.compile( r'lambda_(\d+)')

        # Loop over each temperature & pressure state
        for temperature, pressure in zip( self.system_setup["temperature"], self.system_setup["pressure"] ):
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            # Search for available copies
            files = glob.glob( f"{state_folder}/copy_*/lambda_*/{ensemble}/{ensemble_name}.xvg" )
            files.sort( key=lambda x: ( int(copy_pattern.search(x).group(1)), 
                                        int(lambda_pattern.search(x).group(1))
                                    )
                    )


            if len(files) == 0:
                raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}")

            print(f"Temperature: {temperature} K, Pressure: {pressure} bar\n   "+"\n   ".join(files) + "\n")
            
            # Sort in copies 
            mean_std_list = [ get_free_energy_difference(list(copy_files), temperature, method, fraction, decorrelate, coupling) for 
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

            print(f"\nFollowing combined lambda states were analysed with the '{method}' method:\n   {', '.join([str(l) for l in combined_states])}")
            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json
            json_data = { f"copy_{i}": { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in df.to_dict(orient="records") } for i,df in enumerate(mean_std_list) }
            json_data["average"] = { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in final_df.to_dict(orient="records") }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"
            
            work_json( json_path, { "temperature": temperature, "pressure": pressure,
                                    ensemble: { method : { "data": json_data, "paths": files, "fraction_discarded": fraction, 
                                                "decorrelate": decorrelate, "combined_states": combined_states } } }, "append" )
        
            # Add the extracted values for the analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, pressure): { analysis_folder: { ensemble: final_df }  } } )