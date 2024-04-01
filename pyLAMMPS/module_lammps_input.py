import re
import yaml
import glob
import subprocess
import pandas as pd
from itertools import groupby
from typing import Any, List, Dict, Callable
from .analysis import read_lammps_output
from .tools.general_utils import work_json, merge_nested_dicts
from .tools import ( LAMMPS_molecules, write_lammps_ff, generate_initial_configuration, 
                     generate_input_files, generate_job_file )

## to do:
# add that lammps_ff callable takes correct arguments

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
                            lammps_ff_callable: Callable[...,str]=None ):
        
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}'

        # Write LAMMPS force field file
        if not ff_file:
            # Call the LAMMPS molecule class
            lammps_molecules = LAMMPS_molecules( mol_str = [ mol["graph"] for mol in self.system_setup["molecules"] ],
                                                force_field_path = self.system_setup["paths"]["force_field_path"] 
                                                ) 
            
            # Prepare the LAMMPS force field
            lammps_molecules.prepare_lammps_force_field()

            # Get shake dictionary
            shake_dict = lammps_molecules.get_shake_indices( self.simulation_default["shake_dict"] )
            
            # Write lammps ff file. Either using the write_lammps_ff or any external provided function
            if lammps_ff_callable is not None and callable(lammps_ff_callable):
                print("External LAMMPS force field function is provided!\n")
                lammps_ff_file = lammps_ff_callable(  )
            else:
                lammps_ff_file = write_lammps_ff( ff_template = self.system_setup["paths"]["template"]["lammps_ff_file"], 
                                                lammps_ff_path = f"{sim_folder}/force_field.params", 
                                                potential_kwargs = { **self.simulation_default["non_bonded"]["vdw_style"], 
                                                                     **self.simulation_default["non_bonded"]["coulomb_style"] },
                                                atom_numbers_ges = lammps_molecules.atom_numbers_ges, 
                                                nonbonded = lammps_molecules.nonbonded, 
                                                bond_numbers_ges = lammps_molecules.bond_numbers_ges, 
                                                bonds = lammps_molecules.bonds,
                                                angle_numbers_ges = lammps_molecules.angle_numbers_ges, 
                                                angles = lammps_molecules.angles,
                                                torsion_numbers_ges = lammps_molecules.torsion_numbers_ges, 
                                                torsions = lammps_molecules.torsions,
                                                only_self_interactions = self.simulation_default["non_bonded"]["lammps_mixing"], 
                                                mixing_rule = self.simulation_default["non_bonded"]["mixing"],
                                                ff_kwargs = self.simulation_default["non_bonded"]
                                                )
        else:
            lammps_ff_file = ff_file

            # Use directly the shake dict 
            shake_dict = { "t": self.simulation_default["shake_dict"]["atoms"],
                           "b": self.simulation_default["shake_dict"]["bonds"], 
                           "a": self.simulation_default["shake_dict"]["angles"], 
                         }
            
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
                                                                   molecules_dict_list = self.system_setup["molecules"],
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
                                                               "shake_dict": shake_dict, 
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
                                     output_suffix: str, fraction: float=0.0 ):
        """
        Extracts properties from LAMMPS output files for a specific ensemble.

        Parameters:
            analysis_folder (str): The name of the folder where the analysis will be performed.
            ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
            extracted_properties (List[str]): A list of properties to be extracted from the LAMMPS output files.
            output_suffix (str): Suffix of the LAMMPS output file to be analyzed.
            fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.

        Returns:
            None

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

        # Search output files and sort them after temperature / pressure and then copy
        for i, (temperature, pressure) in enumerate( zip( self.system_setup["temperature"], 
                                                          self.system_setup["pressure"]
                                                    ) ):
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.1f}_pres_{pressure:.1f}"

            # Search for available copies
            files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.{output_suffix}" )
            files.sort( key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ) 
        
            if len(files) == 0:
                raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}")

            print(f"Temperature: {temperature}, Pressure: {pressure}\n   "+"\n   ".join(files) + "\n")

            data_list = []

            for file in files:
                # Analysis data
                data_list.append( read_lammps_output( file = file, 
                                                      keys = extracted_properties, 
                                                      fraction = fraction, 
                                                      average = False ) )

            if len(data_list) == 0:
                raise KeyError("No data was extracted!")
            
            # Mean the values for each copy and exctract mean and standard deviation
            mean_std_list  = [df.iloc[:, 1:].agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'property'}) for df in data_list]
            
            # Extract units from the property column and remove it from the label and make an own unit column
            for df in mean_std_list:
                df['unit']     = df['property'].str.extract(r'\((.*?)\)')
                df['property'] = [ p.split('(')[0].strip() for p in df['property'] ]

            final_df           = pd.concat(mean_std_list,axis=0).groupby("property", sort=False)["mean"].agg(["mean","std"]).reset_index()
            final_df["unit"]   = df["unit"]

            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json
            json_data = { f"copy_{i}": { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in df.to_dict(orient="records") } for i,df in enumerate(mean_std_list) }
            json_data["average"] = { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in final_df.to_dict(orient="records") }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"
            
            work_json( json_path, { "temperature": temperature, "pressure": pressure,
                                    ensemble: { "data": json_data, "paths": files, "fraction_discarded": fraction } }, "append" )
        
            # Add the extracted values for the analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, pressure): { analysis_folder: { ensemble: final_df }  } } )