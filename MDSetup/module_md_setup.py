import yaml
import glob
import subprocess
import numpy as np
import pandas as pd
import os, shutil, re

from rdkit import Chem
from typing import List, Dict, Any
from .forcefield import forcefield
from rdkit.Chem.Descriptors import MolWt
from .analysis.reader import extract_from_lammps, extract_from_gromacs
from .tools.general import work_json, merge_nested_dicts, KwargsError
from .tools.general import FOLDER_PRECISION, JOB_PRECISION, DEFAULTS
from .tools.systemsetup import (
    get_system_volume,
    generate_initial_configuration,
    generate_input_files,
    generate_job_file,
)


# To do:
# check forcefield topology (use row of distrance matrix to map?)
# check for all necessary keys in setup


class MDSetup:
    """
    This class sets up structured and FAIR molecular dynamic simulations. It also has the capability to build a system based on a list of molecules.
    """

    def __init__(
        self,
        system_setup: str,
        simulation_default: str,
        simulation_ensemble: str,
        submission_command: str,
        simulation_sampling: str = "",
    ):
        """
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
        """

        # Open the yaml files and extract the necessary information
        with open(system_setup) as file:
            self.system_setup = yaml.safe_load(file)

        with open(simulation_default) as file:
            self.simulation_default = yaml.safe_load(file)

        with open(simulation_ensemble) as file:
            self.simulation_ensemble = yaml.safe_load(file)

        if simulation_sampling:
            with open(simulation_sampling) as file:
                self.simulation_sampling = yaml.safe_load(file)
        else:
            self.simulation_sampling = {}

        # Check for all necessary keys

        # Print software
        print(f"MD input will be generated for '{self.system_setup['software']}'!")

        # Save molecules in the system (sort out molecules that are not present in system)
        self.system_molecules = [
            mol for mol in self.system_setup["molecules"] if mol["number"] > 0
        ]

        # Get the name (residue) list
        self.residues = [mol["name"] for mol in self.system_molecules]

        # Get molecular mass and number for each molecule
        self.molar_masses = [
            MolWt(Chem.MolFromSmiles(mol["smiles"])) for mol in self.system_molecules
        ]
        self.molecule_numbers = [mol["number"] for mol in self.system_molecules]

        # Get conversion from AA to nm/AA
        self.distance_conversion = (
            1 / 10
            if self.system_setup["software"] == "gromacs"
            else 1
            if self.system_setup["software"] == "lammps"
            else 1
        )

        # Submission command for the cluster
        self.submission_command = submission_command

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

    def write_topology(self):
        print(
            "\nUtilize moleculegraph to generate molecule and topology files of every molecule in the system!\n"
        )

        topology_folder = (
            f'{self.system_setup["folder"]}/{self.system_setup["name"]}/topology'
        )

        os.makedirs(topology_folder, exist_ok=True)

        if not any(self.system_setup["paths"]["force_field_paths"]):
            raise KeyError(
                "No force field paths provided in the system setup yaml file!"
            )

        # Prepare keyword arguments that are parsed to the functions
        # Add all paths to templates and all nonbonded settings to template
        kwargs = {**self.system_setup["paths"], **self.simulation_default["nonbonded"]}

        if any(filter(lambda d: "nrexcl" in d, self.system_molecules)):
            kwargs["nrexcl"] = [mol["nrexcl"] for mol in self.system_molecules]

        # Call the force field class
        ff_molecules = forcefield(
            smiles=[mol["smiles"] for mol in self.system_molecules],
            force_field_paths=self.system_setup["paths"]["force_field_files"],
        )

        # Write molecule files (including gro files in case of GROMACS)
        ff_molecules.write_molecule_files(
            molecule_template=self.system_setup["paths"]["molecule_template"],
            molecule_path=topology_folder,
            residues=self.residues,
            **kwargs,
        )

        # Add writen molecule files to kwargs
        kwargs["molecule_files"] = ff_molecules.molecule_files

        # Add resiude and its numbers to kwargs
        kwargs["residue_dict"] = {
            mol["name"]: mol["number"] for mol in self.system_setup["molecules"]
        }

        # Write topology file
        ff_molecules.write_topology_file(
            topology_template=self.system_setup["paths"]["topology_template"],
            topology_path=topology_folder,
            system_name=self.system_setup["name"] ** kwargs,
        )

        print(
            "Done! Topology paths and molecule coordinates are added within the class.\n"
        )

        # Add topology and gro files to class dictionary
        self.system_setup["paths"]["topology_file"] = ff_molecules.topology_file
        self.system_setup["paths"]["coordinates_files"] = ff_molecules.gro_files

    def prepare_simulation(
        self,
        folder_name: str,
        ensembles: List[str],
        simulation_times: List[float],
        initial_systems: List[str] = [],
        copies: int = 0,
        on_cluster: bool = False,
        off_set: int = 0,
        input_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Prepares the simulation by generating job files for each temperature and pressure combination specified in the simulation setup.
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
        """
        self.job_files = []

        # Define simulation folder
        sim_folder = (
            f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}'
        )

        # Prepare keyword arguments that are parsed to the functions
        kwargs = {
            **self.simulation_default,
            **self.simulation_sampling,
            **input_kwargs,
            **DEFAULTS[self.system_setup["software"]],
        }

        # Copy provided force field file to simulation folder and add it to input kwargs
        os.makedirs(sim_folder, exist_ok=True)
        kwargs["initial_topology"] = shutil.copy(
            self.system_setup["paths"]["topology_file"], sim_folder
        )

        for i, (temperature, pressure, density) in enumerate(
            zip(
                self.system_setup["temperature"],
                self.system_setup["pressure"],
                self.system_setup["density"],
            )
        ):
            job_files = []

            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            # Build system with MD software if none is provided
            build_folder = f"{state_folder}/build"

            if not initial_systems:
                print(
                    "\nBuilding system based on provided molecule numbers and coordinate files!\n"
                )

                # Get intial box lenghts using density estimate
                box = get_system_volume(
                    molar_masses=self.molar_masses,
                    molecule_numbers=self.molecule_numbers,
                    density=density,
                    unit_conversion=self.distance_conversion,
                    **self.system_setup["box"],
                )

                # Coordinates from molecule that are not present in the system are sorted out within the function.
                # Hence, parse the non filtered list of molecules and coordinates here.
                kwargs["initial_coord"] = generate_initial_configuration(
                    destination_folder=build_folder,
                    build_template=self.system_setup["paths"]["build_template"],
                    software=self.system_setup["software"],
                    coordinate_paths=self.system_setup["paths"]["coordinates"],
                    molecules_list=self.system_setup["molecules"],
                    box=box,
                    submission_command=self.submission_command,
                    on_cluster=on_cluster,
                    **kwargs,
                )

                kwargs["restart_flag"] = False

            else:
                os.makedirs(build_folder, exist_ok=True)
                suffix = (
                    "gro"
                    if self.system_setup["software"] == "gromacs"
                    else "data"
                    if self.system_setup["software"] == "lammps"
                    else ""
                )
                kwargs["initial_coord"] = shutil.copy(
                    initial_systems[i], f"{build_folder}/init_conf.{suffix}"
                )

                print(f"\nIntial system provided for at: {initial_systems[i]}\n")

                kwargs["restart_flag"] = ".restart" in kwargs[
                    "initial_coord"
                ] or os.path.exists(initial_systems[i].rsplit(".", 1)[0] + ".cpt")

                if kwargs["restart_flag"]:
                    print("Restart file is provided. Continue simulation from there!\n")

                    if self.system_setup["software"] == "gromacs":
                        # In case of GROMACS, add cpt as input
                        kwargs["initial_cpt"] = shutil.copy(
                            initial_systems[i].rsplit(".", 1)[0] + ".cpt", build_folder
                        )

            # Add compressibility to GROMACS input
            if self.system_setup["software"] == "gromacs":
                kwargs["compressibility"] = self.system_setup["compressibility"][i]

            # Define folder for each copy
            for copy in range(copies + 1):
                copy_folder = f"{state_folder}/copy_{copy}"

                # Produce input files (for each ensemble an own folder 0x_ensemble)
                input_files = generate_input_files(
                    destination_folder=copy_folder,
                    input_template=self.system_setup["paths"]["input_template"],
                    software=self.system_setup["software"],
                    ensemble_definition=self.simulation_ensemble,
                    ensembles=ensembles,
                    simulation_times=simulation_times,
                    dt=self.simulation_default["system"]["dt"],
                    temperature=temperature,
                    pressure=pressure,
                    off_set=off_set,
                    **kwargs,
                )

                # Create job file
                job_files.append(
                    generate_job_file(
                        destination_folder=copy_folder,
                        job_template=self.system_setup["paths"]["job_template"],
                        software=self.system_setup["software"],
                        input_files=input_files,
                        ensembles=ensembles,
                        job_name=f'{self.system_setup["name"]}_{temperature:.{JOB_PRECISION}f}_{pressure:.{JOB_PRECISION}f}',
                        job_out=f"job_{temperature:.{JOB_PRECISION}f}_{pressure:.{JOB_PRECISION}f}.sh",
                        off_set=off_set,
                        **kwargs,
                    )
                )

            self.job_files.append(job_files)

    def submit_simulation(self):
        """
        Function that submits predefined jobs to the cluster.

        Parameters:
            None

        Returns:
            None
        """
        for temperature, pressure, job_files in zip(
            self.system_setup["temperature"],
            self.system_setup["pressure"],
            self.job_files,
        ):
            print(
                f"\nSubmitting simulations at Temperature = {temperature:.{FOLDER_PRECISION}f} K, Pressure = {pressure:.{FOLDER_PRECISION}f} bar\n"
            )

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run([self.submission_command, job_file])
                print("\n")

    def analysis_extract_properties(
        self,
        analysis_folder: str,
        ensemble: str,
        extracted_properties: List[str],
        fraction: float = 0.0,
        **kwargs,
    ):
        """
        Extracts properties from output files for a specific ensemble.

        Parameters:
         - analysis_folder (str): The name of the folder where the analysis will be performed.
         - ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
         - extracted_properties (List[str]): List of properties to extract for LAMMPS.
         - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.

        Keyword arguments:
         - output_suffix (str): File suffix to analyse for LAMMPS.
         - header (int): The number of header lines from which to extract the keys for the reported values for LAMMPS.
         - header_delimiter (str): The delimiter used in the header line for LAMMPS.
         - command (str): GROMACS command to use for extraction for GROMACS.
         - args (List[str]): Additional arguments for the GROMACS command for GROMACS.
         - ensemble_name (str): Name of the ensemble file for GROMACS.
         - output_name (str): Name of the output file for GROMACS.
         - on_cluster (bool): Flag indicating if extraction should be done on a cluster for GROMACS.
         - extract (bool): Flag indicating if extraction should be performed for GROMACS.
         - submission_command (str): Command for submitting extraction to a cluster for GROMACS.
         - extract_template (str): Path to template for extraction for GROMACS.

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
        copy_pattern = re.compile(r"copy_(\d+)")

        if self.system_setup["software"] == "gromacs":
            output_suffix = "edr"
        elif self.system_setup["software"] == "lammps":
            output_suffix = kwargs["output_suffix"]

        # Search output files and sort them after temperature / pressure and then copy
        for i, (temperature, pressure) in enumerate(
            zip(self.system_setup["temperature"], self.system_setup["pressure"])
        ):
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.{FOLDER_PRECISION}f}_pres_{pressure:.{FOLDER_PRECISION}f}"

            # Search for available copies
            files = glob.glob(
                f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.{output_suffix}"
            )
            files.sort(key=lambda x: int(copy_pattern.search(x).group(1)))

            if len(files) == 0:
                raise KeyError(
                    f"No files found machting the ensemble: {ensemble} in folder\n:   {state_folder}"
                )

            print(
                f"Temperature: {temperature}, Pressure: {pressure}\n   "
                + "\n   ".join(files)
                + "\n"
            )

            if self.system_setup["software"] == "gromacs":
                extracted_df_list = extract_from_gromacs(
                    files=files,
                    extracted_properties=extracted_properties,
                    fraction=fraction,
                    **kwargs,
                )

            elif self.system_setup["software"] == "lammps":
                extracted_df_list = extract_from_lammps(
                    files=files,
                    extracted_properties=extracted_properties,
                    fraction=fraction,
                    **kwargs,
                )

            # Get the mean and std of each property over time
            mean_std_list = []
            for df in extracted_df_list:
                df_new = (
                    df.agg(["mean", "std"])
                    .T.reset_index()
                    .rename(columns={"index": "property"})
                )
                df_new["unit"] = df_new["property"].str.extract(r"\((.*?)\)")
                df_new["property"] = [
                    p.split("(")[0].strip() for p in df_new["property"]
                ]
                mean_std_list.append(df_new)

            # Concat the copies and group by properties
            grouped_total_df = pd.concat(mean_std_list, axis=0).groupby(
                "property", sort=False
            )

            # Get the mean over the copies. To get the standard deviation, propagate the std over the copies.
            mean_over_copies = grouped_total_df["mean"].mean()
            std_over_copies = grouped_total_df["std"].apply(
                lambda p: np.sqrt(sum(p**2)) / len(p)
            )

            # Final df has the mean, std and the unit
            final_df = pd.DataFrame([mean_over_copies, std_over_copies]).T.reset_index()
            final_df["unit"] = df_new["unit"]

            print("\nAveraged values over all copies:\n\n", final_df, "\n")

            # Save as json
            json_data = {
                f"copy_{i}": {
                    d["property"]: {
                        key: value for key, value in d.items() if not key == "property"
                    }
                    for d in df.to_dict(orient="records")
                }
                for i, df in enumerate(mean_std_list)
            }
            json_data["average"] = {
                d["property"]: {
                    key: value for key, value in d.items() if not key == "property"
                }
                for d in final_df.to_dict(orient="records")
            }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"

            work_json(
                json_path,
                {
                    "temperature": temperature,
                    "pressure": pressure,
                    ensemble: {
                        "data": json_data,
                        "paths": files,
                        "fraction_discarded": fraction,
                    },
                },
                "append",
            )

            # Add the extracted values for the analysis_folder and ensemble to the class
            merge_nested_dicts(
                self.analysis_dictionary,
                {(temperature, pressure): {analysis_folder: {ensemble: final_df}}},
            )
