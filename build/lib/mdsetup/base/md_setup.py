import glob
import os
import re
import shutil
import subprocess
from itertools import groupby
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mdsetup.analysis.general import extract_from_gromacs, extract_from_lammps
from mdsetup.analysis.mechanial_properties import (
    DEFORMATION_DIRECTIONS,
    METHOD,
    PROPERTIES,
    get_stiffness_tensor,
    plot_deformation,
    show_stiffness_tensor,
)
from mdsetup.analysis.solvation_free_energy import (
    EXTRACT,
    check_missing_lambdas,
    get_free_energy_difference,
)
from mdsetup.base.base import BaseSetup
from mdsetup.forcefield import forcefield
from mdsetup.tools.general import (
    DEFAULTS,
    SUFFIX,
    KwargsError,
    merge_nested_dicts,
    work_json,
)
from mdsetup.tools.systemsetup import (
    change_topology,
    generate_initial_configuration,
    generate_input_files,
    generate_job_file,
    get_system_volume,
)

# Todo: create pydantic config files for input


class MDSetup(BaseSetup):
    """This class sets up structured and FAIR molecular dynamic simulations. It also has
    the capability to build a system based on a list of molecules."""

    def write_topology(self, verbose: bool = False) -> None:
        """This functions writes a topology file using the moleculegraph representation of each molecule in the system as well as the force field files.

        Args:
          verbose (bool, optional): Flag to print detailed information. Defaults to False.

        Returns:
          None

        Raises:
          KeyError: _description_

        """
        print(
            "\nUtilize moleculegraph to generate molecule and topology files "
            "of every molecule in the system!\n"
        )

        topology_folder = f"{self.project_folder}/topology"

        os.makedirs(topology_folder, exist_ok=True)

        if not any(self.system_setup["paths"]["force_field_files"]):
            raise KeyError(
                "No force field paths provided in the system setup yaml file!"
            )

        # Prepare keyword arguments that are parsed to the functions
        # Add all paths to templates and all nonbonded settings to template
        kwargs = {**self.system_setup["paths"], **self.simulation_default["nonbonded"]}

        if self.system_setup["software"] == "gromacs":
            kwargs["nrexcl"] = [mol["nrexcl"] for mol in self.system_molecules]

        # Call the force field class
        ff_molecules = forcefield(
            smiles=[mol["smiles"] for mol in self.system_molecules],
            force_field_paths=self.system_setup["paths"]["force_field_files"],
            verbose=verbose,
        )

        # Write molecule files (including gro files in case of GROMACS)
        ff_molecules.write_molecule_files(
            molecule_path=topology_folder,
            residues=self.residues,
            **kwargs,
        )

        # Add resiude and its numbers to kwargs
        kwargs["residue_dict"] = {
            mol["name"]: mol["number"] for mol in self.system_setup["molecules"]
        }

        # Write topology file
        ff_molecules.write_topology_file(
            topology_path=topology_folder,
            system_name=self.system_setup["name"],
            **kwargs,
        )

        print(
            "\nDone! Added generated paths to class:\n"
            f"\nTopology file:\n {ff_molecules.topology_file}\n"
            f"\nMolecule files:\n {ff_molecules.molecule_files}\n"
        )

        # Add topology and molecules files to class dictionary
        self.system_setup["paths"]["topology_file"] = ff_molecules.topology_file
        self.system_setup["paths"]["coordinates_files"] = ff_molecules.molecule_files

    def prepare_simulation(
        self,
        folder_name: str,
        ensembles: List[str],
        simulation_times: List[float],
        initial_systems: List[str] = None,
        copies: int = 0,
        on_cluster: bool = False,
        off_set: int = 0,
        input_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Prepares the simulation by generating job files for each temperature and pressure combination specified in the simulation setup.

        Args:
          folder_name (str):
            Name of the subfolder where to perform the simulations. Path structure is
            as follows: system.folder/system.name/folder_name
          ensembles (List[str]):
            A list of ensembles to generate input files for. Definitions of each
            ensemble is provided in self.simulation_ensemble.
          simulation_times (List[float]):
            A list of simulation times (ns) for each ensemble.
          initial_systems (List[str]):
            A list of initial system .gro / .data files to be used for each defined
            state (temperature/pressure/density).
          copies (int, optional):
            Number of copies for the specified system. Defaults to 0.
          on_cluster (bool, optional):
            If the build should be submited to the cluster. Defaults to "False".
          off_set (int, optional):
            First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
          input_kwargs (Dict[str, Any], optional):
            Further kwargs that are parsed to the input template. Defaults to "None".
          **kwargs:
            Arbitrary keyword arguments.

        Returns:
          None

        """
        if initial_systems is None:
            initial_systems = []
        if input_kwargs is None:
            input_kwargs = {}
        self.job_files = []

        # Define simulation folder
        sim_folder = f"{self.project_folder}/{folder_name}"

        # Prepare keyword arguments that are parsed to the functions
        kwargs = {
            **self.simulation_default,
            **self.simulation_sampling,
            **input_kwargs,
            **DEFAULTS[self.system_setup["software"]],
            **kwargs,
        }

        # Copy provided force field file to simulation folder and add it to input kwargs
        os.makedirs(sim_folder, exist_ok=True)
        suffix = SUFFIX["topology"][self.system_setup["software"]]
        kwargs["initial_topology"] = shutil.copy(
            self.system_setup["paths"]["topology_file"],
            f"{sim_folder}/init_topology.{suffix}",
        )

        if self.system_setup["software"] == "gromacs":
            change_topology(
                initial_topology=kwargs["initial_topology"],
                system_molecules=self.system_molecules,
                system_name=self.system_setup["name"],
            )

        for i, (temperature, pressure, density) in enumerate(
            zip(
                self.system_setup["temperature"],
                self.system_setup["pressure"],
                self.system_setup["density"],
            )
        ):
            job_files = []

            # Compute mole fraction of component 1
            mole_fraction = self.molecule_numbers[0] / sum(self.molecule_numbers)

            # Define folder with defined state attributes
            state_cond = self.define_state_cond(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            state_folder = f"{sim_folder}/{state_cond}"

            state_text = self.define_state_text(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            # Build system with MD software if none is provided
            build_folder = f"{state_folder}/build"

            if not initial_systems:
                print(
                    "\nBuilding system based on provided molecule numbers "
                    f"and coordinate files for {state_text}!\n"
                )

                # Get intial box lenghts using density estimate
                box = get_system_volume(
                    molar_masses=self.molar_masses,
                    molecule_numbers=self.molecule_numbers,
                    density=density,
                    unit_conversion=self.distance_conversion,
                    **self.system_setup["box"],
                )

                # In case of LAMMPS provide template for build input.
                if self.system_setup["software"] == "lammps":
                    kwargs["build_input_template"] = self.system_setup["paths"][
                        "build_input_template"
                    ]
                    kwargs["force_field_file"] = self.system_setup["paths"][
                        "topology_file"
                    ]

                # Coordinates from molecule that are not present in the system are
                # sorted out within the function.
                # Hence, parse the non filtered list of molecules and coordinates here.
                kwargs["initial_coord"] = generate_initial_configuration(
                    destination_folder=build_folder,
                    build_template=self.system_setup["paths"]["build_template"],
                    software=self.system_setup["software"],
                    coordinate_paths=self.system_setup["paths"]["coordinates_files"],
                    molecules_list=self.system_setup["molecules"],
                    box=box,
                    submission_command=self.submission_command,
                    on_cluster=on_cluster,
                    **kwargs,
                )

                kwargs["restart_flag"] = False

            else:
                os.makedirs(build_folder, exist_ok=True)
                suffix = SUFFIX["coordinate"][self.system_setup["software"]]
                kwargs["initial_coord"] = shutil.copy(
                    initial_systems[i], f"{build_folder}/init_conf.{suffix}"
                )

                print(
                    f"\nIntial system provided for {state_text} at: {initial_systems[i]}\n"
                )

                kwargs["restart_flag"] = ".restart" in kwargs[
                    "initial_coord"
                ] or os.path.exists(initial_systems[i].rsplit(".", 1)[0] + ".cpt")

                if kwargs["restart_flag"]:
                    print("Restart file is provided. Continue simulation from there!\n")

                    if self.system_setup["software"] == "gromacs":
                        # In case of GROMACS, add cpt as input
                        kwargs["initial_cpt"] = shutil.copy(
                            initial_systems[i].rsplit(".", 1)[0] + ".cpt",
                            f"{build_folder}/init_conf.cpt",
                        )
                if kwargs.get("time_to_extend", 0.0) > 0:
                    print(
                        "Extending existing simulation of the first ensemble provided. "
                        "Provided simulation time and gro file will be ignored for the "
                        "first ensemble. Simulation will be extended for "
                        f"{kwargs['time_to_extend']} ns.\n"
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
                        job_name=f'{self.system_setup["name"]}_{state_cond}',
                        job_out=f"job_{state_cond}.sh",
                        off_set=off_set,
                        **kwargs,
                    )
                )

            self.job_files.append(job_files)

    def submit_simulation(self) -> None:
        """Function that submits predefined jobs to the cluster.

        Args:
          None

        Returns:
          None

        """
        for temperature, pressure, density, job_files in zip(
            self.system_setup["temperature"],
            self.system_setup["pressure"],
            self.system_setup["density"],
            self.job_files,
        ):
            # Compute mole fraction of component 1
            mole_fraction = self.molecule_numbers[0] / sum(self.molecule_numbers)

            state_text = self.define_state_text(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            print(f"\nSubmitting simulations at {state_text}.\n")

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run([self.submission_command, job_file])
                print("\n")

    def analysis_extract_properties(
        self,
        analysis_folder: str,
        ensemble: str,
        extracted_properties: List[str],
        time_fraction: float = 0.0,
        **kwargs,
    ) -> None:
        """Extracts properties from output files for a specific ensemble.

        The method searches for output files in the specified analysis folder that match
        the given ensemble. For each group of files with the same state conditions, the
        properties are extracted using the specified suffix and properties list. The
        extracted properties are then averaged over all copies and the mean and standard
        deviation are calculated. The averaged values and the extracted data for each
        copy are saved as a JSON file in the destination folder. The extracted values
        are also added to the class's analysis dictionary.

        Args:
          analysis_folder(str):
            The name of the folder where the analysis will be performed.
          ensemble(str):
            The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
          extracted_properties(List[str]):
            List of properties to extract for LAMMPS.
          time_fraction(float):
            The time fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
          **kwargs:
            Arbitrary keyword arguments.

        Keyword Arguments:
          output_suffix (str):
            File suffix to analyse for LAMMPS.
          header (int, optional):
            The number of header lines from which to extract the keys for the reported values for LAMMPS.
          header_delimiter (str, otpional):
            The delimiter used in the header line for LAMMPS.
          command (str):
            GROMACS command to use for extraction for GROMACS.
          args (List[str]):
            Additional arguments for the GROMACS command for GROMACS.
          ensemble_name (str):
            Name of the ensemble file for GROMACS.
          output_name (str):
            Name of the output file for GROMACS.
          on_cluster (bool, optional):
            Flag indicating if extraction should be done on a cluster for GROMACS.
          extract (bool):
            Flag indicating if extraction should be performed for GROMACS.
          extract_template (str):
            Path to template for extraction for GROMACS.

        Returns:
          None

        """
        # Define folder for analysis
        sim_folder = f"{self.project_folder}/{analysis_folder}"

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])

        # sorting patterns
        copy_pattern = re.compile(r"copy_(\d+)")

        if self.system_setup["software"] == "gromacs":
            output_suffix = "edr"
            KwargsError(
                ["command", "args", "output_name", "extract", "extract_template"],
                kwargs.keys(),
            )

        elif self.system_setup["software"] == "lammps":
            KwargsError(["output_suffix"], kwargs.keys())
            output_suffix = kwargs["output_suffix"]

        # Search output files and sort them after the copy
        for temperature, pressure, density in zip(
            self.system_setup["temperature"],
            self.system_setup["pressure"],
            self.system_setup["density"],
        ):
            # Compute mole fraction of component 1
            mole_fraction = self.molecule_numbers[0] / sum(self.molecule_numbers)

            # Define state conditions
            state_cond = self.define_state_cond(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            state_folder = f"{sim_folder}/{state_cond}"

            state_text = self.define_state_text(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            # Search for available copies
            files = glob.glob(
                f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.{output_suffix}"
            )
            files.sort(key=lambda x: int(copy_pattern.search(x)[1]))

            if not files:
                raise KeyError(
                    f"No files found machting the ensemble: {ensemble} in folder\n: "
                    f"{state_folder}"
                )

            print(
                "\nAnalysis for the following conditions and files: "
                f"{state_text}\n   " + "\n   ".join(files) + "\n"
            )

            if self.system_setup["software"] == "gromacs":
                extracted_df_list = extract_from_gromacs(
                    files=files,
                    extracted_properties=extracted_properties,
                    time_fraction=time_fraction,
                    submission_command=self.submission_command,
                    ensemble_name=ensemble_name,
                    **kwargs,
                )

            elif self.system_setup["software"] == "lammps":
                extracted_df_list = extract_from_lammps(
                    files=files,
                    extracted_properties=extracted_properties,
                    time_fraction=time_fraction,
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
                        key: value for key, value in d.items() if key != "property"
                    }
                    for d in df.to_dict(orient="records")
                }
                for i, df in enumerate(mean_std_list)
            }
            json_data["average"] = {
                d["property"]: {
                    key: value for key, value in d.items() if key != "property"
                }
                for d in final_df.to_dict(orient="records")
            }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"

            extracted_data = {
                ensemble: {
                    "data": json_data,
                    "paths": files,
                    "time_fraction_discarded": time_fraction,
                }
            }

            # Add state there
            for folder_attribute in self.system_setup["folder_attributes"]:
                extracted_data[folder_attribute] = {
                    "temperature": temperature,
                    "pressure": pressure,
                    "density": density,
                    "mole_fraction": mole_fraction,
                }[folder_attribute]

            work_json(json_path, extracted_data, "append")

            # Add the extracted values to the class
            merge_nested_dicts(
                self.analysis_dictionary,
                {state_folder: extracted_data},
            )

    def analysis_mechanical_prpoerties(
        self,
        analysis_folder: str,
        ensemble: str,
        deformation_rates: List[float],
        method: str = "VRH",
        time_fraction: float = 0.0,
        visualize_stress_strain: bool = False,
    ):
        # Flag to only evaluate the undeformed system once
        flag_undeformed = False

        # State dictionary for results
        state_dict = {}

        for deformation_direction in DEFORMATION_DIRECTIONS:
            for deformation_rate in deformation_rates:
                if deformation_rate == 0.0:
                    if not flag_undeformed:
                        # Define the analysis folder
                        tmp_analysis_folder = (
                            f"{analysis_folder}/undeformed/{deformation_rate}"
                        )

                        # Extract properties from LAMMPS and analyse them
                        self.analysis_extract_properties(
                            analysis_folder=tmp_analysis_folder,
                            ensemble=ensemble,
                            extracted_properties=PROPERTIES,
                            output_suffix="pressure",
                            time_fraction=time_fraction,
                        )

                        flag_undeformed = True
                    else:
                        # Define the analysis folder
                        tmp_analysis_folder = (
                            f"{analysis_folder}/undeformed/{deformation_rate}"
                        )
                else:
                    # Define the analysis folder
                    tmp_analysis_folder = (
                        f"{analysis_folder}/{deformation_direction}/{deformation_rate}"
                    )

                    # Extract properties from LAMMPS and analyse them
                    self.analysis_extract_properties(
                        analysis_folder=tmp_analysis_folder,
                        ensemble=ensemble,
                        extracted_properties=PROPERTIES,
                        output_suffix="pressure",
                        time_fraction=time_fraction,
                    )

                # Add each copy to the results dict
                for (
                    i,
                    temperature,
                    pressure,
                    density,
                    mole_fraction,
                ) in self.loop_through_states():
                    # Define folder with defined state attributes
                    state_condition = self.define_state_cond(
                        temperature=temperature,
                        pressure=pressure,
                        density=density,
                        mole_fraction=mole_fraction,
                    )

                    json_path = (
                        f"{self.project_folder}/{tmp_analysis_folder}/"
                        f"{state_condition}/results.json"
                    )

                    # Convert from atm in GPa
                    for copy, copy_dict in work_json(json_path)[ensemble][
                        "data"
                    ].items():
                        sim_results = {}
                        for key, prop in copy_dict.items():
                            if key in PROPERTIES:
                                prop["mean"] *= 101325 / 1e9
                                prop["std"] *= 101325 / 1e9
                                prop["unit"] = "GPa"
                                sim_results[key] = prop

                        state_dict.setdefault((temperature, pressure), {}).setdefault(
                            copy, {}
                        ).setdefault(deformation_direction, {}).update(
                            {deformation_rate: sim_results}
                        )

        # Visualize if specified
        if visualize_stress_strain:
            for (temperature, pressure), s_dict in state_dict.items():
                print(f"Temperature: {temperature:.1f} K, pressure: {pressure:.1f} bar")

                plot_deformation(
                    s_dict["average"],
                    outpath=f"{self.project_folder}/deformation",
                    main_only=True,
                )

        # Get the mechanical properties over each copy and from it the mean and standard deviation
        for (temperature, pressure), s_dict in state_dict.items():
            final_results = {}

            state_condition = self.define_state_cond(
                temperature=temperature,
                pressure=pressure,
            )

            for copy, deformation_dict in s_dict.items():
                if copy == "average":
                    continue

                # Get slope of stress-strain curve at low deformation
                stiffness_tensor = get_stiffness_tensor(deformation_dict)

                # Computation of mechanical properties

                # Voigt Reuss Hill
                K, G, E, nu = METHOD[method](stiffness_tensor)

                final_results[copy] = {
                    "C": {"mean": stiffness_tensor.tolist()},
                    "K": {"mean": K},
                    "G": {"mean": G},
                    "E": {"mean": E},
                    "nu": {"mean": nu},
                }

            # Get the average over all copies
            final_results["average"] = {
                key: {
                    "mean": np.mean(
                        [
                            copy_data[key]["mean"]
                            for copy_data in final_results.values()
                        ],
                        axis=0,
                    ).tolist(),
                    "std": np.std(
                        [
                            copy_data[key]["mean"]
                            for copy_data in final_results.values()
                        ],
                        ddof=1,
                        axis=0,
                    ).tolist(),
                }
                for key in ["C", "K", "G", "E", "nu"]
            }

            show_stiffness_tensor(state_condition, final_results)

            work_json(
                file_path=(
                    f"{self.project_folder}/{analysis_folder}/results_mechanical_"
                    f"{state_condition}.json"
                ),
                data=final_results,
                to_do="write",
            )

    def analysis_solvation_free_energy(
        self,
        analysis_folder: str,
        ensemble: str,
        method: str = "MBAR",
        time_fraction: float = 0.0,
        decorrelate: bool = True,
        coupling: bool = True,
    ):
        # Define simulation folder and insert guest molecule'
        sim_folder = f"{self.project_folder}/{analysis_folder}"
        ensemble_name = "_".join(ensemble.split("_")[1:])

        copy_pattern = re.compile(r"copy_(\d+)")
        lambda_pattern = re.compile(r"lambda_(\d+)")

        for i, (temperature, pressure, density) in enumerate(
            zip(
                self.system_setup["temperature"],
                self.system_setup["pressure"],
                self.system_setup["density"],
            )
        ):
            # Compute mole fraction of component 1
            mole_fraction = self.molecule_numbers[0] / (sum(self.molecule_numbers))

            # Define folder with defined state attributes
            state_condition = self.define_state_cond(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            state_text = self.define_state_text(
                temperature=temperature,
                pressure=pressure,
                density=density,
                mole_fraction=mole_fraction,
            )

            # Search for available copies
            files = glob.glob(
                (
                    f"{sim_folder}/lambda_*/{state_condition}/copy_*/{ensemble}/"
                    f"{ensemble_name}.xvg"
                )
            )
            files.sort(
                key=lambda x: (
                    int(copy_pattern.search(x).group(1)),
                    int(lambda_pattern.search(x).group(1)),
                )
            )

            if not files:
                raise KeyError(
                    f"No files found machting the ensemble: {ensemble} in folder\n: "
                    f"{sim_folder}/lambda_*/{state_condition}/copy_*/"
                )

            print(
                "\nAnalysis for the following conditions and files: "
                f"{state_text}\n   " + "\n   ".join(files) + "\n"
            )

            # Group by copies and perform free energy analysis
            mean_std_list = []

            for _, copy_files in groupby(
                files, key=lambda x: int(copy_pattern.search(x).group(1))
            ):
                copy_files = list(copy_files)

                # Check for missing simulations in case of BAR or MBAR
                if method in ["BAR", "MBAR"]:
                    check_missing_lambdas(copy_files)

                # Analysis free energy
                mean_std_list.append(
                    get_free_energy_difference(
                        software=self.system_setup["software"],
                        fep_files=copy_files,
                        T=temperature,
                        method=method,
                        time_fraction=time_fraction,
                        decorrelate=decorrelate,
                        coupling=coupling,
                    )
                )

            if len(mean_std_list) == 0:
                raise KeyError("No data was extracted!")

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
            final_df["unit"] = mean_std_list[0]["unit"]

            # Get the combined lambda state list
            combined_states = EXTRACT["states"][self.system_setup["software"]](files)

            print(
                f"\nFollowing combined lambda states were analysed with the '{method}' "
                f"method:\n   {', '.join([str(la) for la in combined_states])}"
            )
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
            json_path = f"{sim_folder}/results_solvation_{state_condition}.json"

            extracted_data = {
                ensemble: {
                    method: {
                        "data": json_data,
                        "paths": files,
                        "time_fraction_discarded": time_fraction,
                        "decorrelate": decorrelate,
                        "combined_states": combined_states,
                        "coupling": coupling,
                    }
                }
            }

            # Add state there
            for folder_attribute in self.system_setup["folder_attributes"]:
                extracted_data[folder_attribute] = {
                    "temperature": temperature,
                    "pressure": pressure,
                    "density": density,
                    "mole_fraction": mole_fraction,
                }[folder_attribute]

            work_json(json_path, extracted_data, "append")

            # Add the extracted values to the class
            merge_nested_dicts(
                self.analysis_dictionary,
                {f"{sim_folder}/results_{state_condition}": extracted_data},
            )
