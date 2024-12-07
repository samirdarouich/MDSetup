import glob
import re
from itertools import groupby
from typing import List

import numpy as np
import pandas as pd
from alchemlyb.estimators import BAR, MBAR, TI
from scipy.constants import R

from ...tools.general import merge_nested_dicts, work_json
from .estimators import TI_spline
from .extract_gromacs import extract_combined_states as extract_combined_states_gromacs
from .extract_gromacs import extract_dUdl as extract_dUdl_gromacs
from .extract_gromacs import extract_u_nk as extract_u_nk_gromacs
from .extract_lammps import extract_combined_states as extract_combined_states_lammps
from .extract_lammps import extract_dUdl as extract_dUdl_lammps
from .extract_lammps import extract_u_nk as extract_u_nk_lammps

EXTRACT = {
    "u_nk": {"lammps": extract_u_nk_lammps, "gromacs": extract_u_nk_gromacs},
    "dudl": {"lammps": extract_dUdl_lammps, "gromacs": extract_dUdl_gromacs},
    "states": {
        "lammps": extract_combined_states_lammps,
        "gromacs": extract_combined_states_gromacs,
    },
}


def check_missing_lambdas(files):
    # Define regex pattern to extract lambda values
    lambda_pattern = re.compile(r"lambda_(\d+)")

    # Extract lambda indices from file names and convert them to integers
    lambda_indices = sorted(int(lambda_pattern.search(f).group(1)) for f in files)

    # Check for missing consecutive lambdas
    missing_lambdas = [
        i
        for i in range(lambda_indices[0], lambda_indices[-1])
        if i not in lambda_indices
    ]

    if missing_lambdas:
        raise ValueError(
            f"Missing simulations for following lambda states: {missing_lambdas}"
        )

    return


def get_free_energy_difference(
    software: str,
    fep_files: List[str],
    T: float,
    method: str = "MBAR",
    time_fraction: float = 0.0,
    decorrelate: bool = True,
    coupling: bool = True,
):
    """
    Calculate the free energy difference using different methods.

    Parameters:
     - software (str): Specify which software is used.
     - fep_files (List[str]): A list of file paths to the FEP output files.
     - T (float): The temperature in Kelvin.
     - method (str, optional): The method to use for estimating the free energy difference. Defaults to "MBAR".
     - time_fraction (float, optional): The time_fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
     - decorrelate (bool, optional): Whether to decorrelate the data before estimating the free energy difference. Defaults to True.
     - coupling (bool, optional): If coupling (True) or decoupling (False) is performed. If decoupling,
                                  multiply free energy results *-1 to get solvation free energy. Defaults to True.
    Returns:
     - df (pd.DataFrame): Pandas dataframe with mean, std and unit of the free energy difference

    Raises:
     - KeyError: If the specified method is not implemented.

    Notes:
     - The function supports the following methods: "MBAR", "BAR", "TI", and "TI_spline".
     - The function uses the 'extract_u_nk' function for "MBAR" and "BAR" methods, and the 'extract_dUdl' function for "TI" and "TI_spline" methods.
     - The function concatenates the data from all FEP output files into a single DataFrame.
     - The function fits the free energy estimator using the combined DataFrame.
     - The function extracts the mean and standard deviation of the free energy difference from the fitted estimator.

    """
    assert len(fep_files) > 0, "Expected free energy files"

    if method in ["MBAR", "BAR"]:
        # Get combined df for all lambda states
        combined_df = pd.concat(
            [
                EXTRACT["u_nk"][software](
                    file, T=T, time_fraction=time_fraction, decorrelate=decorrelate
                )
                for file in fep_files
            ]
        )

        # Get free energy estimator
        FE = MBAR() if method == "MBAR" else BAR()

    elif method in ["TI", "TI_spline"]:
        # Get combined df for all lambda states
        combined_df = pd.concat(
            [
                EXTRACT["dudl"][software](
                    file, T=T, time_fraction=time_fraction, decorrelate=decorrelate
                )
                for file in fep_files
            ]
        )

        # Get free energy estimator
        FE = TI() if method == "TI" else TI_spline()

    else:
        raise KeyError(
            f"Specified free energy method '{method}' is not implemented. Available "
            "are: 'MBAR', 'BAR', 'TI' or 'TI_spline' "
        )

    # Get free energy difference
    FE.fit(combined_df)

    # Extract mean and std
    mean, std = FE.delta_f_.iloc[0, -1], FE.d_delta_f_.iloc[0, -1]

    # In case decoupling is performed, negate the value to get solvation free energy
    if not coupling:
        mean *= -1

    # BAR only provides the standard deviation from adjacent intermediates. Hence,
    # to get the global std propagate the error
    if method == "BAR":
        std = np.sqrt(
            (
                np.array(
                    [
                        FE.d_delta_f_.iloc[i, i + 1]
                        for i in range(FE.d_delta_f_.shape[0] - 1)
                    ]
                )
                ** 2
            ).sum()
        )

    # Convert from dimensionless to kJ/mol
    df = pd.DataFrame(
        {
            "property": "solvation_free_energy",
            "mean": mean * R * T / 1000,
            "std": std * R * T / 1000,
            "unit": "kJ/mol",
        },
        index=[0],
    )

    return df


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
            f"method:\n   {', '.join([str(l) for l in combined_states])}"
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
