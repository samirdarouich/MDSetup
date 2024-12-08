import re
from typing import List

import numpy as np
import pandas as pd
from alchemlyb.estimators import BAR, MBAR, TI
from scipy.constants import R

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
