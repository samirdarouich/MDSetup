import logging
import re
from typing import List

from alchemlyb.parsing.gmx import _extract_state, _get_headers
from alchemlyb.parsing.gmx import extract_dHdl as extract_dHdl_alchemylb
from alchemlyb.parsing.gmx import extract_u_nk as extract_u_nk_alchemylb
from alchemlyb.preprocessing import decorrelate_dhdl, decorrelate_u_nk

# Prevent alchemlyb correlation info to be printed to screen
logging.getLogger("alchemlyb.preprocessing.subsampling").disabled = True


def extract_u_nk(
    file_path: str, T: float, time_fraction: float = 0.0, decorrelate: bool = False
):
    """
    Extracts and processes reduced potential energy data (`u_nk`) from a specified file.

    This function extracts the reduced potential energies as a function of time and
    filters out data before a specified time fraction. Optionally, it can decorrelate the data.

    Args:
        file_path (str):
          Path to the file containing the energy data.
        T (float):
          Temperature of the system (in Kelvin).
        time_fraction (float, optional):
          Fraction of the total time to discard data from the start. Defaults to 0.0,
          meaning no data is discarded.
        decorrelate (bool, optional):
          If True, the function will decorrelate the data. Defaults to False.

    Returns:
        pd.DataFrame:
          The processed reduced potential energy data, filtered and decorrelated if
          specified.
    """
    # Extraxt data
    u_k = extract_u_nk_alchemylb(file_path, T=T)

    # Discard everything from start to fraction
    idx = (
        u_k.index.get_level_values("time").values
        >= u_k.index.get_level_values("time").values.max() * time_fraction
    )
    u_k = u_k[idx]

    # Decorrelate the samples
    if decorrelate:
        try:
            u_k = decorrelate_u_nk(u_k)
        except Exception as e:
            print(f"Can't decorrelate. Reason: {e}")

    return u_k


def extract_dUdl(
    file_path: str, T: float, time_fraction: float = 0.0, decorrelate: bool = False
):
    """
    Extracts and processes the derivative of the Hamiltonian with respect to lambda (`dU/dλ`) from a specified file.

    This function extracts the `dU/dλ` values as a function of time and filters out data
    before a specified time fraction. Optionally, it can decorrelate the data.

    Args:
        file_path (str):
          Path to the file containing the derivative data.
        T (float):
          Temperature of the system (in Kelvin).
        time_fraction (float, optional):
          Fraction of the total time to discard data from the start. Defaults to 0.0,
          meaning no data is discarded.
        decorrelate (bool, optional):
          If True, the function will decorrelate the data. Defaults to False.

    Returns:
        pd.DataFrame:
          The processed derivative data (`dU/dλ`), filtered and decorrelated if specified.
    """
    # Extraxt data
    u_k = extract_dHdl_alchemylb(file_path, T=T)

    # Discard everything from start to fraction
    idx = (
        u_k.index.get_level_values("time").values
        > u_k.index.get_level_values("time").values.max() * time_fraction
    )
    u_k = u_k[idx]

    # Decorrelate the samples
    if decorrelate:
        try:
            u_k = decorrelate_dhdl(u_k)
        except Exception as e:
            print(f"Can't decorrelate. Reason: {e}")

    return u_k


def extract_current_state(file: str):
    headers = _get_headers(file)
    state, lambdas, statevec = _extract_state(file, headers)
    return state, lambdas, statevec


def extract_combined_states(fep_files: List[str]):
    """
    Function that takes a list of paths to GROMACS fep output files, sort them after copy and lambda, and then extract from the first copy the combined state vector.

    Parameters:
     - fep_files (List[str]): List of paths to GROMACS fep output files

    Returns:
     - combined_lambdas (List[float]): List with lambda states
    """

    copy_pattern = re.compile(r"copy_(\d+)")
    lambda_pattern = re.compile(r"lambda_(\d+)")

    fep_files.sort(
        key=lambda x: (
            int(copy_pattern.search(x).group(1)),
            int(lambda_pattern.search(x).group(1)),
        )
    )

    unique_copy = [file for file in fep_files if "copy_0" in file]
    combined_states = [sum(extract_current_state(file)[2]) for file in unique_copy]

    return combined_states
