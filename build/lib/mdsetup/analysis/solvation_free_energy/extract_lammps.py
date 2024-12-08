import re
from typing import List

import pandas as pd
from alchemlyb.preprocessing import decorrelate_dhdl, decorrelate_u_nk
from scipy.constants import R

from mdsetup.analysis.general import read_lammps_output


def extract_current_state(file_path: str):
    """
    Extracts the current lambda state from the LAMMPS fep output file.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        tuple: A tuple containing the list of lambda types and their value.

    """
    with open(file_path) as file:
        title = file.readline()

    lambdas = list([p.split("=")[0].strip() for p in title.split(":")[1].split(",")])
    statevec = tuple(
        [float(p.split("=")[1].strip()) for p in title.split(":")[1].split(",")]
    )

    return lambdas, statevec


def extract_combined_states(files: List[str], precision: int = 5):
    """
    Function that takes a list of paths to LAMMPS fep output files, sort them after copy and lambda, and then extract from the first copy the combined state vector.

    Parameters:
     - files (List[str]): List of paths to LAMMPS fep output files
     - precision (int,optional): Number of digits for lambda states.

    Returns:
     - combined_lambdas (List[float]): List with lambda states
    """

    copy_pattern = re.compile(r"copy_(\d+)")
    lambda_pattern = re.compile(r"lambda_(\d+)")

    files.sort(
        key=lambda x: (
            int(copy_pattern.search(x).group(1)),
            int(lambda_pattern.search(x).group(1)),
        )
    )

    unique_copy = [file for file in files if "copy_0" in file]
    combined_states = [
        round(sum(extract_current_state(file)[1]), precision) for file in unique_copy
    ]

    return combined_states


def extract_u_nk(
    file_path: str, T: float, time_fraction: float = 0.0, decorrelate: bool = False
):
    """
    Extracts the reduced potential energy values for different lambda states from a LAMMPS output file.

    Parameters:
     - file_path (str): The path to the LAMMPS output file.
     - T (float): The temperature in Kelvin.
     - time_fraction (float, optional): The time_fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
     - decorrelate (bool, optional): Decorrelate dUdl values using alchempylb. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the reduced potential energy values (u_nk) for different lambda states at different times, compared to simulated lambda state.

    Note:
        - The function assumes that the LAMMPS output file has a specific format.
        - The function uses the 'extract_current_state' function to get the current lambda state from the LAMMPS output file.
        - The function uses the 'read_lammps_output' function to read the LAMMPS output file and extract the data.

    """
    u_col_match = "\Delta U"
    pv_col_match = "NAN"

    # Read in current state
    lambdas, statevec = extract_current_state(file_path)

    # Read in free energy output from LAMMPS
    df = read_lammps_output(file_path, time_fraction=time_fraction)

    # Get the time
    times = df[df.columns[0]]

    # Want to grab only Delta U columns
    DUcols = [col for col in df.columns if (u_col_match in col)]
    dU = df[DUcols]

    # Checl if pV is given; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    u_k = dict()
    cols = list()

    # Convert from kcal/mol to dimensionless
    beta = 4184 / (R * T)

    for col in dU:
        # Extract lambda of difference state
        u_col = tuple(
            float(p) for p in col.split("to")[1].split("[")[1].split("]")[0].split()
        )

        # calculate reduced potential u_k = Delta U + pV

        u_k[u_col] = beta * dU[col].values
        if pv_cols:
            u_k[u_col] += beta * pv.values

        cols.append(u_col)

    u_k = pd.DataFrame(
        u_k, columns=cols, index=pd.Index(times.values, name="time", dtype="Float64")
    )

    # Create columns for each lambda, indicating state each row sampled from
    for i, l in enumerate(lambdas):
        try:
            u_k[l] = statevec[i]
        except TypeError:
            u_k[l] = statevec

    # set up new multi-index
    newind = ["time"] + lambdas
    u_k = u_k.reset_index().set_index(newind)

    u_k.name = "u_nk"

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
    Extracts the derivative of the reduced potential values du/dl from a LAMMPS output file.

    Parameters:
     - file_path (str): The path to the LAMMPS output file.
     - T (float): The temperature in Kelvin.
     - time_fraction (float, optional): The time_fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
     - decorrelate (bool, optional): Decorrelate dUdl values using alchempylb. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame containing the derivative of the reduced potential values du/dl for the lambda state at different times.

    Note:
        - The function assumes that the LAMMPS output file has a specific format with columns for DU/dl and pV.
        - The function uses the extract_current_state and read_lammps_output functions to extract the current lambda state and read the LAMMPS output file, respectively.
        - The function converts the dU/dl and pV values to reduced potential values (u_nk) using the provided temperature (T) and the Boltzmann constant (R).
        - The function creates columns for each lambda state, indicating the state each row was sampled from.
        - The function sets up a new multi-index for the DataFrame with the time and lambda states as indices.
        - The function returns the DataFrame containing the derivative of the reduced potential values du/dl with the name "dH/dl".

    """

    dudl_col_match = "dU/dl"
    pv_col_match = "NAN"

    # Read in current state
    lambdas, statevec = extract_current_state(file_path)

    # Read in free energy output from LAMMPS
    df = read_lammps_output(file_path, time_fraction=time_fraction)

    # Get the time
    times = df[df.columns[0]]

    # Want to grab only Delta U columns
    dudlcols = [col for col in df.columns if (dudl_col_match in col)]
    dUdl = df[dudlcols]

    # Checl if pV is given; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    u_k = dict()
    cols = list()

    # Convert from kcal/mol to dimensionless
    beta = 4184 / (R * T)

    for col in dUdl:
        # Extract kind of derivative
        u_col = col.split("_")[1].split("{")[1].split("}")[0]

        # calculate reduced potential u_k = Delta U + pV

        u_k[u_col] = beta * dUdl[col].values
        if pv_cols:
            u_k[u_col] += beta * pv.values

        cols.append(u_col)

    u_k = pd.DataFrame(
        u_k, columns=cols, index=pd.Index(times.values, name="time", dtype="Float64")
    )

    # Create columns for each lambda, indicating state each row sampled from
    for i, l in enumerate(lambdas):
        try:
            u_k[l] = statevec[i]
        except TypeError:
            u_k[l] = statevec

    # set up new multi-index
    newind = ["time"] + lambdas
    u_k = u_k.reset_index().set_index(newind)

    u_k.name = "dH/dl"

    # Decorrelate the samples
    if decorrelate:
        try:
            u_k = decorrelate_dhdl(u_k)
        except Exception as e:
            print(f"Can't decorrelate. Reason: {e}")

    return u_k
