import logging
import numpy as np
import pandas as pd

from typing import List
from scipy.constants import R
from alchemlyb.estimators import BAR, MBAR, TI
from .general_analysis import read_lammps_output, plot_data
from alchemlyb.preprocessing import decorrelate_u_nk, decorrelate_dhdl

# Prevent alchemlyb correlation info to be printed to screen
logging.getLogger('alchemlyb').setLevel('WARNING')

## To do:
# Check pV contribution
# Add that lambdas are also written in the json file
# Add writing json from free energy analysis
def extract_current_state( file_path: str ):
    """
    Extracts the current lambda state from the LAMMPS fep output file.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        tuple: A tuple containing the list of lambda types and their value.

    """
    with open( file_path ) as file:
        title = file.readline()
        
    lambdas = list( map(lambda p: p.split("=")[0].strip(), title.split(":")[1].split(",") ) )
    statevec = tuple( map(lambda p: float(p.split("=")[1].strip()), title.split(":")[1].split(",") ) )
    
    return lambdas, statevec

def extract_u_nk( file_path: str, T: float, fraction: float=0.0, decorrelate: bool=False ):
    """
    Extracts the reduced potential energy values for different lambda states from a LAMMPS output file.

    Parameters:
     - file_path (str): The path to the LAMMPS output file.
     - T (float): The temperature in Kelvin.
     - fraction (float, optional): The fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
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
    lambdas, statevec = extract_current_state( file_path )

    # Read in free energy output from LAMMPS
    df = read_lammps_output( file_path, fraction = fraction )

    # Get the time
    times = df[df.columns[0]]

    # Want to grab only Delta U columns
    DUcols = [ col for col in df.columns if (u_col_match in col) ]
    dU = df[DUcols]

    # Checl if pV is given; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    u_k = dict()
    cols = list()

    # Convert from kcal/mol to dimensionless
    beta = 4184 / ( R * T )

    for col in dU:
        # Extract lambda of difference state
        u_col = tuple( float(p) for p in col.split("to")[1].split("[")[1].split("]")[0].split() )

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
        u_k = decorrelate_u_nk( u_k )
    
    return u_k

def extract_dUdl( file_path: str, T: float, fraction: float=0.0, decorrelate: bool=False ):
    """
    Extracts the derivative of the reduced potential values du/dl from a LAMMPS output file.

    Parameters:
     - file_path (str): The path to the LAMMPS output file.
     - T (float): The temperature in Kelvin.
     - fraction (float, optional): The fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
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
    lambdas, statevec = extract_current_state( file_path )

    # Read in free energy output from LAMMPS
    df = read_lammps_output( file_path, fraction = fraction )

    # Get the time
    times = df[df.columns[0]]

    # Want to grab only Delta U columns
    dudlcols = [ col for col in df.columns if (dudl_col_match in col) ]
    dUdl = df[dudlcols]

    # Checl if pV is given; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    u_k = dict()
    cols = list()

    # Convert from kcal/mol to dimensionless
    beta = 4184 / ( R * T )

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
        u_k = decorrelate_dhdl( u_k )

    return u_k


class TI_spline():
    def __init__(self) -> None:
        pass

def get_free_energy_difference( fep_files: List[str], T: float, method: str="MBAR", fraction: float=0.0, 
                                decorrelate: bool=True ):
    """
    Calculate the free energy difference using different methods.

    Parameters:
     - fep_files (List[str]): A list of file paths to the FEP output files.
     - T (float): The temperature in Kelvin.
     - method (str, optional): The method to use for estimating the free energy difference. Defaults to "MBAR".
     - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
     - decorrelate (bool, optional): Whether to decorrelate the data before estimating the free energy difference. Defaults to True.

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

    if method in [ "MBAR","BAR" ]:
        # Get combined df for all lambda states
        combined_df = pd.concat( [ extract_u_nk( file, T = T, fraction = fraction, decorrelate = decorrelate ) for file in fep_files ] )
        
        # Get free energy estimator
        FE = MBAR() if method == "MBAR" else BAR()

    elif method in [ "TI", "TI_spline" ]:
        # Get combined df for all lambda states
        combined_df = pd.concat( [ extract_dUdl( file, T = T, fraction = fraction, decorrelate = decorrelate ) for file in fep_files ] )

        # Get free energy estimator
        FE = TI() if method == "TI" else TI_spline()

    else:
        raise KeyError(f"Specified free energy method '{method}' is not implemented. Available are: 'MBAR', 'BAR', 'TI' or 'TI_spline' ")
    
    # Get free energy difference
    FE.fit( combined_df )
    
    # Extract mean and std
    mean, std = FE.delta_f_.iloc[0,-1], FE.d_delta_f_.iloc[0,-1]

    # BAR only provides the standard deviation from adjacent intermediates. Hence, to get the global std propagate the error
    if method == "BAR":
        std = np.sqrt( ( np.array( [ FE.d_delta_f_.iloc[i, i+1] for i in range(FE.d_delta_f_.shape[0]-1) ] )**2 ).sum() )

    # Convert from dimensionless to kJ/mol
    df = pd.DataFrame( { "property": "solvation_free_energy", "mean": mean * R * T, "std": std * R * T, "unit": "kJ/mol" }, index = [0] )

    return df

def visualize_dudl( fep_files: List[str], T: float, 
                    fraction: float=0.0, decorrelate: bool=True,
                    save_path: str=""  
                  ):
    

    # Get combined df for all lambda states
    combined_df = pd.concat( [ extract_dUdl( file, T = T, fraction = fraction, decorrelate = decorrelate ) for file in fep_files ] )

    # Extract vdW and Coulomb portion
    vdw_dudl = combined_df.groupby("vdw-lambda")["vdw"].agg(["mean","std"])
    coul_dudl = combined_df.groupby("coul-lambda")["coul"].agg(["mean","std"])
    
    # Plot vdW part
    datas = [ [ vdw_dudl.index.values, vdw_dudl["mean"].values, None, vdw_dudl["std"].values ] ]
    set_kwargs = { "xlabel": "$\lambda_\mathrm{vdW}$",
                   "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{vdW}}} \ / \ (k_\mathrm{B}T)$",
                   "xlim": (0,1)
                 }
    plot_data( datas, save_path = f"{save_path}/dudl_vdw.png", set_kwargs = set_kwargs ) 

    # Plot Coulomb part
    datas = [ [ coul_dudl.index.values, coul_dudl["mean"].values, None, coul_dudl["std"].values ] ]
    set_kwargs = { "xlabel": "$\lambda_\mathrm{coul}$",
                   "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{coul}}} \ / \ (k_\mathrm{B}T)$",
                   "xlim": (0,1)
                 }
    plot_data( datas, save_path = f"{save_path}/dudl_coul.png", set_kwargs = set_kwargs ) 