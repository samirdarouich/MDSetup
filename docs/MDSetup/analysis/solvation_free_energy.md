Module MDSetup.analysis.solvation_free_energy
=============================================

Functions
---------

    
`extract_combined_states(files: List[str], precision: int = 5)`
:   Function that takes a list of paths to LAMMPS fep output files, sort them after copy and lambda, and then extract from the first copy the combined state vector.
    
    Parameters:
     - files (List[str]): List of paths to LAMMPS fep output files
     - precision (int,optional): Number of digits for lambda states.
    
    Returns:
     - combined_lambdas (List[float]): List with lambda states

    
`extract_current_state(file_path: str)`
:   Extracts the current lambda state from the LAMMPS fep output file.
    
    Parameters:
        file_path (str): The path to the file.
    
    Returns:
        tuple: A tuple containing the list of lambda types and their value.

    
`extract_dUdl(file_path: str, T: float, fraction: float = 0.0, decorrelate: bool = False)`
:   Extracts the derivative of the reduced potential values du/dl from a LAMMPS output file.
    
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

    
`extract_u_nk(file_path: str, T: float, fraction: float = 0.0, decorrelate: bool = False)`
:   Extracts the reduced potential energy values for different lambda states from a LAMMPS output file.
    
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

    
`get_free_energy_difference(fep_files: List[str], T: float, method: str = 'MBAR', fraction: float = 0.0, decorrelate: bool = True, coupling: bool = True)`
:   Calculate the free energy difference using different methods.
    
    Parameters:
     - fep_files (List[str]): A list of file paths to the FEP output files.
     - T (float): The temperature in Kelvin.
     - method (str, optional): The method to use for estimating the free energy difference. Defaults to "MBAR".
     - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
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

    
`visualize_dudl(fep_files: List[str], T: float, fraction: float = 0.0, decorrelate: bool = True, save_path: str = '')`
:   

Classes
-------

`TI_spline()`
: