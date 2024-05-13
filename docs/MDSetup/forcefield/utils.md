Module MDSetup.forcefield.utils
===============================

Functions
---------

    
`check_pair_style(style, ffiatom: Dict[str, str | float], ffjatom: Dict[str, str | float])`
:   

    
`get_mixed_parameters(ffiatom: Dict[str, str | float], ffjatom: Dict[str, str | float], mixing_rule: str = 'arithmetic', precision: int = 4)`
:   Calculate the mixed parameters for a pair of interacting particles.
    
    Parameters:
        ffiatom (Dict[str,float|str]): Force field information of particle i.
        ffjatom (Dict[str,float|str]): Force field information of particle j.
        mixing_rule (str, optional): The mixing rule to use. Valid options are "arithmetic", "geometric", and "sixthpower". Defaults to "arithmetic".
        precision (int, optional): The number of decimal places to round the results to. Defaults to 4.
    
    Returns:
        tuple: A tuple containing the mixed sigma, epsilon, n and m parameters.
    
    Raises:
        KeyError: If the specified mixing rule is not implemented.

    
`get_pair_style(local_attributes: Dict[str, Any], vdw_pair_styles: List[str], coul_pair_styles: List[str], pair_style_kwargs: Dict[str, str])`
:   This function takes in several parameters and returns a string representing the combined pair style for a molecular simulation.
    
    Parameters:
    - local_attributes (Dict[str,Any]): A dictionary containing local attributes for the pair style.
    - vdw_pair_styles (List[str]): A list of strings representing the Van der Waals pair styles to be used.
    - coul_pair_styles (List[str]): A list of strings representing the Coulombic pair styles to be used.
    - pair_style_kwargs (Dict[str,str]): A dictionary mapping pair styles to their corresponding arguments.
    
    Returns:
    - A string representing the combined pair style for the simulation.
    
    The function iterates over the unique Van der Waals pair styles and Coulombic pair styles provided. For each pair style, it constructs a substring 
    by concatenating the pair style name with the corresponding arguments from the local_attributes dictionary. These substrings are then appended to the combined_pair_style list.
    
    If multiple pair styles are used, the function inserts the "hybrid/overlay" style at the beginning of the combined_pair_style list.
    
    Finally, the function returns the combined pair style as a string, with each pair style separated by two spaces.
    
    Note: The function assumes that the local_attributes dictionary contains all the necessary arguments for each pair style specified 
          in vdw_pair_styles and coul_pair_styles. If an argument is missing, a KeyError will be raised.