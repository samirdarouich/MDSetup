import numpy as np

from typing import Dict, List, Any


def check_pair_style( style, ffiatom: Dict[str,str|float], ffjatom: Dict[str,str|float] ):
    if ffiatom[style] != ffjatom[style]:
        raise KeyError(f"Atom '{ffiatom['name']}' and atom '{ffjatom['name']}' has different {style}:\n  {ffiatom[style]} vs {ffjatom[style]}")

def get_mixed_parameters( ffiatom: Dict[str,float|str], ffjatom: Dict[str,float|str],
                          mixing_rule: str="arithmetic", precision: int=4 ):
    """
    Calculate the mixed parameters for a pair of interacting particles.

    Parameters:
        ffiatom (Dict[str,float|str]): Force field information of particle i.
        ffjatom (Dict[str,float|str]): Force field information of particle j.
        mixing_rule (str, optional): The mixing rule to use. Valid options are "arithmetic", "geometric", and "sixthpower". Defaults to "arithmetic".
        precision (int, optional): The number of decimal places to round the results to. Defaults to 4.

    Returns:
        tuple: A tuple containing the mixed sigma, epsilon, n and m parameters.

    Raises:
        KeyError: If the specified mixing rule is not implemented.

    """
    
    # Check if both have the same pair style
    check_pair_style( "vdw_style", ffiatom, ffjatom )

    sigma_i, epsilon_i = ffiatom["sigma"], ffiatom["epsilon"]
    sigma_j, epsilon_j = ffjatom["sigma"], ffjatom["epsilon"]

    if mixing_rule == "arithmetic": 
        sigma_ij   = ( sigma_i + sigma_j ) / 2
        epsilon_ij = np.sqrt( epsilon_i * epsilon_j )

    elif mixing_rule ==  "geometric":
        sigma_ij   = np.sqrt( sigma_i * sigma_j )
        epsilon_ij = np.sqrt( epsilon_i * epsilon_j )

    elif mixing_rule ==  "sixthpower": 
        sigma_ij   = ( 0.5 * ( sigma_i**6 + sigma_j**6 ) )**( 1 / 6 ) 
        epsilon_ij = 2 * np.sqrt( epsilon_i * epsilon_j ) * sigma_i**3 * sigma_j**3 / ( sigma_i**6 + sigma_j**6 )
    
    else:
        raise KeyError(f"Specified mixing rule is not implemented: '{mixing_rule}'. Valid options are: 'arithmetic', 'geometric', and 'sixthpower'")
    
    if "n" in ffiatom.keys() and "n" in ffjatom.keys():
        n_ij  = ( ffiatom["n"] + ffjatom["n"] ) / 2
        m_ij  = ( ffiatom["m"] + ffjatom["m"] ) / 2
    else:
        n_ij  = 12
        m_ij  = 6
    
    return np.round(sigma_ij, precision), np.round(epsilon_ij, precision), n_ij, m_ij

def get_pair_style( local_attributes: Dict[str,Any], vdw_pair_styles: List[str], 
                    coul_pair_styles: List[str], pair_style_kwargs: Dict[str,str] ):
    """
    This function takes in several parameters and returns a string representing the combined pair style for a molecular simulation.

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
    """
    combined_pair_style = []

    for vdw_pair_style in set(vdw_pair_styles):
        sub_string = f"{vdw_pair_style} " + ' '.join( [ str(local_attributes[arg]) for arg in pair_style_kwargs[vdw_pair_style] ] )
        combined_pair_style.append( sub_string )

    for coul_pair_style in set(coul_pair_styles):
        sub_string = f"{coul_pair_style} " + ' '.join( [ str(local_attributes[arg]) for arg in pair_style_kwargs[coul_pair_style] ] )
        combined_pair_style.append( sub_string )

    # Add hybrid/overlay style in case several styles are used
    if len(combined_pair_style) > 1:
        combined_pair_style.insert( 0, "hybrid/overlay" )

    return "  ".join( combined_pair_style )



    
