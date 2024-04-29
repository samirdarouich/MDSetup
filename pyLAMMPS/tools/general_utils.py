#### General utilities ####

import os
import json
import numpy as np

from scipy.constants import Avogadro
from typing import List, Tuple, Dict, Any

def deep_get(obj: Dict[str,Any]|List[Any]|Any, keys: str, default: Any={} ):
    """
    Function that searches an (nested) python object and extract the item at the end of the key chain.
    Keys are provided as one string and seperated by ".".

    Args:
        obj (Dict[str,Any]|List[Any]|Any): Object from which the (nested) keys are extracted
        keys (str): Keys to extract. Chain of keys should be seperated by ".". Integers to get list items will be converted from string.
        default (dict, optional): Default return if key is not found. Defaults to {}.

    Returns:
        d (Any): Element that is extracted
    """
    d = obj
    for key in keys.split("."):
        if isinstance(d, dict):
            d = d.get(key, default)          
        elif isinstance(d,(list,np.ndarray,tuple)):
            d = d[int(key)]
        elif isinstance(d, object):
            d = getattr(d,key,default)
        else:
            raise KeyError(f"Subtype is not implemented for extraction: '{type(d)}'")
        
        if not isinstance(d,np.ndarray) and d == default:
            print(f"\nKey: '{key}' not found! Return default!\n")

    return d

def flatten_list(lst: List[Any]):
    """
    Function that flattens a list with sublists, of items.
    E.g: 
    test = [1,2,3,[4,5,6]] 
    flatten_list(a)
    >> [1,2,3,4,5,6]
    """
    return [ item for sublist in lst for item in (sublist if isinstance(sublist, list) or isinstance(sublist, np.ndarray) else [sublist]) ]



def map_function_input(all_attributes: dict, argument_map: dict) -> dict:
    """
    Function that maps the elements in the attributes dictionary based on the provided mapping dictionary.

    Args:
        all_attributes (dict): A dictionary containing all the attributes from which the elements will be mapped.
        argument_map (dict): A dictionary that defines the mapping between the elements in all_attributes and the desired keys in the output dictionary.

    Returns:
        dict: A new dictionary with the mapped elements.
    """
    function_input = {}
    for arg, item in argument_map.items():
        if isinstance(item, dict):
            function_input[arg] = {subarg: deep_get(all_attributes, subitem) for subarg, subitem in item.items()}
        else:
            function_input[arg] = deep_get(all_attributes, item)
    return function_input

def merge_nested_dicts(existing_dict: Dict[str, Any], new_dict: Dict[str, Any]):
    """
    Function that merges nested dictionaries

    Args:
        existing_dict (Dict): Existing dictionary that will be merged with the new dictionary
        new_dict (Dict): New dictionary
    """
    for key, value in new_dict.items():
        if key in existing_dict and isinstance(existing_dict[key], dict) and isinstance(value, dict):
            # If both the existing and new values are dictionaries, merge them recursively
            merge_nested_dicts(existing_dict[key], value)
        else:
            # If the key doesn't exist in the existing dictionary or the values are not dictionaries, update the value
            existing_dict[key] = value


def serialize_json(data: Dict | List | np.ndarray | Any, target_class: Tuple=(), precision: int=3 ):
    """
    Function that recoursevly inspect data for classes and remove them from the data. Also convert 
    numpy arrys to lists and round floats to a given precision.

    Args:
        data (Dict | List | np.ndarray | Any): Input data.
        target_class (Tuple, optional): Class instances that should be removed from the data. Defaults to ().
        precision (int, optional): Number of decimals for floats.

    Returns:
        Dict | List | np.ndarray | Any: Input data, just without the target classes and lists instead arrays.
    """
    if isinstance(data, dict):
        return {key: serialize_json(value, target_class) for key, value in data.items() if not isinstance(value, target_class)}
    elif isinstance(data, list):
        return [serialize_json(item, target_class) for item in data]
    elif isinstance(data, np.ndarray):
        return np.round( data, precision ).tolist()
    elif isinstance(data, float):
        return round( data, precision )
    else:
        return data

def work_json(file_path: str, data: Dict={}, to_do: str="read", indent: int=2):
    """
    Function to work with json files

    Args:
        file_path (string): Path to json file
        data (dict): If write is choosen, provide input dictionary
        to_do (string): Action to do, chose between "read", "write" and "append". Defaults to "read".

    Returns:
        data (dict): If read is choosen, returns dictionary
    """
    
    if to_do=="read":
        return json.load( open(file_path) )
    
    elif to_do=="write":
        json.dump(data, open(file_path,"w"), indent=indent)

    elif to_do=="append":
        if not os.path.exists(file_path):
            json.dump(data, open(file_path,"w"), indent=indent)
        else:
            current_data = json.load( open(file_path) )
            merge_nested_dicts(current_data,data)
            json.dump(current_data, open(file_path,"w"), indent=indent)
        
    else:
        raise KeyError("Wrong task defined: %s"%to_do)


def get_system_volume( molar_masses: List[float], molecule_numbers: List[int], density: float, box_type: str="cubic" ):
    """
    Calculate the volume of a system and the dimensions of its bounding box based on molecular masses, numbers and density.

    Parameters:
    - molar_masses (List[List[float]]): A list with the molar masses of each molecule in the system.
    - molecule_numbers (List[int]): A list containing the number of molecules of each type in the system.
    - density (float): The density of the mixture in kg/m^3.
    - box_type (str, optional): The type of box to calculate dimensions for. Currently, only 'cubic' is implemented.

    Returns:
    - dict: A dictionary with keys 'box_x', 'box_y', and 'box_z', each containing a list with the negative and positive half-lengths of the box in Angstroms.

    Raises:
    - KeyError: If the `box_type` is not 'cubic', since other box types are not implemented yet.
    """
    # Account for mixture density
    molar_masses = np.array( molar_masses )

    # mole fraction of mixture (== numberfraction)
    x = np.array( molecule_numbers ) / np.sum( molecule_numbers )

    # Average molar weight of mixture [g/mol]
    M_avg = np.dot( x, molar_masses )

    # Total mole n = N/NA [mol] #
    n = np.sum( molecule_numbers ) / Avogadro

    # Total mass m = n*M, convert from g in kg. [kg]
    mass = n * M_avg / 1000

    # Volume = mass / mass_density = kg / kg/m^3, convert from m^3 to A^3. [A^3]
    volume = mass / density * 1e30


    # Compute box lenght L (in Angstrom) using the volume V=m/rho
    if box_type == "cubic":
        # Cubix box: L/2 = V^(1/3) / 2
        boxlen = volume**(1/3) / 2

        box = { "box_x": [ -boxlen, boxlen ],
                "box_y": [ -boxlen, boxlen ],
                "box_z": [ -boxlen, boxlen ]
                }
    else:
        raise KeyError(f"Specified box type '{box_type}' is not implemented yet. Available are: 'cubic'.")

    return box
