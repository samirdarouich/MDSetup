#### General utilities ####

import os
import json
import numpy as np

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



