#### General utilities ####

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Any

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
        return round( data, precision)
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
        with open(file_path) as f:
            data = json.load(f)
        return data
    
    elif to_do=="write":
        with open(file_path,"w") as f:
            json.dump(data,f,indent=indent)

    elif to_do=="append":
        if not os.path.exists(file_path):
            with open(file_path,"w") as f:
                json.dump(data,f,indent=indent)
        else:
            with open(file_path) as f:
                current_data = json.load(f)
            merge_nested_dicts(current_data,data)
            with open(file_path,"w") as f:
                json.dump(current_data,f,indent=indent)
        
    else:
        raise KeyError("Wrong task defined: %s"%to_do)



