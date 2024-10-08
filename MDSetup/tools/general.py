#### General utilities ####

import json
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import yaml

############## General settings ##############

# Define supported software
SOFTWARE_LIST = ["lammps", "gromacs"]

# Define suffix list
SUFFIX = {
    "topology": {"lammps": "params", "gromacs": "top"},
    "coordinate": {"lammps": "data", "gromacs": "gro"},
    "input": {"lammps": "input", "gromacs": "mdp"},
}

# Define default settings based on software
DEFAULTS = {"gromacs": {"init_step": 0, "initial_cpt": ""}, "lammps": {}}

# Define precision of folder (e.g.: f"temp_{temperature:.{FOLDER_PRECISION}f}" )
FOLDER_PRECISION = 1

# Define unit dict
UNITS = {
    "temperature": "K",
    "pressure": "bar",
    "density": "kg/m^3",
    "mole_fraction": "mol/mol",
}

# Define distance conversion (base unit is Angstrom)
DISTANCE = {"lammps": 1, "gromacs": 1 / 10}

# Define time conversion (base unit is nano seconds)
TIME = {"lammps": 1e6, "gromacs": 1e3}


# Define some error classes
class SoftwareError(Exception):
    """Software error class"""

    def __init__(self, software: str):
        message = f"Wrong software specified '{software}'. Available are: '{', '.join(SOFTWARE_LIST)}'."
        super().__init__(message)
        raise self


class KwargsError(Exception):
    """Kwargs missing error class"""

    def __init__(self, keys: List[str], kwargs_keys):
        if any(key not in kwargs_keys for key in keys):
            missing_keys = [key for key in keys if key not in kwargs_keys]
            message = (
                "Missing key in provided keyword arguments. "
                f"Expected '{', '.join(missing_keys)}'. "
                f"Available are: '{', '.join(kwargs_keys)}'."
            )
            super().__init__(message)
            raise self


class FFTypeMatchError(Exception):
    """Software error class"""

    def __init__(self, interaction_type: str):
        message = f"Unmachted {interaction_type}(s) found. " "Check force field input."
        super().__init__(message)
        raise self


############## Helpfull functions ##############


def find_key_by_value(my_dict: Dict[str, Any], target_value: str | float | int):
    for key, values in my_dict.items():
        if target_value in values:
            return key
    raise KeyError(
        f"Target value '{target_value}' is not pressented in any value "
        "of the dictionary."
    )


def unique_by_key(
    iterables: List[Dict[str, Any] | List[Any]], key: str | int
) -> List[Dict[str, Any]]:
    """
    Filters a list of dictionaries or other iterables, returning a list containing only
    the first occurrence of each unique value associated with a specified key.

    Args:
        dicts (List[Dict[str, Any]|List[Any]]): A list of dictionaries or lists from
        which to filter unique items.
        key (str|int): The key in the dictionaries or list used to determine uniqueness.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries that contains only the first
        dictionary for each unique value found under the specified key.
    """
    seen = []
    unique_iterables = []
    for d in iterables:
        name = d[key]
        if name not in seen:
            seen.append(name)
            unique_iterables.append(d)
    return unique_iterables


def deep_get(obj: Dict[str, Any] | List[Any] | Any, keys: str, default: Any = None):
    """
    Function that searches an (nested) python object and extract the item at the end of
    the key chain. Keys are provided as one string and seperated by ".".

    Args:
        obj (Dict[str,Any]|List[Any]|Any): Object from which the (nested) keys
        are extracted
        keys (str): Keys to extract. Chain of keys should be seperated by ".". Integers
        to get list items will be converted from string.
        default (dict, optional): Default return if key is not found. Defaults to {}.

    Returns:
        d (Any): Element that is extracted
    """
    if default is None:
        default = {}
    d = obj
    for key in keys.split("."):
        if isinstance(d, dict):
            d = d.get(key, default)
        elif isinstance(d, (list, np.ndarray, tuple)):
            d = d[int(key)]
        elif isinstance(d, object):
            d = getattr(d, key, default)
        else:
            raise KeyError(f"Subtype is not implemented for extraction: '{type(d)}'")

        if not isinstance(d, np.ndarray) and d == default:
            print(f"\nKey: '{key}' not found! Return default!\n")

    return d


def flatten_list(
    lst: List[List | np.ndarray | float | int | str],
    filter_function: Callable[..., bool] = lambda p: True,
):
    """
    Function that flattens a list with sublists, of items. Possibility to filter out
    certain types is possible via filter function.

    Parameters:
     - lst (List[List|np.ndarray|float|int|str]): List that should be flatten, can
     contain sublists or numpy arrays.
     - filter_function (Callable[...,bool]): Callable to filter out certain values if
     wanted. Defaults to 'lambda p: True', so no filter aplied.

    Returns:
     - filtered_list (List[float|int|str]): Flattended and (filtered) list.

    E.g:
    test = [1,2,3,[4,5,6]]
    flatten_list(a)
    >> [1,2,3,4,5,6]
    """

    flattened_list = [
        item
        for sublist in lst
        for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
    ]
    return [item for item in flattened_list if filter_function(item)]


def map_function_input(all_attributes: dict, argument_map: dict) -> dict:
    """
    Function that maps the elements in the attributes dictionary based on the provided
    mapping dictionary.

    Args:
        all_attributes (dict): A dictionary containing all the attributes from which
        the elements will be mapped.
        argument_map (dict): A dictionary that defines the mapping between the elements
        in all_attributes and the desired keys in the output dictionary.

    Returns:
        dict: A new dictionary with the mapped elements.
    """
    return {
        arg: (
            {
                subarg: deep_get(all_attributes, subitem)
                for subarg, subitem in item.items()
            }
            if isinstance(item, dict)
            else deep_get(all_attributes, item)
        )
        for arg, item in argument_map.items()
    }


def merge_nested_dicts(existing_dict: Dict[str, Any], new_dict: Dict[str, Any]):
    """
    Function that merges nested dictionaries

    Args:
        existing_dict (Dict): Existing dictionary that will be merged with the new dictionary
        new_dict (Dict): New dictionary
    """
    for key, value in new_dict.items():
        if (
            key in existing_dict
            and isinstance(existing_dict[key], dict)
            and isinstance(value, dict)
        ):
            # If both the existing and new values are dictionaries, merge them recursively
            merge_nested_dicts(existing_dict[key], value)
        else:
            # If the key doesn't exist in the existing dictionary or the values are not dictionaries, update the value
            existing_dict[key] = value


def serialize_json(
    data: Dict | List | np.ndarray | Any, target_class: Tuple = (), precision: int = 3
):
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
        return {
            key: serialize_json(value, target_class)
            for key, value in data.items()
            if not isinstance(value, target_class)
        }
    elif isinstance(data, list):
        return [serialize_json(item, target_class) for item in data]
    elif isinstance(data, np.ndarray):
        return np.round(data, precision).tolist()
    elif isinstance(data, float):
        return round(data, precision)
    else:
        return data


def work_json(file_path: str, data: Dict = None, to_do: str = "read", indent: int = 2):
    """
    Function to work with json files

    Args:
        file_path (string): Path to json file
        data (dict): If write is choosen, provide input dictionary
        to_do (string): Action to do, chose between "read", "write" and "append". Defaults to "read".

    Returns:
        data (dict): If read is choosen, returns dictionary
    """

    if data is None:
        data = {}
    if to_do == "append" and os.path.exists(file_path):
        current_data = json.load(open(file_path))
        merge_nested_dicts(current_data, data)
        json.dump(current_data, open(file_path, "w"), indent=indent)

    elif (
        to_do == "append"
        and not os.path.exists(file_path)
        or to_do != "append"
        and to_do != "read"
        and to_do == "write"
    ):
        json.dump(data, open(file_path, "w"), indent=indent)
    elif to_do == "read":
        return json.load(open(file_path))

    else:
        raise KeyError(f"Wrong task defined: {to_do}")


def load_yaml(file_path):
    """
    Load and parse a YAML file.

    Args:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        dict: The parsed content of the YAML file.
    """
    with open(file_path) as file:
        return yaml.safe_load(file)


def add_nan_if_no_brackets(lst: List[Any]):
    """
    Function that checks if round brackets are in every key of a list, if not add (NaN) to the entry
    """
    updated_list = []
    for item in lst:
        if "(" in item and ")" in item:
            updated_list.append(item)
        else:
            updated_list.append(f"{item} (NaN)")
    return updated_list


def generate_series(desired_mean, desired_std, size):
    """
    Generate a series of random numbers with a specified mean and standard deviation.

    This function creates a series of random numbers that follow a normal distribution
    with the desired mean and standard deviation. It first generates random numbers
    from a standard normal distribution, then scales and shifts them to achieve
    the desired properties.

    Parameters:
    - desired_mean (float): The mean value desired for the random numbers.
    - desired_std (float): The standard deviation desired for the random numbers.
    - size (int): The number of random numbers to generate.

    Returns:
    - numpy.ndarray: An array of random numbers with the specified mean and standard deviation.
    """
    # Generate random numbers from a standard normal distribution
    random_numbers = np.random.randn(size)

    # Calculate the Z-scores
    z_scores = (random_numbers - np.mean(random_numbers)) / np.std(random_numbers)

    return z_scores * desired_std + desired_mean


def update_paths(config: str, base_dir: str):
    """Recursively update relative paths in the config dictionary to absolute paths.

    OS is recognizing absolute paths and do not update them.
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = os.path.join(base_dir, value)
            elif isinstance(value, list):
                config[key] = [os.path.join(base_dir, item) for item in value]
            elif isinstance(value, dict):
                update_paths(value, base_dir)

    elif isinstance(config, str):
        config = os.path.join(base_dir, config)
    return config
