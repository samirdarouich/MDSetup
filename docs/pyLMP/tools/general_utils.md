Module pyLMP.tools.general_utils
================================

Functions
---------

    
`deep_get(obj: Union[Dict[str, Any], List[Any], Any], keys: str, default: Any = {})`
:   Function that searches an (nested) python object and extract the item at the end of the key chain.
    Keys are provided as one string and seperated by ".".
    
    Args:
        obj (Dict[str,Any]|List[Any]|Any): Object from which the (nested) keys are extracted
        keys (str): Keys to extract. Chain of keys should be seperated by ".". Integers to get list items will be converted from string.
        default (dict, optional): Default return if key is not found. Defaults to {}.
    
    Returns:
        d (Any): Element that is extracted

    
`find_key_by_value(my_dict: Dict[str, Any], target_value: str | float | int)`
:   

    
`flatten_list(lst: List[Union[List, numpy.ndarray, float, int, str]], filter_function: Callable[..., bool] = <function <lambda>>)`
:   Function that flattens a list with sublists, of items. Possibility to filter out certain types is possible via filter function.
    
    Parameters:
     - lst (List[List|np.ndarray|float|int|str]): List that should be flatten, can contain sublists or numpy arrays.
     - filter_function (Callable[...,bool]): Callable to filter out certain values if wanted. Defaults to 'lambda p: True', so no filter aplied.
    
    Returns:
     - filtered_list (List[float|int|str]): Flattended and (filtered) list.
    
    E.g: 
    test = [1,2,3,[4,5,6]] 
    flatten_list(a)
    >> [1,2,3,4,5,6]

    
`get_system_volume(molar_masses: List[float], molecule_numbers: List[int], density: float, box_type: str = 'cubic', z_x_relation: float = 1.0, z_y_relation: float = 1.0)`
:   Calculate the volume of a system and the dimensions of its bounding box based on molecular masses, numbers and density.
    
    Parameters:
    - molar_masses (List[List[float]]): A list with the molar masses of each molecule in the system.
    - molecule_numbers (List[int]): A list containing the number of molecules of each type in the system.
    - density (float): The density of the mixture in kg/m^3.
    - box_type (str, optional): The type of box to calculate dimensions for. Currently, only 'cubic' is implemented.
    - z_x_relation (float, optional): Relation of z to x length. z = z_x_relation*x. Defaults to 1.0.
    - z_y_relation (float, optional): Relation of z to y length. z = z_y_relation*y. Defaults to 1.0.
    
    Returns:
    - dict: A dictionary with keys 'box_x', 'box_y', and 'box_z', each containing a list with the negative and positive half-lengths of the box in Angstroms.
    
    Raises:
    - KeyError: If the `box_type` is not 'cubic' or 'orthorhombic', since other box types are not implemented yet.

    
`map_function_input(all_attributes: dict, argument_map: dict) ‑> dict`
:   Function that maps the elements in the attributes dictionary based on the provided mapping dictionary.
    
    Args:
        all_attributes (dict): A dictionary containing all the attributes from which the elements will be mapped.
        argument_map (dict): A dictionary that defines the mapping between the elements in all_attributes and the desired keys in the output dictionary.
    
    Returns:
        dict: A new dictionary with the mapped elements.

    
`merge_nested_dicts(existing_dict: Dict[str, Any], new_dict: Dict[str, Any])`
:   Function that merges nested dictionaries
    
    Args:
        existing_dict (Dict): Existing dictionary that will be merged with the new dictionary
        new_dict (Dict): New dictionary

    
`serialize_json(data: Union[Dict, List, numpy.ndarray, Any], target_class: Tuple = (), precision: int = 3)`
:   Function that recoursevly inspect data for classes and remove them from the data. Also convert 
    numpy arrys to lists and round floats to a given precision.
    
    Args:
        data (Dict | List | np.ndarray | Any): Input data.
        target_class (Tuple, optional): Class instances that should be removed from the data. Defaults to ().
        precision (int, optional): Number of decimals for floats.
    
    Returns:
        Dict | List | np.ndarray | Any: Input data, just without the target classes and lists instead arrays.

    
`unique_by_key(iterables: List[Union[Dict[str, Any], List[Any]]], key: str | int) ‑> List[Dict[str, Any]]`
:   Filters a list of dictionaries or other iterables, returning a list containing only the first occurrence of each unique value associated with a specified key.
    
    Args:
        dicts (List[Dict[str, Any]|List[Any]]): A list of dictionaries or lists from which to filter unique items.
        key (str|int): The key in the dictionaries or list used to determine uniqueness.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries that contains only the first dictionary for each unique value found under the specified key.

    
`work_json(file_path: str, data: Dict = {}, to_do: str = 'read', indent: int = 2)`
:   Function to work with json files
    
    Args:
        file_path (string): Path to json file
        data (dict): If write is choosen, provide input dictionary
        to_do (string): Action to do, chose between "read", "write" and "append". Defaults to "read".
    
    Returns:
        data (dict): If read is choosen, returns dictionary