Module pyLMP.analysis.general_analysis
======================================

Functions
---------

    
`add_nan_if_no_brackets(lst: List[Any])`
:   Function that checks if round brackets are in every key of a list, if not add (NaN) to the entry

    
`contains_pattern(text: str, pattern: str) ‑> bool`
:   

    
`plot_data(datas: List[List[List]], labels: List[str] = [], colors: List[str] = [], sns_context: str = 'poster', save_path: str = '', label_size: int = 24, data_kwargs: List[Dict[str, Any]] = [], fig_kwargs: Dict[str, Any] = {}, set_kwargs: Dict[str, Any] = {}, ax_kwargs: Dict[str, Any] = {}, legend_kwargs: Dict[str, Any] = {})`
:   Plot data function.
    
    This function plots data using matplotlib and seaborn libraries.
    
    Parameters:
    - datas (List[List[List]]): A list of data to be plotted. Each element in the list represents a separate dataset. Each dataset is a list of two or four elements. 
                                If the dataset has two elements, it represents x and y values. If the dataset has four elements, it represents x, y, x error, and y error values.
                                If the dataset has two elemenets and the y value contains sublists, it represents x, y_upper, y_lower
    - labels (List[str], optional): A list of labels for each dataset. Default is an empty list.
    - colors (List[str], optional): A list of colors for each dataset. Default is an empty list.
    - sns_context (str, optional): The seaborn plot context. Default is "poster".
    - save_path (str, optional): The path to save the plot. Default is an empty string.
    - label_size (int, optional): The font size of the labels. Default is 24.
    - data_kwargs (List[Dict[str,Any]], optional): A list of dictionaries containing additional keyword arguments for each dataset. Default is an empty list.
    - fig_kwargs (Dict[str,Any], optional): A dictionary containing additional keyword arguments for the figure. Default is an empty dictionary.
    - set_kwargs (Dict[str,Any], optional): A dictionary containing additional keyword arguments for setting properties of the axes. Default is an empty dictionary.
    - ax_kwargs (Dict[str,Any], optional): A dictionary containing additional keyword arguments for the axes. Default is an empty dictionary.
    - legend_kwargs (Dict[str,Any], optional): A dictionary containing additional keyword arguments for the legend. Default is an empty dictionary.
    
    Returns:
    None

    
`read_lammps_output(file_path: str, fraction: float = 0.0, header: int = 2, header_delimiter: str = ',')`
:   Reads a LAMMPS output file and returns a pandas DataFrame containing the data.
    
    Parameters:
        file_path (str): The path to the LAMMPS output file.
        fraction (float, optional): The fraction of data to keep based on the maximum value of the first column. Defaults to 0.0.
        header (int, optional): The number of header lines from which to extract the keys for the reported values. Defaults to 2.
        header_delimiter (str, optional): The delimiter used in the header line. Defaults to ",".
    
    Returns:
        pandas.DataFrame: The DataFrame containing the data from the LAMMPS output file.
    
    Raises:
        KeyError: If the LAMMPS output file does not have enough titles.
    
    Note:
        - The function assumes that the LAMMPS output file has a timestamp in the first line.
        - If the timestamp is not present, the provided fraction parameter will be ignored.
        - The function expects the LAMMPS output file to have a specific format, with titles starting with '#'.