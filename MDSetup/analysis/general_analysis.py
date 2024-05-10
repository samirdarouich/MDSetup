import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from collections.abc import Iterable
from ..tools.general import deep_get

def add_nan_if_no_brackets(lst: List[Any]):
    """
    Function that checks if round brackets are in every key of a list, if not add (NaN) to the entry
    """
    updated_list = []
    for item in lst:
        if '(' in item and ')' in item:
            updated_list.append(item)
        else:
            updated_list.append(f"{item} (NaN)")
    return updated_list

def contains_pattern(text: str, pattern: str) -> bool:
    regex = re.compile(pattern)
    return bool(regex.search(text))

def read_lammps_output( file_path: str, fraction: float=0.0, header: int=2,
                        header_delimiter: str= "," ):
    """
    Reads a LAMMPS output file and returns a pandas DataFrame containing the data.

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

    """
    with open( file_path ) as file:
        titles = [ file.readline() for _ in range(3) ]
    
    titles = [ t.replace("#","").strip() for t in titles if t.startswith("#") ]

    if len(titles) < header:
        raise KeyError(f"LAMMPS output file has only '{len(titles)}' titles. Cannot use title n°'{header}' !")
    else:
        lammps_header = [ h.strip() for h in titles[header-1].split(header_delimiter) ]

    # Check if for every key a unit is provided, if not add NaN.
    lammps_header = add_nan_if_no_brackets( lammps_header )

    df = pd.read_csv( file_path, comment="#", delimiter = " ", names = lammps_header )

    if any( key in df.columns[0].lower() for key in ["time","step"] ):
        idx = df.iloc[:,0] > df.iloc[:,0].max() * fraction
        return df.loc[idx,:]
    else:
        print(f"\nNo timestamp provided in first line of LAMMPS output. Hence, can't discard the provided fraction '{fraction}'")
        return df


def plot_data(datas: List[ List[List]], 
              labels: List[str]=[], 
              colors: List[str]=[],
              sns_context: str="poster", 
              save_path: str="", 
              label_size: int=24, 
              data_kwargs: List[Dict[str,Any]]=[],
              fig_kwargs: Dict[str,Any]={},
              set_kwargs: Dict[str,Any]={},
              ax_kwargs: Dict[str,Any]={},
              legend_kwargs: Dict[str,Any]={} ):
    
    """
    Plot data function.

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
    """
    
    # Set seaborn plot contect
    sns.set_context( sns_context )

    # Define inital figure kwargs
    if not fig_kwargs:
        fig_kwargs = { "figsize": (6.6,4.6) }

    # Define initial legend kwargs
    if not legend_kwargs:
        legend_kwargs = { "fontsize": 12 }

    # Provide inital colors
    if not colors:  
        colors = sns.color_palette("tab10", n_colors = len(datas) )
    
    # Predefine labels
    if not labels:
        labels = [ "" for _ in datas ]
    
    # If no data kwars are presented
    if not data_kwargs: 
        data_kwargs = [ {} for _ in datas ]

    if not len(labels) == len(colors) == len(data_kwargs) == len(datas):
        if len(labels) != len(datas):
            raise TypeError("Provided labels do not have the same lenght as the data!")
        elif len(colors) != len(datas):
            raise TypeError("Provided colors do not have the same lenght as the data!")
        else:
            raise TypeError("Provided data kwargs do not have the same lenght as the data!")
    
    # Create figure and ax object
    fig, ax = plt.subplots( **fig_kwargs )

    # Add label size to set kwargs and apply them
    for key,item in set_kwargs.items():
        if isinstance(item,dict):
            deep_get(ax, f"set_{key}",lambda x: None)(**item)
        elif key in ["xlabel","ylabel","title"]:
            deep_get(ax, f"set_{key}",lambda x: None)(**{ key: item, "fontsize": label_size })
        else:
            deep_get(ax, f"set_{key}",lambda x: None)(item) 

    # Apple direct ax kwargs
    for key, value in ax_kwargs.items():
        if isinstance(value,dict):
            deep_get(ax, key,lambda x: None)(**value)
        else:
            deep_get(ax, key,lambda x: None)(value)

    for data,label,color,kwargs in zip(datas,labels,colors,data_kwargs):

        error = True if len(data) == 4 else False
        # Fill is used when the y data has a np.array/list with two dimensions
        fill  = (isinstance(data[1], np.ndarray) and data[1].ndim == 2) or (isinstance(data[1],list) and all(isinstance(sublist,Iterable) for sublist in data[1]))

        if fill:
            d_kwargs = { "facecolor": color, "alpha": 0.3, "label": label, **kwargs }
            ax.fill_between (data[0], data[1][0], data[1][1], **d_kwargs )

        elif error:
            d_kwargs = { "elinewidth": mpl.rcParams['lines.linewidth']*0.7 if not "linewidth" in kwargs.keys() else kwargs["linewidth"]*0.7, 
                         "capsize": mpl.rcParams['lines.linewidth']*0.7 if not "linewidth" in kwargs.keys() else kwargs["linewidth"]*0.7, 
                         "label": label,
                         "color": color,
                         **kwargs }

            ax.errorbar( data[0], data[1], xerr = data[2], yerr = data[3], **d_kwargs )

        else:
            d_kwargs = { "label": label,
                         "color": color,
                         **kwargs }
            
            ax.plot( data[0], data[1], **d_kwargs )

    # Plot legend if any label provided
    if any(labels):
        ax.legend( **legend_kwargs )
    
    fig.tight_layout()

    if save_path: 
        os.makedirs( os.path.dirname(save_path), exist_ok=True)
        fig.savefig( save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return