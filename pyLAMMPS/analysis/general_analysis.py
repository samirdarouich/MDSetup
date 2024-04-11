import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from collections.abc import Iterable

def contains_pattern(text: str, pattern: str) -> bool:
    regex = re.compile(pattern)
    return bool(regex.search(text))

def read_lammps_output(file: str, keys: List[str]=[], fraction: float=0.0, average: bool=True) -> pd.DataFrame:
    
    """
    Function that reads in a LAMMPS output file and return (time average) of given properties
    
    Parameters
    ----------
    
    file (str): Path to LAMMPS sampling output file 
    keys (List[str]): Keys to average from output (do not include step).
    fraction (float, optional): Fraction of simulation output that is discarded from the beginning. Defaults to 0.0.
    average (bool, optional): If the whole dataframe is returned or only the time averaged values.

    Return
    ------
    
    data (DataFrame): (Time averaged) properties
    """
    
    with open(file) as f:
        # Skip first line as it is title
        f.readline()

        # 2nd line is first header
        header = f.readline().replace("#","").strip()

        # Check if there is a second header
        snd_header  = f.readline()
        header_flag = False

        # get the length of the first header
        first_len = len(header.replace("#","").split())
        
        # If there is a 2nd header extract the arguments from there
        if snd_header.startswith("#"):
            header = snd_header.replace("#","").strip()
            header_flag = True
            print("2nd header found. Match keys with this header!\n")

        if "," in header:
            bracket_bool = all( contains_pattern(i,r'\(*.\)') for i in header.split(",") )
            keys_lmp = [ i.strip() if bracket_bool else i.strip() + "(NaN)" for i in header.split(",") ]
        elif ")" in header:
            keys_lmp = [ i.strip()+")" for i in header.split(")")[:-1] ]
        else:
            bracket_bool = all( contains_pattern(i,r'\(*.\)') for i in header.split() )
            keys_lmp = [ i.strip() if bracket_bool else i.strip() + " (NaN)" for i in header.split() ]
        
        # Match keys with LAMMPS keys
        matching_keys = [ i.split("(")[0].strip() for i in keys_lmp ]
        idx_spec_keys = np.array( [ matching_keys.index(key) for key in keys ] )

        if len(idx_spec_keys) != len(keys):
            raise KeyError(f"Not all provided keys are found in LAMMPS file! Found keys are: "+ "\n".join([keys_lmp[i] for i in idx_spec_keys]))

        # Read in data
        if header_flag:
            final_keys = [ "step (fs)" ] + keys_lmp
            lines = [  ]
        else:
            # In case no 2nd header is used, the first line represents the time
            idx_spec_keys = np.insert( idx_spec_keys, 0, 0 )
            final_keys = np.array(keys_lmp)[idx_spec_keys]
            lines = [ np.array( snd_header.split("\n")[0].split() ).astype("float")[idx_spec_keys] ]

        for line in f:
            if header_flag:
                if len(line.split()) == first_len:
                    time = float( line.split()[0] )
                else:
                    # Add time stamp to data
                    lines.append( np.insert(  np.array( line.split("\n")[0].split() ).astype("float")[idx_spec_keys], 0, time ) )
            else:
                lines.append( np.array( line.split("\n")[0].split() ).astype("float")[idx_spec_keys] )
    
    lines      = np.array(lines)
    time       = lines[:,0]
    start_time = fraction*time[-1]
    
    data       = pd.DataFrame( { key: lines[:,i][time>start_time] for i,key in enumerate( final_keys ) } )
    
    if average:
        return data.mean()
    else:
        return data


def plot_data(datas: List[ List[List]], 
              labels: List[str]=[], 
              colors: List[str]=[],
              sns_context: str="paper", 
              save_path: str="", 
              label_size: int=24, 
              data_kwargs: List[Dict[str,Any]]=[],
              fig_kwargs: Dict[str,Any]={},
              set_kwargs: Dict[str,Any]={},
              ax_kwargs: Dict[str,Any]={},
              legend_kwargs: Dict[str,Any]={} ):
    
    
    
    # Set seaborn plot contect
    sns.set_context( sns_context )

    # Define inital figure kwargs
    if not fig_kwargs:
        fig_kwargs = { "figsize": (8,6) }

    # Define initial legend kwargs
    if not legend_kwargs:
        legend_kwargs = { "fontsize": 12 }
    
    # Predefine ax kwargs
    if not ax_kwargs:
        ax_kwargs = { "minorticks_on": {} }

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
            getattr(ax, f"set_{key}")(**item)
        elif key in ["xlabel","ylabel","title"]:
            getattr(ax, f"set_{key}")(**{ key: item, "fontsize": label_size })
        else:
            getattr(ax, f"set_{key}")(item) 

    # Apple direct ax kwargs
    for key, value in ax_kwargs.items():
        if not isinstance(value,dict):
            raise TypeError(f"Specified ax kwargs is not a dict: '{key}': ", print(value))
        getattr(ax, key)(**value)

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