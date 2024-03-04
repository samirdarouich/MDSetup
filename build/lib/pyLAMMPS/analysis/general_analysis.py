import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib.ticker import AutoMinorLocator

def mean_properties(file: str, keys: List[str]=[], fraction: float=0.7, average: bool=True) -> pd.DataFrame:
    
    """
    Function that reads in a LAMMPS output file and return (time average) of given properties
    
    Parameters
    ----------
    
    file (str): Path to LAMMPS sampling output file 
    keys (List[str]): Keys to average from output (do not include TimeStep)
    fraction (float, optional): Fraction of simulation output that is discarded from the beginning. Defaults to 0.7.
    average (bool, optional): If the whole dataframe is returned or only the time averaged values.

    Return
    ------
    
    data (DataFrame): (Time averaged) properties
    """
    
    with open(file) as f:
        f.readline()
        keys_lmp = f.readline().split()
        idx_spec_keys = np.array([0]+[keys_lmp.index(k)-1 for k in keys if k in keys_lmp])
        lines = np.array([np.array(line.split("\n")[0].split()).astype("float")[idx_spec_keys] for line in f])

    time       = lines[:,0]
    start_time = fraction*time[-1]
    
    data       = pd.DataFrame( { key: lines[:,i][time>start_time] for i,key in enumerate( ["timestep", *keys] ) } )
    
    if average:
        return data.mean()
    else:
        return data


def plot_data(datas: List[ List[List]], labels: List[str], colors: List[str],
              path_out: str="", linestyle: List[str]=[], markerstyle: List[str]=[], ax_lim: List[List[float]]=[[],[]],
              ticks: List[np.ndarray]=[np.array([]),np.array([])], label_size: int=24,
              legend_size: int=18, tick_size: int=24, size: Tuple[float]=(8,6),lr: bool=False, fill: List[bool]=[]):
    
    """
    Function that plots data.
    
    Args:
        datas (List[ List[ List, List, List, List ]]): Data list containing in each sublist x and y data 
                                                       (if errorbar plot is desired, datas should contain as 3rd entry the x-error and as 4th entry the y-error )
        labels (List[str]): Label list containing a label for each data in datas. 2nd last entry is x-axis label and last entry for y-axis
        colors (List[str]): Color list containing a color for each data in datas.
        path_out (str, optional): Path to save plot.
        linestyle (List[str], optional): Linestyle list containing a linestyle for each data in datas.
        markerstyle (List[str], optional): Markerstyle list containing a markerstyle for each data in datas.
        ax_lim (List[List[float],List[float]], optional): List with ax limits. First entry x ax, 2nd entry y ax.
        ticks (List[np.ndarray, np.ndarray] optional): List with ticks for x and y axis. First entry x ax, 2nd entry y ax.
        label_size (int,optional): Size of axis labels (also influence the linewidth with factor 1/7). Defaults to 24.
        legend_size (int,optional): Size of labels in legend. Defaults to 18.
        tick_size (int,optional): Size of ticks on axis. Defaults to 24.
        size (Tuple[float, float],optional): Figure size. Defaults to (8,6).
        lr (boolean,optional): If true, then the plot will have y ticks on both sides.
        fill (list of booleans): If entry i is True then a filled plot will be made. Therefore, datas[i][1] should contain 2 entries -> y_low,y_high instead of one -> y
    """
    
    if not bool(fill): fill = [False for _ in datas]

    fig,ax = plt.subplots(figsize=size)

    for i,(data,label,color) in enumerate(zip(datas,labels,colors)):

        # Prevent plotting of emtpy data:
        if (not np.any(data[0]) and not isinstance(data[0],float)) or \
           (not np.any(data[1])) or \
           (not np.any(data[0]) and not np.any(data[1])): continue

        ls       = linestyle[i] if len(linestyle)>0 else "solid"
        ms       = markerstyle[i] if len(markerstyle)>0 else "."
        fill_bol = fill[i]
        error    = True if len(data) == 4 else False

        if fill_bol:
            ax.fill_between(data[0],data[1][0],data[1][1], facecolor=color,alpha=0.3)
        elif error:
            ax.errorbar(data[0],data[1],xerr=data[2],yerr=data[3],linestyle=ls,marker=ms,markersize=label_size/2,
                        linewidth=label_size/7,elinewidth=label_size/7*0.7,capsize=label_size/5,color=color,label=label )
        else:
            ax.plot(data[0],data[1],linestyle=ls,marker=ms,markersize=label_size/2,linewidth=label_size/7,color=color,label=label)

    # Plot legend if any label provided
    if any(labels[:-2]):
        ax.legend(fontsize=legend_size)
    
    try: 
        if len(ax_lim[0]) > 0: ax.set_xlim(*ax_lim[0])
    except: pass
    try: 
        if len(ax_lim[1]) > 0: ax.set_ylim(*ax_lim[1])
    except: pass

    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(2)

    if len(ticks[0])>0:
        ax.set_xticks(ticks[0])
    elif len(ticks[1])>0:
        ax.set_yticks(ticks[1])

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major",labelsize=tick_size,direction="out",width=tick_size/9,length=tick_size/5,right=lr)
    ax.tick_params(which="minor",labelsize=tick_size,direction="out",width=tick_size/12,length=tick_size/8,right=lr)
    ax.set_xlabel(labels[-2],fontsize=label_size)
    ax.set_ylabel(labels[-1],fontsize=label_size)
    fig.tight_layout()
    if bool(path_out): fig.savefig(path_out,dpi=400,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return