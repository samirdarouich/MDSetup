import os
from collections.abc import Iterable
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mdsetup.tools.general import deep_get


def plot_data(
    datas: List[List[List]],
    labels: List[str] = [],
    colors: List[str] = [],
    sns_context: str = "poster",
    save_path: str = "",
    label_size: int = 24,
    data_kwargs: List[Dict[str, Any]] = [],
    fig_kwargs: Dict[str, Any] = {},
    set_kwargs: Dict[str, Any] = {},
    ax_kwargs: Dict[str, Any] = {},
    legend_kwargs: Dict[str, Any] = {},
):
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
    sns.set_context(sns_context)

    # Define inital figure kwargs
    if not fig_kwargs:
        fig_kwargs = {"figsize": (6.6, 4.6)}

    # Define initial legend kwargs
    if not legend_kwargs:
        legend_kwargs = {"fontsize": 12}

    # Provide inital colors
    if not colors:
        colors = sns.color_palette("tab10", n_colors=len(datas))

    # Predefine labels
    if not labels:
        labels = ["" for _ in datas]

    # If no data kwars are presented
    if not data_kwargs:
        data_kwargs = [{} for _ in datas]

    if not len(labels) == len(colors) == len(data_kwargs) == len(datas):
        if len(labels) != len(datas):
            raise TypeError("Provided labels do not have the same lenght as the data!")
        elif len(colors) != len(datas):
            raise TypeError("Provided colors do not have the same lenght as the data!")
        else:
            raise TypeError(
                "Provided data kwargs do not have the same lenght as the data!"
            )

    # Create figure and ax object
    fig, ax = plt.subplots(**fig_kwargs)

    # Add label size to set kwargs and apply them
    for key, item in set_kwargs.items():
        if isinstance(item, dict):
            deep_get(ax, f"set_{key}", lambda x: None)(**item)
        elif key in ["xlabel", "ylabel", "title"]:
            deep_get(ax, f"set_{key}", lambda x: None)(
                **{key: item, "fontsize": label_size}
            )
        else:
            deep_get(ax, f"set_{key}", lambda x: None)(item)

    # Apple direct ax kwargs
    for key, value in ax_kwargs.items():
        if isinstance(value, dict):
            deep_get(ax, key, lambda x: None)(**value)
        else:
            deep_get(ax, key, lambda x: None)(value)

    for data, label, color, kwargs in zip(datas, labels, colors, data_kwargs):
        error = True if len(data) == 4 else False
        # Fill is used when the y data has a np.array/list with two dimensions
        fill = (isinstance(data[1], np.ndarray) and data[1].ndim == 2) or (
            isinstance(data[1], list)
            and all(isinstance(sublist, Iterable) for sublist in data[1])
        )

        if fill:
            d_kwargs = {"facecolor": color, "alpha": 0.3, "label": label, **kwargs}
            ax.fill_between(data[0], data[1][0], data[1][1], **d_kwargs)

        elif error:
            d_kwargs = {
                "elinewidth": mpl.rcParams["lines.linewidth"] * 0.7
                if "linewidth" not in kwargs.keys()
                else kwargs["linewidth"] * 0.7,
                "capsize": mpl.rcParams["lines.linewidth"] * 0.7
                if "linewidth" not in kwargs.keys()
                else kwargs["linewidth"] * 0.7,
                "label": label,
                "color": color,
                **kwargs,
            }

            ax.errorbar(data[0], data[1], xerr=data[2], yerr=data[3], **d_kwargs)

        else:
            d_kwargs = {"label": label, "color": color, **kwargs}

            ax.plot(data[0], data[1], **d_kwargs)

    # Plot legend if any label provided
    if any(labels):
        ax.legend(**legend_kwargs)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()

    return
