from typing import Dict

from ..visualize import plot_data


def plot_deformation(
    deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]],
    main_only: bool = True,
    outpath: str = "",
):
    deformation_directions = ["xx", "yy", "zz", "yz", "xz", "xy"][
        : 3 if main_only else 6
    ]
    pressure_keys = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"][: 3 if main_only else 6]

    labels = [rf"$\sigma_\mathrm{{{p.replace('p','')}}}$" for p in pressure_keys]
    set_kwargs = {"ylabel": r"$\sigma$ / GPa"}
    ax_kwargs = {
        "tick_params": {
            "which": "both",
            "right": True,
            "direction": "in",
            "width": 2,
            "length": 6,
            "pad": 8,
            "labelsize": 16,
        },
    }

    data_kwargs = [
        {"linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 10},
        {"linestyle": ":", "marker": "^", "linewidth": 2, "markersize": 10},
        {"linestyle": "--", "marker": "x", "linewidth": 2, "markersize": 10},
        {"linestyle": "-", "marker": "+", "linewidth": 2, "markersize": 10},
        {"linestyle": ":", "marker": "s", "linewidth": 2, "markersize": 10},
        {"linestyle": "--", "marker": "d", "linewidth": 2, "markersize": 10},
    ][: 3 if main_only else 6]

    for deformation_direction in deformation_directions:
        set_kwargs["xlabel"] = r"$\epsilon_\mathrm{%s}$" % deformation_direction

        pressure_dict = dict(sorted(deformation_dict[deformation_direction].items()))

        datas = [
            [
                [-dk for dk in pressure_dict],
                [val[pkey]["mean"] for _, val in pressure_dict.items()],
                None,
                [val[pkey]["std"] for _, val in pressure_dict.items()],
            ]
            for pkey in pressure_keys
        ]

        plot_data(
            datas=datas,
            labels=labels,
            save_path=f"{outpath}/epsilon_{deformation_direction}.pdf"
            if outpath
            else "",
            data_kwargs=data_kwargs,
            set_kwargs=set_kwargs,
            ax_kwargs=ax_kwargs,
            fig_kwargs={"figsize": (6.6, 4.6)},
            legend_kwargs={"fontsize": 14, "frameon": False},
            label_size=18,
        )
