from typing import Dict

import numpy as np

from mdsetup.analysis.general import plot_data


def show_stiffness_tensor(state_condition, final_results):
    txt = f"\nState point: {state_condition}\n"
    txt += "Averaged Stiffness Tensor\n"
    txt += "\n"
    for i in range(1, 7):
        txt += "  ".join(["C%d%d" % (i, j) for j in range(1, 7)]) + "\n"
    txt += "\n"
    for i in range(0, 6):
        txt += (
            "  ".join(
                [
                    "%.2f ± %.2f" % (st, std)
                    for st, std in zip(
                        np.array(final_results["average"]["C"]["mean"])[i, :],
                        np.array(final_results["average"]["C"]["std"])[i, :],
                    )
                ]
            )
            + "\n"
        )
    txt += "\n"
    txt += "\nAveraged mechanical properties with Voigt Reuss Hill: \n"
    txt += "\n"
    txt += "Bulk modulus K = %.0f ± %.0f GPa \n" % (
        final_results["average"]["K"]["mean"],
        final_results["average"]["K"]["std"],
    )
    txt += "Shear modulus G = %.0f ± %.0f GPa \n" % (
        final_results["average"]["G"]["mean"],
        final_results["average"]["G"]["std"],
    )
    txt += "Youngs modulus E = %.0f ± %.0f GPa \n" % (
        final_results["average"]["E"]["mean"],
        final_results["average"]["E"]["std"],
    )
    txt += "Poission ratio nu = %.3f ± %.3f \n" % (
        final_results["average"]["nu"]["mean"],
        final_results["average"]["nu"]["std"],
    )

    print(txt)


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
