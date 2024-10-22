from typing import List

import pandas as pd

from ..visualize import plot_data
from .solvation_energy import EXTRACT


def visualize_dudl(
    software: str,
    fep_files: List[str],
    T: float,
    fraction: float = 0.0,
    decorrelate: bool = True,
    save_path: str = "",
):
    # Get combined df for all lambda states
    combined_df = pd.concat(
        [
            EXTRACT["dudl"][software](
                file, T=T, fraction=fraction, decorrelate=decorrelate
            )
            for file in fep_files
        ]
    )

    # Extract vdW and Coulomb portion
    vdw_dudl = combined_df.groupby("vdw-lambda")["vdw"].agg(["mean", "std"])
    coul_dudl = combined_df.groupby("coul-lambda")["coul"].agg(["mean", "std"])

    # Plot vdW part
    datas = [
        [vdw_dudl.index.values, vdw_dudl["mean"].values, None, vdw_dudl["std"].values]
    ]
    set_kwargs = {
        "xlabel": "$\lambda_\mathrm{vdW}$",
        "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{vdW}}} \ / \ (k_\mathrm{B}T)$",
        "xlim": (0, 1),
    }
    plot_data(datas, save_path=f"{save_path}/dudl_vdw.png", set_kwargs=set_kwargs)

    # Plot Coulomb part
    datas = [
        [
            coul_dudl.index.values,
            coul_dudl["mean"].values,
            None,
            coul_dudl["std"].values,
        ]
    ]
    set_kwargs = {
        "xlabel": "$\lambda_\mathrm{coul}$",
        "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{coul}}} \ / \ (k_\mathrm{B}T)$",
        "xlim": (0, 1),
    }
    plot_data(datas, save_path=f"{save_path}/dudl_coul.png", set_kwargs=set_kwargs)
