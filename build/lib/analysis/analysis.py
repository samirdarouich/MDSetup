from typing import List

from ..base.base import BaseSetup
from .mechanial_properties import analysis_mechanical_proerties
from .solvation_free_energy import analysis_solvation_free_energy


class Analysis(BaseSetup):
    def __init__(
        self,
        system_setup: str,
        simulation_default: str,
        simulation_ensemble: str,
        submission_command: str,
        simulation_sampling: str = "",
    ):
        super().__init__(
            system_setup=system_setup,
            simulation_default=simulation_default,
            simulation_ensemble=simulation_ensemble,
            submission_command=submission_command,
            simulation_sampling=simulation_sampling,
        )

    def analysis_solvation_free_energy(
        self,
        analysis_folder: str,
        ensemble: str,
        method: str = "MBAR",
        time_fraction: float = 0.0,
        decorrelate: bool = True,
        coupling: bool = True,
    ):
        analysis_solvation_free_energy(
            self=self,
            analysis_folder=analysis_folder,
            ensemble=ensemble,
            method=method,
            time_fraction=time_fraction,
            decorrelate=decorrelate,
            coupling=coupling,
        )

    def analysis_mechanical_proerties(
        self,
        analysis_folder: str,
        ensemble: str,
        deformation_rates: List[float],
        method: str = "VRH",
        time_fraction: float = 0.0,
        visualize_stress_strain: bool = False,
    ):
        analysis_mechanical_proerties(
            self=self,
            analysis_folder=analysis_folder,
            ensemble=ensemble,
            deformation_rates=deformation_rates,
            method=method,
            time_fraction=time_fraction,
            visualize_stress_strain=visualize_stress_strain,
        )
