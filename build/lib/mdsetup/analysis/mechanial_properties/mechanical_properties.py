from typing import List

import numpy as np

from mdsetup.tools.general import work_json
from .stiffness import get_stiffness_tensor
from .visualize import plot_deformation
from .voigt_reuss_hill import compute_VRH

# Properties to extract
PROPERTIES = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]

# Define deformation directions (order used to compute stiffness tensor)
DEFORMATION_DIRECTIONS = ["xx", "yy", "zz", "yz", "xz", "xy"]

METHOD = {"VRH": compute_VRH}


def analysis_mechanical_prpoerties(
    self,
    analysis_folder: str,
    ensemble: str,
    deformation_rates: List[float],
    method: str = "VRH",
    time_fraction: float = 0.0,
    visualize_stress_strain: bool = False,
):
    # Flag to only evaluate the undeformed system once
    flag_undeformed = False

    # State dictionary for results
    state_dict = {}

    for deformation_direction in DEFORMATION_DIRECTIONS:
        for deformation_rate in deformation_rates:
            if deformation_rate == 0.0:
                if not flag_undeformed:
                    # Define the analysis folder
                    tmp_analysis_folder = (
                        f"{analysis_folder}/undeformed/{deformation_rate}"
                    )

                    # Extract properties from LAMMPS and analyse them
                    self.analysis_extract_properties(
                        analysis_folder=tmp_analysis_folder,
                        ensemble=ensemble,
                        extracted_properties=PROPERTIES,
                        output_suffix="pressure",
                        time_fraction=time_fraction,
                    )

                    flag_undeformed = True
                else:
                    # Define the analysis folder
                    tmp_analysis_folder = (
                        f"{analysis_folder}/undeformed/{deformation_rate}"
                    )
            else:
                # Define the analysis folder
                tmp_analysis_folder = (
                    f"{analysis_folder}/{deformation_direction}/{deformation_rate}"
                )

                # Extract properties from LAMMPS and analyse them
                self.analysis_extract_properties(
                    analysis_folder=tmp_analysis_folder,
                    ensemble=ensemble,
                    extracted_properties=PROPERTIES,
                    output_suffix="pressure",
                    time_fraction=time_fraction,
                )

            # Add each copy to the results dict
            for i, (temperature, pressure, density) in enumerate(
                zip(
                    self.system_setup["temperature"],
                    self.system_setup["pressure"],
                    self.system_setup["density"],
                )
            ):
                # Compute mole fraction of component 1
                mole_fraction = self.molecule_numbers[0] / (sum(self.molecule_numbers))

                # Define folder with defined state attributes
                state_condition = self.define_state_cond(
                    temperature=temperature,
                    pressure=pressure,
                    density=density,
                    mole_fraction=mole_fraction,
                )

                json_path = (
                    f"{self.project_folder}/{tmp_analysis_folder}/"
                    f"{state_condition}/results.json"
                )

                # Convert from atm in GPa
                for copy, copy_dict in work_json(json_path)[ensemble]["data"].items():
                    sim_results = {}
                    for key, prop in copy_dict.items():
                        if key in PROPERTIES:
                            prop["mean"] *= 101325 / 1e9
                            prop["std"] *= 101325 / 1e9
                            prop["unit"] = "GPa"
                            sim_results[key] = prop

                    state_dict.setdefault((temperature, pressure), {}).setdefault(
                        copy, {}
                    ).setdefault(deformation_direction, {}).update(
                        {deformation_rate: sim_results}
                    )

    # Visualize if specified
    if visualize_stress_strain:
        for (temperature, pressure), s_dict in state_dict.items():
            print(f"Temperature: {temperature:.1f} K, pressure: {pressure:.1f} bar")

            plot_deformation(
                s_dict["average"],
                outpath=f"{self.project_folder}/deformation",
                main_only=True,
            )

    # Get the mechanical properties over each copy and from it the mean and standard deviation
    for (temperature, pressure), s_dict in state_dict.items():
        final_results = {}

        state_condition = self.define_state_cond(
            temperature=temperature,
            pressure=pressure,
        )

        for copy, deformation_dict in s_dict.items():
            if copy == "average":
                continue

            # Get slope of stress-strain curve at low deformation
            stiffness_tensor = get_stiffness_tensor(deformation_dict)

            # Computation of mechanical properties

            # Voigt Reuss Hill
            K, G, E, nu = METHOD[method](stiffness_tensor)

            final_results[copy] = {
                "C": {"mean": stiffness_tensor.tolist()},
                "K": {"mean": K},
                "G": {"mean": G},
                "E": {"mean": E},
                "nu": {"mean": nu},
            }

        # Get the average over all copies
        final_results["average"] = {
            key: {
                "mean": np.mean(
                    [copy_data[key]["mean"] for copy_data in final_results.values()],
                    axis=0,
                ).tolist(),
                "std": np.std(
                    [copy_data[key]["mean"] for copy_data in final_results.values()],
                    ddof=1,
                    axis=0,
                ).tolist(),
            }
            for key in ["C", "K", "G", "E", "nu"]
        }

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

        work_json(
            file_path=(
                f"{self.project_folder}/{analysis_folder}/results_mechanical_"
                f"{state_condition}.json"
            ),
            data=final_results,
            to_do="write",
        )
