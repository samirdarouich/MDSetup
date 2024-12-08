from typing import Dict

import numpy as np


def get_stiffness_tensor(
    deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]],
):
    """
    Calculate the stiffness tensor based on the given deformation_dict

    Parameters:
    deformation_dict (Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]]):
        pressure (List[List[List[float]]]): A 3D list representing the pressure values for each deformation direction.
            The innermost list contains the pressure values for one deformation magnitude in the following order:
            [pxx, pyy, pzz, pyz, pxz, pxy].
            The middle list contains the pressure values for each deformation magnitude.
            The outermost list contains the pressure values for each deformation magnitude in each deformation direction (x,y,z,yz,xz,xy).
        deformation_magnitudes (List[float]): A list of deformation magnitudes.

    Returns:
        numpy.ndarray: The stiffness tensor, a 6x6 numpy array.
            The entries of the stiffness tensor are calculated as follows:
            - C11: elongation in x direction and slope of pressure in x direction
            - C21: elongation in x direction and slope of pressure in y direction
            - C31: elongation in x direction and slope of pressure in z direction
            - C41: elongation in x direction and slope of pressure in yz direction
            - C51: elongation in x direction and slope of pressure in xz direction
            - C61: elongation in x direction and slope of pressure in xy direction

    Note:
        - The stiffness tensor is calculated by performing a linear fit for each pressure along the deformation in each direction.
        - The stiffness tensor is symmetric, so the resulting tensor is rounded to ensure symmetry.

    """
    ## Entries of stiffness tensor ##

    # C11: elongation in x direction and slope of pressure in x direction
    # C21: elongation in x direction and slope of pressure in y direction
    # C31: elongation in x direction and slope of pressure in z direction
    # C41: elongation in x direction and slope of pressure in yz direction
    # C51: elongation in x direction and slope of pressure in xz direction
    # C61: elongation in x direction and slope of pressure in xy direction

    deformation_directions = ["xx", "yy", "zz", "yz", "xz", "xy"]

    stiffness_tensor = np.zeros((6, 6))

    for i, deformation_direction in enumerate(deformation_directions):
        pressure_dict = dict(sorted(deformation_dict[deformation_direction].items()))

        deformation_magnitudes = []
        pxx, pyy, pzz = [], [], []
        pyz, pxz, pxy = [], [], []

        for deformation_rate, pressure_values in pressure_dict.items():
            deformation_magnitudes.append(deformation_rate)
            pxx.append(pressure_values["pxx"]["mean"])
            pyy.append(pressure_values["pyy"]["mean"])
            pzz.append(pressure_values["pzz"]["mean"])
            pyz.append(pressure_values["pyz"]["mean"])
            pxz.append(pressure_values["pxz"]["mean"])
            pxy.append(pressure_values["pxy"]["mean"])

        # (linear) Fit for each pressure along the deformation in xx,yy,zz,... direction
        stiffness_tensor[0][i] = -np.polyfit(deformation_magnitudes, pxx, 1)[0]
        stiffness_tensor[1][i] = -np.polyfit(deformation_magnitudes, pyy, 1)[0]
        stiffness_tensor[2][i] = -np.polyfit(deformation_magnitudes, pzz, 1)[0]
        stiffness_tensor[3][i] = -np.polyfit(deformation_magnitudes, pyz, 1)[0]
        stiffness_tensor[4][i] = -np.polyfit(deformation_magnitudes, pxz, 1)[0]
        stiffness_tensor[5][i] = -np.polyfit(deformation_magnitudes, pxy, 1)[0]

    # Ensure symmetry
    stiffness_tensor = np.round((stiffness_tensor + stiffness_tensor.T) / 2, 3)

    return stiffness_tensor
