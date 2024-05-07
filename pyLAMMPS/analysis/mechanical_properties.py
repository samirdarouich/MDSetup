import numpy as np
from typing import Dict, List

from .general_analysis import plot_data


def plot_deformation( deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]], main_only: bool=True, outpath: str="" ):

    deformation_directions = [ "xx", "yy", "zz", "yz", "xz", "xy" ][:3 if main_only else 6] 
    pressure_keys = [ "pxx", "pyy", "pzz", "pxy", "pxz", "pyz" ][:3 if main_only else 6] 
    
    labels = [  rf"$\sigma_\mathrm{{{p.replace('p','')}}}$" for p in pressure_keys]
    set_kwargs = { "ylabel": r"$\sigma$ / GPa" }
    ax_kwargs = { "tick_params": {"which": "both", "right": True, "direction": "in", "width": 2, "length": 6, "pad":8, "labelsize":16 },
                }

    data_kwargs = [ { "linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 10 }, { "linestyle": ":", "marker": "^", "linewidth": 2, "markersize": 10 }, { "linestyle": "--", "marker": "x", "linewidth": 2, "markersize": 10 },
                    { "linestyle": "-", "marker": "+", "linewidth": 2, "markersize": 10 }, { "linestyle": ":", "marker": "s", "linewidth": 2, "markersize": 10 }, { "linestyle": "--", "marker": "d", "linewidth": 2, "markersize": 10 } ][:3 if main_only else 6] 
    
    for deformation_direction in deformation_directions:

        set_kwargs["xlabel"] = r"$\epsilon_\mathrm{%s}$"%deformation_direction
        
        pressure_dict = dict( sorted( deformation_dict[deformation_direction].items() ) )

        datas = [ [ [-dk for dk in pressure_dict], [val[pkey]["mean"] for _,val in pressure_dict.items()], None, [val[pkey]["std"] for _,val in pressure_dict.items()] ] for pkey in pressure_keys ]

        plot_data( datas = datas,
                labels = labels, 
                save_path = f"{outpath}/epsilon_{deformation_direction}.pdf" if outpath else "",
                data_kwargs = data_kwargs,
                set_kwargs = set_kwargs,
                ax_kwargs = ax_kwargs,
                fig_kwargs = { "figsize": (6.6,4.6) },
                legend_kwargs= { "fontsize": 14, "frameon": False },
                label_size = 18
        )


def get_stiffness_tensor( deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]] ):
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

    deformation_directions = [ "xx", "yy", "zz", "yz", "xz", "xy" ]

    stiffness_tensor = np.zeros((6,6))

    for i, deformation_direction in enumerate( deformation_directions ):
            
        pressure_dict = dict( sorted( deformation_dict[deformation_direction].items() ) )

        deformation_magnitudes = []
        pxx, pyy, pzz = [],[],[]
        pyz, pxz, pxy = [],[],[]
        
        for deformation_rate, pressure_values in pressure_dict.items():
            deformation_magnitudes.append( deformation_rate )
            pxx.append( pressure_values["pxx"]["mean"] )
            pyy.append( pressure_values["pyy"]["mean"] )
            pzz.append( pressure_values["pzz"]["mean"] )
            pyz.append( pressure_values["pyz"]["mean"] )
            pxz.append( pressure_values["pxz"]["mean"] )
            pxy.append( pressure_values["pxy"]["mean"] )

        # (linear) Fit for each pressure along the deformation in xx,yy,zz,... direction
        stiffness_tensor[0][i] = -np.polyfit( deformation_magnitudes, pxx, 1 )[0]
        stiffness_tensor[1][i] = -np.polyfit( deformation_magnitudes, pyy, 1 )[0]
        stiffness_tensor[2][i] = -np.polyfit( deformation_magnitudes, pzz, 1 )[0]
        stiffness_tensor[3][i] = -np.polyfit( deformation_magnitudes, pyz, 1 )[0]
        stiffness_tensor[4][i] = -np.polyfit( deformation_magnitudes, pxz, 1 )[0]
        stiffness_tensor[5][i] = -np.polyfit( deformation_magnitudes, pxy, 1 )[0] 

    # Ensure symmetry
    stiffness_tensor = np.round( ( stiffness_tensor + stiffness_tensor.T ) / 2 , 3 )

    return stiffness_tensor

def compute_VRH( stiffness_tensor: np.ndarray ):
    """
    Compute the Voigt-Reuss-Hill (VRH) averages of the elastic constants.

    Parameters:
        stiffness_tensor (np.ndarray): The 6x6 stiffness tensor representing the elastic constants.

    Returns:
        tuple: A tuple containing the VRH averages of the elastic constants in the following order:
            - K (float): Bulk modulus.
            - G (float): Shear modulus.
            - E (float): Young's modulus.
            - nu (float): Poisson's ratio.

    Notes:
        - The stiffness tensor should be in Voigt notation.
        - The stiffness tensor should be symmetric.
        - The stiffness tensor should be positive definite.

    References:
        - Voigt, W. (1928). Lehrbuch der Kristallphysik. Teubner.
        - Reuss, A. (1929). Berechnung der Fließgrenze von Mischkristallen auf Grund der Plastizitätsbedingung für Einkristalle. ZAMM - Journal of Applied Mathematics and Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik, 9(1), 49–58.
        - Hill, R. (1952). The Elastic Behaviour of a Crystalline Aggregate. Proceedings of the Physical Society. Section A, 65(5), 349–354.
    """
    c_sq = ( stiffness_tensor[0][0] + stiffness_tensor[0][1] ) * stiffness_tensor[2][2] - 2 * stiffness_tensor[0][2]**2
    M    = stiffness_tensor[0][0] + stiffness_tensor[0][1] + 2 * stiffness_tensor[2][2] - 4 * stiffness_tensor[0][2]

    K_V  = ( 1 / 9 * ( stiffness_tensor[0][0] + stiffness_tensor[1][1] + stiffness_tensor[2][2] ) + 
            2 / 9 * ( stiffness_tensor[0][1] + stiffness_tensor[0][2] + stiffness_tensor[1][2] ) )

    K_R  = c_sq / M

    G_V  = 1 / 30 * ( M + 12 * stiffness_tensor[3][3] + 12 * stiffness_tensor[5][5] ) 

    G_R  = ( 5 / 2 * c_sq * stiffness_tensor[3][3] * stiffness_tensor[5][5] / 
            ( c_sq * ( stiffness_tensor[3][3] + stiffness_tensor[5][5] ) + 3 * K_V * stiffness_tensor[3][3] * stiffness_tensor[5][5] ) )

    K    = 0.5 * ( K_V + K_R )
    G    = 0.5 * ( G_V + G_R )
    E    = 9 * K * G / ( 3 * K + G )
    nu   = (3 * K - 2 * G ) / ( 6 * K + 2 * G )

    return K, G, E, nu
