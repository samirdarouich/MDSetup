Module MDSetup.analysis.mechanical_properties
=============================================

Functions
---------

    
`compute_VRH(stiffness_tensor: numpy.ndarray)`
:   Compute the Voigt-Reuss-Hill (VRH) averages of the elastic constants.
    
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

    
`get_stiffness_tensor(deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]])`
:   Calculate the stiffness tensor based on the given deformation_dict
    
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

    
`plot_deformation(deformation_dict: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]], main_only: bool = True, outpath: str = '')`
: