Module pyLMP.tools.lammps_utils
===============================

Functions
---------

    
`get_bonded_style(bonded_numbers: List[int], bonded_dict: List[Dict[str, Any]], n_eval: int = 1000)`
:   Get bonded styles and parameters for a given list of bonded numbers and bonded dictionary for LAMMPS ff input. 
    This can be used for bonds, angles, dihedrals, etc...
    
    Parameters:
    - bonded_numbers (List[int]): A list of integers representing the bonded numbers.
    - bonded_dict (List[Dict[str, Any]]): A list with dictionaries containing the bonded styles and parameters.
    - n_eval (int, optional): The number of evaluations. Default is 1000.
    
    Returns:
    - bonded_styles (List[str]): A list of unique bonded styles.
    - bonded_paras (List[List]): A list of bonded parameters.

    
`get_coupling_lambdas(combined_lambdas: List[float], coupling: bool = True, precision: int = 3)`
:   Calculate the van der Waals (vdW) and Coulomb coupling lambdas.
    
    Parameters:
    - combined_lambdas (List[float]): A list of combined lambdas.
    - coupling (bool, optional): Whether to calculate coupling or decoupling lambdas. Defaults to true.
    - precision (int, optional): The precision of the lambdas (default is 3).
    
    Returns:
    - vdw_lambdas (List[str]): A list of formatted vdW lambdas.
    - coul_lambdas (List[str]): A list of formatted Coulomb lambdas.
    
    The vdW lambdas are calculated as the minimum of each combined lambda and 1.0, rounded to the specified precision.
    The Coulomb lambdas are calculated as the maximum of each combined lambda minus 1.0 and 0.0, rounded to the specified precision.
    If coupling is False, the vdW lambdas are calculated as 1 minus the maximum of each combined lambda minus 1.0, rounded to the specified precision.
    If coupling is False, the Coulomb lambdas are calculated as 1 minus the minimum of each combined lambda and 1.0, rounded to the specified precision.
    The Coulomb lambdas are adjusted to be at least 1e-9 to avoid division by 0.

    
`get_mixed_parameter(sigma_i: float, sigma_j: float, epsilon_i: float, epsilon_j: float, mixing_rule: str = 'arithmetic', precision: int = 4)`
:   Calculate the mixed parameters for a pair of interacting particles.
    
    Parameters:
        sigma_i (float): The sigma parameter of particle i.
        sigma_j (float): The sigma parameter of particle j.
        epsilon_i (float): The epsilon parameter of particle i.
        epsilon_j (float): The epsilon parameter of particle j.
        mixing_rule (str, optional): The mixing rule to use. Valid options are "arithmetic", "geometric", and "sixthpower". Defaults to "arithmetic".
        precision (int, optional): The number of decimal places to round the results to. Defaults to 4.
    
    Returns:
        tuple: A tuple containing the mixed sigma and epsilon parameters.
    
    Raises:
        KeyError: If the specified mixing rule is not implemented.
    
    Example:
        >>> get_mixed_parameter(3.5, 2.5, 0.5, 0.8, mixing_rule="arithmetic", precision=3)
        (3.0, 0.632)

    
`get_pair_style(local_attributes: Dict[str, Any], vdw_pair_styles: List[str], coul_pair_styles: List[str], pair_style_kwargs: Dict[str, str])`
:   This function takes in several parameters and returns a string representing the combined pair style for a molecular simulation.
    
    Parameters:
    - local_attributes (Dict[str,Any]): A dictionary containing local attributes for the pair style.
    - vdw_pair_styles (List[str]): A list of strings representing the Van der Waals pair styles to be used.
    - coul_pair_styles (List[str]): A list of strings representing the Coulombic pair styles to be used.
    - pair_style_kwargs (Dict[str,str]): A dictionary mapping pair styles to their corresponding arguments.
    
    Returns:
    - A string representing the combined pair style for the simulation.
    
    The function iterates over the unique Van der Waals pair styles and Coulombic pair styles provided. For each pair style, it constructs a substring 
    by concatenating the pair style name with the corresponding arguments from the local_attributes dictionary. These substrings are then appended to the combined_pair_style list.
    
    If multiple pair styles are used, the function inserts the "hybrid/overlay" style at the beginning of the combined_pair_style list.
    
    Finally, the function returns the combined pair style as a string, with each pair style separated by two spaces.
    
    Note: The function assumes that the local_attributes dictionary contains all the necessary arguments for each pair style specified 
          in vdw_pair_styles and coul_pair_styles. If an argument is missing, a KeyError will be raised.

    
`write_coupled_lammps_ff(ff_template: str, lammps_ff_path: str, potential_kwargs: Dict[str, List[str]], solute_numbers: List[int], combined_lambdas: List[float], coupling_potential: Dict[str, Any], coupling_soft_core: Dict[str, float], atom_numbers_ges: List[int], nonbonded: List[Dict[str, str]], bond_numbers_ges: List[int], bonds: List[Dict[str, str]], angle_numbers_ges: List[int], angles: List[Dict[str, str]], torsion_numbers_ges: List[int], torsions: List[Dict[str, str]], mixing_rule: str, shake_dict: Dict[str, List[Union[str, int]]] = {'t': [], 'b': [], 'a': []}, coupling: bool = True, precision: int = 3, ff_kwargs: Dict[str, Any] = {}, n_eval: int = 1000) ‑> str`
:   This function prepares LAMMPS understandable force field interactions for coupling simulations and writes them to a file.
    
    Parameters:
     - ff_template (str, optional): Path to the force field template for LAMMPS
     - lammps_ff_path (str, optional): Destination of the external force field file. 
     - potential_kwargs (Dict[str,List[str]]): Dictionary that contains the LAMMPS arguments for every pair style that is used. 
                                               Should contain 'pair_style', 'vdw_style' and 'coulomb_style' key.
     - coupling_potential (Dict[str,Any]): Define the coupling potential that is used. Should contain 'vdw' and 'coulomb' key.
     - coupling_soft_core (Dict[str,float]): Soft core potential parameters.
     - solute_numbers (List[int]): List with unique force field type identifiers for the solute which is coupled/decoupled.
     - combined_lambdas (List[float]): Combined lambdas. Check "coupling" parameter for description.
     - atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
     - nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: vdw_style, coul_style, sigma, epsilon, name, m 
     - bond_numbers_ges (List[int]): List with the unique bond type identifiers used in LAMMPS
     - bonds (List[Dict[str,str]]): List with the unique bonds dictionaries containing
     - angle_numbers_ges (List[int]): List with the unique angle type identifiers used in LAMMPS
     - angles (List[Dict[str,str]]): List with the unique angles dictionaries containing
     - torsion_numbers_ges (List[int]): List with the unique dihedral type identifiers used in LAMMPS
     - torsions (List[Dict[str,str]]): List with the unique dihedral dictionaries containing
     - mixing_rule (str): Provide mixing rule. Defaults to "arithmetic"
     - shake_dict (Dict[str,List[str|int]]): Shake dictionary containing the unique atom,bond,angle identifier for SHAKE algorithm.
                                             Defaults to {"t":[],"b":[],"a":[]}.
     - coupling (bool, optional): If True, coupling is used (lambdas between 0 and 1 are vdW, 1 to 2 are Coulomb). 
                                  For decoupling (lambdas between 0 and 1 are Coulomb, 1 to 2 are vdW). Defaults to "True".
     - precision (int, optional): Precision of coupling lambdas. Defaults to 3.
     - ff_kwargs (Dict[str,Any], optional): Parameter kwargs, which are directly parsed to the template. Defaults to {}.
     - n_eval (int, optional): Number of spline interpolations saved if tabled bond is used. Defaults to 1000.
    
    Returns:
     -  lammps_ff_path (str): Destination of the external force field file.
    
    Raises:
        FileExistsError: If the force field template do not exist.

    
`write_fep_sampling(fep_template: str, fep_outfile: str, combined_lambdas: List[float], charge_list: List[List[int | float]], current_state: int, precision: int = 3, coupling: bool = True, kwargs: Dict[str, Any] = {})`
:   Write FEP sampling.
    
    This function takes in various parameters to generate a free energy sampling file based on a provided FEP sampling template.
    
    Parameters:
    - fep_template (str): The path to the FEP sampling template file.
    - fep_outfile (str): The path to the output file where the generated sampling file will be saved.
    - combined_lambdas (List[float]): A list of combined lambda values.
    - charge_list (List[List[int|float]]): A list of charge lists, where each charge list contains the charges for each atom in the system.
    - current_state (int): The index of the current state in the combined_lambdas list.
    - precision (int, optional): The precision of the lambda values. Defaults to 3.
    - coupling (bool, optional): Whether to use coupling or decoupling lambdas. Defaults to true.
    - kwargs (Dict[str,Any], optional): Additional keyword arguments to be passed to the template rendering. Defaults to {}.
    
    Returns:
    - str: The path to the generated FEP sampling file.
    
    Raises:
    - FileExistsError: If the provided FEP sampling template file does not exist.

    
`write_lammps_ff(ff_template: str, lammps_ff_path: str, potential_kwargs: Dict[str, List[str]], atom_numbers_ges: List[int], nonbonded: List[Dict[str, str]], bond_numbers_ges: List[int], bonds: List[Dict[str, str]], angle_numbers_ges: List[int], angles: List[Dict[str, str]], torsion_numbers_ges: List[int], torsions: List[Dict[str, str]], shake_dict: Dict[str, List[Union[str, int]]] = {'t': [], 'b': [], 'a': []}, only_self_interactions: bool = True, mixing_rule: str = 'arithmetic', ff_kwargs: Dict[str, Any] = {}, n_eval: int = 1000) ‑> str`
:   This function prepares LAMMPS understandable force field interactions and writes them to a file.
    
    Parameters:
     - ff_template (str, optional): Path to the force field template for LAMMPS
     - lammps_ff_path (str, optional): Destination of the external force field file. 
     - potential_kwargs (Dict[str,List[str]]): Dictionary that contains the LAMMPS arguments for every pair style that is used. 
                                               Should contain 'pair_style', 'vdw_style' and 'coulomb_style' key.
     - atom_numbers_ges (List[int]): List with the unique force field type identifiers used in LAMMPS
     - nonbonded (List[Dict[str,str]]): List with the unique force field dictionaries containing: vdw_style, coul_style, sigma, epsilon, name, m 
     - bond_numbers_ges (List[int]): List with the unique bond type identifiers used in LAMMPS
     - bonds (List[Dict[str,str]]): List with the unique bonds dictionaries containing
     - angle_numbers_ges (List[int]): List with the unique angle type identifiers used in LAMMPS
     - angles (List[Dict[str,str]]): List with the unique angles dictionaries containing
     - torsion_numbers_ges (List[int]): List with the unique dihedral type identifiers used in LAMMPS
     - torsions (List[Dict[str,str]]): List with the unique dihedral dictionaries containing
     - shake_dict (Dict[str,List[str|int]]): Shake dictionary containing the unique atom,bond,angle identifier for SHAKE algorithm.
                                             Defaults to {"t":[],"b":[],"a":[]}.
     - only_self_interactions (bool, optional): If only self interactions should be written and LAMMPS does the mixing. Defaults to "True".
     - mixing_rule (str, optional): In case this function should do the mixing, the used mixing rule. Defaultst to "arithmetic"
     - ff_kwargs (Dict[str,Any], optional): Parameter kwargs, which are directly parsed to the template. Defaults to {}.
     - n_eval (int, optional): Number of spline interpolations saved if tabled bond is used. Defaults to 1000.
    
    Returns:
     -  lammps_ff_path (str): Destination of the external force field file.
    
    Raises:
        FileExistsError: If the force field template do not exist.

Classes
-------

`LAMMPS_molecules(mol_str: List[str], force_field_paths: List[str])`
:   This class writes LAMMPS data input for arbitrary mixtures using moleculegraph.

    ### Methods

    `get_shake_indices(self, shake_dict: Dict[str, List[List[str]]] = {'atoms': [], 'bonds': [], 'angles': []})`
    :   Function that get the unique type identifier for atoms, bonds or angles that should be constrained using SHAKE.
        
        Args:
            shake_dict (Dict[str,List[str]], optional): Dictionary with force field keys that should be constrained. Defaults to {"atoms":[],"bonds":[],"angles":[]}.
        
        Returns:
            {"t":[], "b":[], "a":[]}: LAMMPS compatible dict with constraint types per section (t: types, b: bonds, a:angles)

    `prepare_lammps_force_field(self)`
    :   Save force field parameters (atoms, bonds, angles, torsions) used for LAMMPS input.

    `write_lammps_data(self, xyz_path: str, data_template: str, data_path: str, nmol_list: List[int], density: float, box_type: str = 'cubic', z_x_relation: float = 1.0, z_y_relation: float = 1.0)`
    :   Function that generates a LAMMPS data file.
        
        Args:
            xyz_path (str): Path to the xyz file for this system.
            data_template (str): Path to the jinja2 template for the LAMMPS data file.
            data_path (str): Path where the LAMMPS data file should be generated.