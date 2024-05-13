Module MDSetup.forcefield.writer
================================

Functions
---------

    
`angles_molecule(angle_list: List[List[int]], angle_names: List[List[str]], software: str, **kwargs)`
:   Generates a dictionary of angles formatted based on the specified software.
    
    Parameters:
    - angle_list (List[List[int]]): A list of lists, each containing three integers representing a angle.
    - angle_names (List[List[str]]): A list of lists, each containing strings representing the names of the angles.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.
    
    Returns:
        dict: A dictionary with a key "angles" that has a list of formatted bond information.
    
    Raises:
        SoftwareError: If the specified software is not supported.

    
`angles_topology(angles_ff: List[Dict[str, str | float]], software: str, **kwargs)`
:   Generates a dictionary representing the topology of angles based on the specified software and additional parameters.
    
    Parameters:
    - angles_ff (List[Dict[str, Union[float, str]]]): A list of dictionaries where each dictionary contains details of a angle force field. 
                                                       Each dictionary must have keys 'list', 'style', and 'p'.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Additional keyword arguments.
    
    Keyword Args:
    - angles_hybrid_flag (bool): If several angle styles are used for LAMMPS.
    
    Returns:
        Dict[str, List]: A dictionary with a single key 'angles' that maps to a list of angle configurations. Each configuration is a list that may vary depending on the software used.
    
    Raises:
        SoftwareError: If the specified software is not supported.
        KwargsError: If required keyword arguments are missing when using 'lammps'.

    
`atoms_molecule(molecule_ff: List[Dict[str, str | float]], software: str, **kwargs)`
:   Generates a dictionary of atom properties formatted for specified molecular simulation software.
    
    Parameters:
     - molecule_ff (List[Dict[str, Union[float, str]]]): List of dictionaries, each representing an atom.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - **kwargs: Additional keyword arguments required by specific software formats.
    
    Keyword Args:
    - residue (str): Specifies the residue name in GROMACS.
    - coordinates (List[List[float]]): A list of coordinates corresponding to each atom.
       
    Returns:
     - dict: A dictionary containing atom data formatted according to the specified software's requirements.
    
    Raises:
        SoftwareError: If the specified software is not supported.
        KwargsError: If required keyword arguments are missing for the specified software.

    
`atoms_topology(system_ff: List[Dict[str, str | float]], software: str, **kwargs)`
:   Generates a dictionary representing the topology of atoms in a system for different simulation software.
    
    Parameters:
     - system_ff (List[Dict[str, float | str]]): List of dictionaries containing force field information for each atom.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    
    Keyword Args:
     - do_mixing (bool): Flag to determine if mixing rules should be applied for LAMMPS.
     - mixing_rule (str): The mixing rule to be used if do_mixing is True.
     - pair_hybrid_flag (bool): Flag to indicate if hybrid pair styles are used in LAMMPS.
     - potential_kwargs (dict): Additional keyword arguments specific to potential types and styles.
    
    Returns:
     - dict: A dictionary with keys 'atoms' and optionally 'coulomb', containing lists of atom and interaction information.
    
    Raises:
     - SoftwareError: If the specified software is not supported.
     - KwargsError: If required keyword arguments for LAMMPS are missing.

    
`bonds_molecule(bond_list: List[List[int]], bond_names: List[List[str]], software: str, **kwargs)`
:   Generates a dictionary of bonds formatted based on the specified software.
    
    Parameters:
    - bond_list (List[List[int]]): A list of lists, each containing two integers representing a bond.
    - bond_names (List[List[str]]): A list of lists, each containing strings representing the names of the bonds.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.
    
    Returns:
        dict: A dictionary with a key "bonds" that has a list of formatted bond information.
    
    Raises:
        SoftwareError: If the specified software is not supported.

    
`bonds_topology(bonds_ff: List[Dict[str, str | float]], software: str, **kwargs)`
:   Generates a dictionary representing the topology of bonds based on the specified software and additional parameters.
    
    Parameters:
    - bonds_ff (List[Dict[str, Union[float, str]]]): A list of dictionaries where each dictionary contains details of a bond force field. 
                                                       Each dictionary must have keys 'list', 'style', and 'p'.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Additional keyword arguments.
    
    Keyword Args:
    - bonds_hybrid_flag (bool): If several bond styles are used for LAMMPS.
    
    Returns:
        Dict[str, List]: A dictionary with a single key 'bonds' that maps to a list of bond configurations. Each configuration is a list that may vary depending on the software used.
    
    Raises:
        SoftwareError: If the specified software is not supported.
        KwargsError: If required keyword arguments are missing when using 'lammps'.

    
`dihedrals_molecule(dihedral_list: List[List[int]], dihedral_names: List[List[str]], software: str, **kwargs)`
:   Generates a dictionary of dihedrals formatted based on the specified software.
    
    Parameters:
    - dihedral_list (List[List[int]]): A list of lists, each containing four integers representing a dihedral.
    - dihedral_names (List[List[str]]): A list of lists, each containing strings representing the names of the dihedrals.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.
    
    Returns:
        dict: A dictionary with a key "dihedrals" that has a list of formatted bond information.
    
    Raises:
        SoftwareError: If the specified software is not supported.

    
`dihedrals_topology(dihedrals_ff: List[Dict[str, str | float]], software: str, **kwargs)`
:   Generates a dictionary representing the topology of dihedrals based on the specified software and additional parameters.
    
    Parameters:
    - dihedrals_ff (List[Dict[str, Union[float, str]]]): A list of dictionaries where each dictionary contains details of a dihedral force field. 
                                                       Each dictionary must have keys 'list', 'style', and 'p'.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Additional keyword arguments.
    
    Keyword Args:
    - dihedrals_hybrid_flag (bool): If several dihedral styles are used for LAMMPS.
    
    Returns:
        Dict[str, List]: A dictionary with a single key 'dihedrals' that maps to a list of dihedral configurations. Each configuration is a list that may vary depending on the software used.
    
    Raises:
        SoftwareError: If the specified software is not supported.
        KwargsError: If required keyword arguments are missing when using 'lammps'.

    
`style_topology(system_ff: List[Dict[str, str | float]], bonds_ff: List[Dict[str, str | float]], angles_ff: List[Dict[str, str | float]], dihedrals_ff: List[Dict[str, str | float]], **kwargs)`
:   This function takes in several parameters and returns two dictionaries representing the pair styles and hybrid flags for a molecular simulation.
    
    Parameters:
    - system_ff (List[Dict[str,float|str]]): A list of dictionaries representing the force field parameters for the system.
    - bonds_ff (List[Dict[str,float|str]]): A list of dictionaries representing the force field parameters for the bonds.
    - angles_ff (List[Dict[str,float|str]]): A list of dictionaries representing the force field parameters for the angles.
    - dihedrals_ff (List[Dict[str,float|str]]): A list of dictionaries representing the force field parameters for the dihedrals.
    - **kwargs: Additional keyword arguments.
        
    Keyword Args:
    - rcut (float): Cutoff radius.
    - potential_kwargs (Dict[str,List[str]]): Additional keyword arguments specific to potential types and styles.
    
    
    Returns:
    - style_dict (Dict[str,Any]): A dictionary containing the pair styles and other force field styles.
    - hybrid_dict (Dict[str,bool]): A dictionary containing the hybrid flags for the pair styles and other force field styles.

    
`write_gro_file(molecule, gro_template: str, destination: str, residue: str, **kwargs)`
: