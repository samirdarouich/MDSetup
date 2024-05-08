Module pyLMP.tools.playmol_utils
================================

Functions
---------

    
`prepare_playmol_input(mol_str: List[str], ff: Dict[str, Dict[str, float]], playmol_template: str, playmol_ff_path: str)`
:   Function that writes playmol force field using a jinja2 template given a list of moleculegraph strings
    
    Parameters:
        - mol_str (List[str]): List of moleculegraph interpretable strings
        - ff (Dict[str,Dict[str,float]]): Dictinoary with moleculegraph interpretable keys for the force field.
        - playmol_template (str): Path to playmol template for system building.
        - playmol_ff_path (str): Path were the new playmol force field file should be writen to.

    
`write_playmol_input(mol_str: List[str], molecule_numbers: List[int], density: float, nb_all: List[Dict[str, str | float]], playmol_template: str, playmol_path: str, playmol_ff_path: str, xyz_paths: List[str], playmol_execute_template: str, submission_command: str, on_cluster: bool = False)`
:   Function that generates input file for playmol to build the specified system, as well as execute playmol to build the system
    
    Parameters:
     - mol_str (List[str]): List of moleculegraph interpretable strings
     - molecule_numbers (List[int]): List of molecule numbers per component
     - density (float): Estimated density of the system.
     - nb_all (List[Dict[str,str|float]]): List with nonbonded information of every atomtype in the system (not only the unique ones)
     - playmol_template (str): Path to playmol input template.
     - playmol_path (str): Path where the playmol .mol file is writen and executed.
     - playmol_ff_path (str): Path to the playmol force field file.
     - xyz_paths (List[str]): List with the path(s) to the xyz file(s) for each component.
     - playmol_executeable_template (str): Path to bash template to execute playmol.
     - submission_command (str, optional): The command used to submit the job files.
     - on_cluster (bool, optional): If the PLAYMOL build should be submited to the cluster. Defaults to "False".
    
    Returns:
     - xyz_path (str): Path to build coordinate file.