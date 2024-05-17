import os
import subprocess
import numpy as np


from jinja2 import Template
from .general import KwargsError
from typing import List, Dict, Any
from scipy.constants import Avogadro
from .submission import submit_and_wait
from ..forcefield.reader import extract_number_dict_from_mol_files

def get_system_volume(
    molar_masses: List[float],
    molecule_numbers: List[int],
    density: float,
    unit_conversion: float,
    box_type: str = "cubic",
    z_x_relation: float = 1.0,
    z_y_relation: float = 1.0,
):
    """
    Calculate the volume of a system and the dimensions of its bounding box based on molecular masses, numbers and density.

    Parameters:
    - molar_masses (List[List[float]]): A list with the molar masses of each molecule in the system.
    - molecule_numbers (List[int]): A list containing the number of molecules of each type in the system.
    - density (float): The density of the mixture in kg/m^3.
    - unit_conversion (float): Unit conversion from Angstrom to xx.
    - box_type (str, optional): The type of box to calculate dimensions for. Currently, only 'cubic' is implemented.
    - z_x_relation (float, optional): Relation of z to x length. z = z_x_relation*x. Defaults to 1.0.
    - z_y_relation (float, optional): Relation of z to y length. z = z_y_relation*y. Defaults to 1.0.

    Returns:
    - dict: A dictionary with keys 'box_x', 'box_y', and 'box_z', each containing a list with the negative and positive half-lengths of the box in Angstroms.

    Raises:
    - KeyError: If the `box_type` is not 'cubic' or 'orthorhombic', since other box types are not implemented yet.
    """
    # Account for mixture density
    molar_masses = np.array(molar_masses)

    # mole fraction of mixture (== numberfraction)
    x = np.array(molecule_numbers) / np.sum(molecule_numbers)

    # Average molar weight of mixture [g/mol]
    M_avg = np.dot(x, molar_masses)

    # Total mole n = N/NA [mol]
    n = np.sum(molecule_numbers) / Avogadro

    # Total mass m = n*M, convert from g in kg. [kg]
    mass = n * M_avg / 1000

    # Volume = mass / mass_density = kg / kg/m^3, convert from m^3 to A^3. [A^3]
    volume = mass / density * 1e30

    # Compute box lenghts (in Angstrom) using the volume V=m/rho
    if box_type == "cubic":
        # Cubix box: L/2 = V^(1/3) / 2
        boxlen = volume ** (1 / 3) / 2 * unit_conversion

        dimensions = {
            "box_x": [-boxlen, boxlen],
            "box_y": [-boxlen, boxlen],
            "box_z": [-boxlen, boxlen],
        }
        box_type = "block"

    elif box_type == "orthorhombic":
        # Orthorhombic: V = x * y * z, with x = z / z_x_relation and y = z / z_y_relation
        # V = z^3 * 1 / z_x_relation * 1 / z_y_relation
        # z = (V*z_x_relation*z_y_relation)^(1/3)
        z = (volume * z_x_relation * z_y_relation) ** (1 / 3) * unit_conversion
        x = z / z_x_relation
        y = z / z_y_relation

        dimensions = {
            "box_x": [-x / 2, x / 2],
            "box_y": [-y / 2, y / 2],
            "box_z": [-z / 2, z / 2],
        }
        box_type = "block"

    else:
        raise KeyError(
            f"Specified box type '{box_type}' is not implemented yet. Available are: 'cubic', 'orthorhombic'."
        )

    box = { "type": box_type, "dimensions": dimensions }
    return box


def generate_initial_configuration(
    destination_folder: str,
    build_template: str,
    software: str,
    coordinate_paths: List[str],
    molecules_list: List[Dict[str, str | int]],
    box: Dict[str, str|float],
    on_cluster: bool = False,
    initial_system: str = "",
    n_try: int = 10000,
    submission_command: str = "qsub",
    **kwargs,
):
    """
    Generate initial configuration for molecular dynamics simulation with LAMMPS/GROMACS.

    Parameters:
     - destination_folder (str): The destination folder where the initial configurations will be saved.
     - build_template (str): Template for system building.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - coordinate_paths (List[str]): List of paths to coordinate files for each molecule.
     - molecules_list (List[Dict[str, str|int]]): List with dictionaries with numbers and names of the molecules.
     - box (Dict[str,str|float]): Dictionary with "box_type" and "dimensions" as keys. 
     - on_cluster (bool, optional): If the build should be submited to the cluster. Defaults to "False".
     - initial_system (str, optional): Path to initial system, if initial system should be used to add molecules rather than new box. Defaults to "".
     - n_try (int, optional): Number of attempts to insert molecules. Defaults to 10000.
     - submission_command (str, optional): Command to submit jobs for cluster,
     - **kwargs (Any): Arbitrary keyword arguments.

    Keyword Args:
     - build_input_template (str): Template for lammps input file that can build the system for LAMMPS.
     - force_field_file (str): File with force field parameters for LAMMPS.

    Returns:
     - intial_coord (str): Path of inital configuration

    """

    # Create and the output folder of the box
    os.makedirs(destination_folder, exist_ok=True)

    # Check if job template file exists
    if not os.path.isfile(build_template):
        raise FileNotFoundError(f"Build template file { build_template } not found.")
    else:
        with open(build_template) as f:
            template = Template(f.read())

    # Sort out molecules that are zero
    non_zero_coord_mol_no = [
        (coord, value["name"], value["number"])
        for coord, value in zip(coordinate_paths, molecules_list)
        if value["number"] > 0
    ]

    # Define output bash file
    bash_file = f"{destination_folder}/build_box.sh"


    if software == "gromacs":
        suffix = "gro"
    
    elif software == "lammps":

        # Check necessary input kwargs
        KwargsError(["build_input_template","force_field_file"], kwargs.keys())

        if not os.path.isfile(kwargs["build_input_template"]):
            raise FileNotFoundError(f"LAMMPS build template file { kwargs['build_input_template'] } not found.")

        if not os.path.isfile(kwargs["force_field_file"]):
            raise FileNotFoundError(f"LAMMPS force field file { kwargs['force_field_file'] } not found.")
        
        suffix = "data"

        lmp_build_file = f"{destination_folder}/build_box.in"

        kwargs["build_input_file"] = os.path.basename( lmp_build_file )
        kwargs["force_field_file"] = os.path.relpath(
                kwargs["force_field_file"], os.path.dirname(lmp_build_file)
        )

        # Get number of types for atoms, bonds, angles and dihedrals
        mol_files = [ p[0] for p in non_zero_coord_mol_no ]
        kwargs["types_no"] = extract_number_dict_from_mol_files( mol_files, **kwargs )

    # Define output coordinate
    intial_coord = f"{destination_folder}/init_conf.{suffix}"

    # Define template settings
    template_settings = {
        "coord_mol_no": non_zero_coord_mol_no,
        "box": box,
        "initial_system": initial_system,
        "n_try": n_try,
        "folder": os.path.dirname(intial_coord),
        "output_coord": os.path.basename(intial_coord),
        **kwargs,
    }

    # Write LAMMPS input file
    if software == "lammps":

        with open( kwargs["build_input_template"] ) as f:
            template_lmp = Template(f.read())

        with open(lmp_build_file, "w") as f:
            f.write(template_lmp.render(template_settings))

    # Write bash file
    with open(bash_file, "w") as f:
        f.write(template.render(template_settings))

    if on_cluster:
        print("\nSubmit build to cluster and wait untils it is finished.\n")
        submit_and_wait(job_files=[bash_file], submission_command=submission_command)
    else:
        print("\nBuild system locally! Wait until it is finished.\n")
        # Call the bash to build the box. Write output to file.
        with open(f"{destination_folder}/build_output.txt", "w") as f:
            subprocess.run(
                ["bash", f"{destination_folder}/build_box.sh"], stdout=f, stderr=f
            )

    # Check if the system is build
    if not os.path.isfile(intial_coord):
        raise FileNotFoundError(
            f"Something went wrong during the box building! { intial_coord } not found."
        )
    print("Build successful\n")

    return intial_coord


def generate_input_files(
    destination_folder: str,
    input_template: str,
    software: str,
    ensembles: List[str],
    simulation_times: List[float],
    ensemble_definition: Dict[str, Any | Dict[str, str | float]],
    dt: float,
    temperature: float,
    pressure: float,
    off_set: int = 0,
    **kwargs,
):
    """
    Generate input files for simulation pipeline.

    Parameters:
     - destination_folder (str): The destination folder where the input files will be saved. Will be saved under destination_folder/0x_ensebmle/ensemble.input
     - input_template (str): The path to the input template file.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - ensembles (List[str]): A list of ensembles to generate input files for.
     - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
     - ensemble_definition (Dict[str, Any|Dict[str, str|float]]): Dictionary containing the ensemble settings for each ensemble.
     - dt (float): The time step for the simulation.
     - temperature (float): The temperature for the simulation.
     - pressure (float): The pressure for the simulation.
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - **kwargs (Any): Arbitrary keyword arguments.

    Keyword Args:
     - initial_coord (str): Absolute path of LAMMPS data file for LAMMPS.
     - initial_topology (str): Absolute path of LAMMPS force field file for LAMMPS.
     - compressibility (float): Compressibility of the system for GROMACS.
     - init_step (int): Initial step to continue simulation for GROMACS.

    Raises:
     - KeyError: If an invalid ensemble is specified.
     - FileNotFoundError: If any input file does not exists.

    Returns:
     - input_files (List[str]): List with paths of the input files

    """

    # Check if input template file exists
    if not os.path.isfile(input_template):
        raise FileNotFoundError(f"Input template file { input_template } not found.")

    # Open template
    with open(input_template) as f:
        template = Template(f.read())

    if software == "gromacs":
        # Check necessary input kwargs
        KwargsError(["compressibility", "init_step"], kwargs.keys())

        # nano in pico second
        time_conversion = 1e3

        # Define file suffix based on software
        suffix = "mdp"

    elif software == "lammps":
        # Check necessary input kwargs
        KwargsError(["initial_coord", "initial_topology"], kwargs.keys())

        # nano in femto second
        time_conversion = 1e6

        # Define file suffix based on software
        suffix = "input"

        # Check if datafile exists
        if not os.path.isfile(kwargs["initial_coord"]):
            raise FileNotFoundError(f"Data file { kwargs['initial_coord'] } not found.")

        # Check if force field file exists
        if not os.path.isfile(kwargs["initial_topology"]):
            raise FileNotFoundError(
                f"Force field file { kwargs['initial_topology'] } not found."
            )

    # Save ensemble names
    ensemble_names = [
        f"{'0' if (j+off_set) < 10 else ''}{j+off_set}_{step}"
        for j, step in enumerate(ensembles)
    ]

    # Define template dictionary
    renderdict = {**kwargs}

    # Produce input files for simulation pipeline
    input_files = []

    for j, (ensemble, time) in enumerate(zip(ensembles, simulation_times)):
        try:
            ensemble_settings = ensemble_definition[ensemble]
        except:
            raise KeyError(
                f"Wrong ensemple specified: {ensemble}. Valid options are: {', '.join(ensemble_definition.keys())} "
            )

        # Ensemble name
        renderdict["ensemble_name"] = ensemble

        # Output file
        input_out = f"{destination_folder}/{'0' if (j+off_set) < 10 else ''}{j+off_set}_{ensemble}/{ensemble}.{suffix}"

        if software == "gromacs":
            # Add temperature of sim to ensemble settings
            if "t" in ensemble_settings.keys():
                ensemble_settings["t"]["ref_t"] = temperature

            # Add pressure and compressibility to ensemble settings
            if "p" in ensemble_settings.keys():
                ensemble_settings["p"].update(
                    {"ref_p": pressure, "compressibility": kwargs["compressibility"]}
                )

            # Overwrite the ensemble settings
            renderdict["ensemble"] = ensemble_settings

            # Define if restart
            renderdict["restart_flag"] = (
                "no" if ensemble == "em" or ensembles[j - 1] == "em" else "yes"
            )

            # Add extension to first system (if wanted)
            if j == 0 and kwargs["init_step"] > 0:
                renderdict["system"]["init_step"] = kwargs["init_step"]

        elif software == "lammps":
            # Add ensemble variables
            values = []
            for v in ensemble_settings["variables"]:
                if v == "temperature":
                    values.append(temperature)
                elif v == "pressure":
                    values.append(round(pressure / 1.01325, 3))
                else:
                    raise KeyError(
                        f"Variable is not implemented: '{v}'. Currently implemented are 'temperature' or 'pressure'. "
                    )

            # Overwrite the ensemble settings
            renderdict["ensemble"] = {
                "var_val": zip(ensemble_settings["variables"], values),
                "command": ensemble_settings["command"],
            }

            renderdict["force_field_file"] = os.path.relpath(
                kwargs["initial_topology"], os.path.dirname(input_out)
            )

            # If its the first ensemble use provided data path, otherwise use the previous restart file. Hence set restart flag
            if j == 0:
                renderdict["initial_coord"] = os.path.relpath(
                    kwargs["initial_coord"], os.path.dirname(input_out)
                )
                renderdict["restart_flag"] = False
            else:
                renderdict[
                    "initial_coord"
                ] = f"../{ensemble_names[j-1]}/{ensembles[j-1]}.restart"
                renderdict["restart_flag"] = True

        # Simulation time is provided in nano seconds and dt in pico/fico seconds, hence multiply with factor 1e3/1e6
        renderdict["system"]["nsteps"] = (
            int(time_conversion * time / dt) if not ensemble == "em" else int(time)
        )
        renderdict["system"]["dt"] = dt

        # Provide a seed for tempearture generating:
        renderdict["seed"] = np.random.randint(0, 1e5)

        # Create the destination folder
        os.makedirs(os.path.dirname(input_out), exist_ok=True)

        # Render template
        rendered = template.render(**renderdict)

        with open(input_out, "w") as f:
            f.write(rendered)

        input_files.append(input_out)

    return input_files


def generate_job_file(
    destination_folder: str,
    job_template: str,
    software: str,
    input_files: List[str],
    ensembles: List[str],
    job_name: str,
    job_out: str = "job.sh",
    off_set: int = 0,
    **kwargs,
):
    """
    Generate initial job file for a set of simulation ensembles.

    Parameters:
     - destination_folder (str): Path to the destination folder where the job file will be created.
     - job_template (str): Path to the job template file.
     - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
     - input_files (List[List[str]]): List of lists containing the paths to the input files for each simulation phase.
     - ensembles (List[str], optional): List of simulation ensembles.
     - job_name (str): Name of the job.
     - job_out (str, optional): Name of the job file. Defaults to "job.sh".
     - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.
     - **kwargs (Any): Arbitrary keyword arguments.

    Keyword Args:
     - initial_topology (str): Path to the initial topology file for GROMACS.
     - intial_coord (str): Path to the initial coordinate file for GROMACS.
     - initial_cpt (str): Path to the inital checkpoint file for GROMACS.
     - init_step (int): Initial step to continue simulation for GROMACS.

    Returns:
     - job_file (str): Path of job file

    Raises:
     - FileNotFoundError: If the job template file does not exist.
     - FileNotFoundError: If any of the input files does not exist.
     - FileNotFoundError: If the initial coordinate file does not exist.
     - FileNotFoundError: If the initial topology file does not exist.
     - FileNotFoundError: If the initial checkpoint file does not exist.
    """

    # Check if job template file exists
    if not os.path.isfile(job_template):
        raise FileNotFoundError(f"Job template file { job_template } not found.")

    # Check for input files
    for file in input_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Input file { file  } not found.")

    # Read in template
    with open(job_template) as f:
        template = Template(f.read())

    # Define folders
    job_file_settings = {
        "ensembles": {
            f"{'0' if (j+off_set) < 10 else ''}{j+off_set}_{step}": {}
            for j, step in enumerate(ensembles)
        },
        **kwargs,
    }
    ensemble_names = list(job_file_settings["ensembles"].keys())

    # Create the simulation folder
    os.makedirs(destination_folder, exist_ok=True)

    # Define LOG output
    log_path = f"{destination_folder}/LOG"

    # Add to job file settings
    job_file_settings.update(
        {"job_name": job_name, "log_path": log_path, "working_path": destination_folder}
    )

    if software == "gromacs":
        # Check necessary input kwargs
        KwargsError(
            ["initial_topology", "intial_coord", "initial_cpt", "init_step"],
            kwargs.keys(),
        )

        # Check if topology file exists
        if not os.path.isfile(["initial_topology"]):
            raise FileNotFoundError(
                f"Topology file { kwargs['initial_topology'] } not found."
            )

        # Check if coordinate file exists
        if not os.path.isfile(kwargs["intial_coord"]):
            raise FileNotFoundError(
                f"Coordinate file { kwargs['intial_coord'] } not found."
            )

        # Check if checkpoint file exists
        if kwargs["initial_cpt"] and not os.path.isfile(kwargs["initial_cpt"]):
            raise FileNotFoundError(
                f"Checkpoint file { kwargs['initial_cpt'] } not found."
            )

        # Relative paths for each mdp file for each simulation phase
        mdp_relative = [
            os.path.relpath(input_files[j], f"{destination_folder}/{step}")
            for j, step in enumerate(ensemble_names)
        ]

        # Relative paths for each coordinate file (for energy minimization use initial coodinates, otherwise use the preceeding output)
        cord_relative = [
            f"../{ensemble_names[j-1]}/{ensembles[j-1]}.gro"
            if j > 0
            else os.path.relpath(kwargs["intial_coord"], f"{destination_folder}/{step}")
            for j, step in enumerate(job_file_settings["ensembles"].keys())
        ]

        # Relative paths for each checkpoint file
        cpt_relative = [
            f"../{ensemble_names[j-1]}/{ensembles[j-1]}.cpt"
            if j > 0
            else os.path.relpath(kwargs["initial_cpt"], f"{destination_folder}/{step}")
            if kwargs["initial_cpt"] and not ensembles[j] == "em"
            else ""
            for j, step in enumerate(ensemble_names)
        ]

        # Relative paths for topology
        topo_relative = [
            os.path.relpath(kwargs["initial_topology"], f"{destination_folder}/{step}")
            for j, step in enumerate(ensemble_names)
        ]

        # output file
        out_relative = [f"{step}.tpr -maxwarn 10" for step in ensembles]

        for j, step in enumerate(ensemble_names):
            # If first or preceeding step is energy minimization, or if there is no cpt file to read in
            if ensembles[j - 1] == "em" or ensembles[j] == "em" or not cpt_relative[j]:
                job_file_settings["ensembles"][step][
                    "grompp"
                ] = f"grompp -f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -o {out_relative[j]}"
            else:
                job_file_settings["ensembles"][step][
                    "grompp"
                ] = f"grompp -f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -t {cpt_relative[j]} -o {out_relative[j]}"

            # Define mdrun command
            if j == 0 and kwargs["initial_cpt"] and kwargs["init_step"] > 0:
                # In case extension of the first simulation in the pipeline is wanted
                job_file_settings["ensembles"][step][
                    "grompp"
                ] = f"grompp -f {mdp_relative[j]} -c {ensembles[j]}.gro -p {topo_relative[j]} -t {ensembles[j]}.cpt -o {out_relative[j]}"
                job_file_settings["ensembles"][step][
                    "mdrun"
                ] = f"mdrun -deffnm {ensembles[j]} -cpi {ensembles[j]}.cpt"
            else:
                job_file_settings["ensembles"][step][
                    "mdrun"
                ] = f"mdrun -deffnm {ensembles[j]}"

    elif software == "lammps":
        # Relative paths for each input file for each simulation ensemble
        for j, step in enumerate(ensemble_names):
            job_file_settings["ensembles"][step]["mdrun"] = os.path.relpath(
                input_files[j], f"{destination_folder}/{step}"
            )

    # Write template
    job_file = f"{destination_folder}/{job_out}"
    os.makedirs(os.path.dirname(job_file), exist_ok=True)

    # Write new file
    rendered = template.render(**job_file_settings)

    with open(job_file, "w") as f:
        f.write(rendered)

    return job_file
