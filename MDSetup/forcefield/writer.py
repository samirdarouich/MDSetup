import os
from jinja2 import Template
from typing import List, Dict
from .utils import get_mixed_parameters, get_pair_style
from moleculegraph.molecule_utils import sort_force_fields
from ..tools.general import SOFTWARE_LIST, SoftwareError, KwargsError


################ Functions for "molecule" files ################


def atoms_molecule(molecule_ff: List[Dict[str, float | str]], software: str, **kwargs):
    """
    Generates a dictionary of atom properties formatted for specified molecular simulation software.

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
    """
    if software == "gromacs":
        atoms_dict = {"atoms": []}

        # Check necessary kwargs
        KwargsError(["residue"], kwargs)

        for iatom, ffatom in enumerate(molecule_ff):
            atoms_dict["atoms"].append(
                [
                    iatom + 1,
                    ffatom["name"],
                    1,
                    kwargs["residue"],
                    ffatom["name"],
                    iatom + 1,
                    ffatom["charge"],
                    ffatom["mass"],
                ]
            )

    elif software == "lammps":
        atoms_dict = {"coords": [], "types": [], "charges": []}

        # Check necessary kwargs
        KwargsError(["coordinates"], kwargs)

        for iatom, ffatom in enumerate(molecule_ff):
            atoms_dict["types"].append([iatom + 1, ffatom["name"]])
            atoms_dict["charges"].append([iatom + 1, ffatom["charge"]])
            atoms_dict["coords"].append([iatom + 1, *kwargs["coordinates"][iatom]])
    else:
        raise SoftwareError(software)

    return atoms_dict


def bonds_molecule(
    bond_list: List[List[int]], bond_names: List[List[str]], software: str, **kwargs
):
    """
    Generates a dictionary of bonds formatted based on the specified software.

    Parameters:
    - bond_list (List[List[int]]): A list of lists, each containing two integers representing a bond.
    - bond_names (List[List[str]]): A list of lists, each containing strings representing the names of the bonds.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary with a key "bonds" that has a list of formatted bond information.

    Raises:
        SoftwareError: If the specified software is not supported.
    """

    bonds_dict = {"bonds": []}

    if software == "gromacs":
        for ibond, (bl, bn) in enumerate(zip(bond_list, bond_names)):
            bonds_dict["bonds"].append([*bl, "#", " ".join(bn)])

    elif software == "lammps":
        for ibond, (bl, bn) in enumerate(zip(bond_list, bond_names)):
            bonds_dict["bonds"].append([ibond + 1, "_".join(bn), *bl])
    else:
        raise SoftwareError(software)

    return bonds_dict


def angles_molecule(
    angle_list: List[List[int]], angle_names: List[List[str]], software: str, **kwargs
):
    """
    Generates a dictionary of angles formatted based on the specified software.

    Parameters:
    - angle_list (List[List[int]]): A list of lists, each containing three integers representing a angle.
    - angle_names (List[List[str]]): A list of lists, each containing strings representing the names of the angles.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary with a key "angles" that has a list of formatted bond information.

    Raises:
        SoftwareError: If the specified software is not supported.
    """
    angles_dict = {"angles": []}

    if software == "gromacs":
        for iangle, (al, an) in enumerate(zip(angle_list, angle_names)):
            angles_dict["angles"].append([*al, "#", " ".join(an)])

    elif software == "lammps":
        for iangle, (al, an) in enumerate(zip(angle_list, angle_names)):
            angles_dict["angles"].append([iangle + 1, "_".join(an), *al])

    else:
        raise SoftwareError(software)

    return angles_dict


def dihedrals_molecule(
    dihedral_list: List[List[int]],
    dihedral_names: List[List[str]],
    software: str,
    **kwargs,
):
    """
    Generates a dictionary of dihedrals formatted based on the specified software.

    Parameters:
    - dihedral_list (List[List[int]]): A list of lists, each containing four integers representing a dihedral.
    - dihedral_names (List[List[str]]): A list of lists, each containing strings representing the names of the dihedrals.
    - software (str): The simulation software to format the output for ('gromacs' or 'lammps').
    - **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary with a key "dihedrals" that has a list of formatted bond information.

    Raises:
        SoftwareError: If the specified software is not supported.
    """
    dihedrals_dict = {"dihedrals": []}
    if software == "gromacs":
        for idihedral, (dl, dn) in enumerate(zip(dihedral_list, dihedral_names)):
            dihedrals_dict["dihedrals"].append([*dl, "#", " ".join(dn)])

    elif software == "lammps":
        for idihedral, (dl, dn) in enumerate(zip(dihedral_list, dihedral_names)):
            dihedrals_dict["dihedrals"].append([idihedral + 1, "_".join(dn), *dl])

    else:
        raise SoftwareError(software)

    return dihedrals_dict


def write_gro_file(
    molecule, gro_template: str, destination: str, residue: str, **kwargs
):
    if not os.path.exists(gro_template):
        raise FileExistsError(f"Gro file template does not exists:\n   {gro_template}")

    with open(gro_template) as file:
        template = Template(file.read())

    # Make path absolute
    destination = os.path.abspath(destination)

    os.makedirs(destination, exist_ok=True)

    coord_destination = f"{destination}/{residue}.gro"

    gromacs = {
        "name": residue,
        "no_atoms": f"{molecule.atom_number:5d}",
        "atoms": [],
        "box_dimension": "%10.5f%10.5f%10.5f" % (0.0, 0.0, 0.0),
    }

    # Gromacs guesses the atomic radii for the insert based on the name in the gro file, hence it makes sense to name the atoms with their element

    # Xyz coordinates are given in Angstrom, convert in nm
    # Provide GROMACS with  RESno, RESNAME. attyp, running number, x, y ,z (optional velocities)
    gromacs["atoms"] = [
        "%5d%-5s%5s%5d%8.3f%8.3f%8.3f # %s"
        % (
            1,
            residue[:5],
            atsym[:5],
            j + 1,
            float(x) / 10,
            float(y) / 10,
            float(z) / 10,
            attyp,
        )
        for j, (attyp, atsym, (x, y, z)) in enumerate(
            zip(molecule.atom_names, molecule.atomsymbols, molecule.coordinate)
        )
    ]

    rendered = template.render(gromacs)

    with open(coord_destination, "w") as fh:
        fh.write(rendered)

    return coord_destination


################ Functions for "topology" file ################


def style_topology(
    system_ff: List[Dict[str, float | str]],
    bonds_ff: List[Dict[str, float | str]],
    angles_ff: List[Dict[str, float | str]],
    dihedrals_ff: List[Dict[str, float | str]],
    **kwargs,
):
    """
    This function takes in several parameters and returns two dictionaries representing the pair styles and hybrid flags for a molecular simulation.

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

    """

    # Check necessary kwargs
    KwargsError(["potential_kwargs", "rcut"], kwargs)

    style_dict = {}
    hybrid_dict = {}

    # Define cut off radius
    rcut = kwargs["rcut"]

    # Get pair style defined in force field
    local_variables = locals()

    # Get pair style defined in force field
    vdw_pair_styles = list({p["vdw_style"] for p in system_ff if p["vdw_style"]})
    coul_pair_styles = list(
        {p["coulomb_style"] for p in system_ff if p["coulomb_style"]}
    )

    pair_style = get_pair_style(
        local_variables,
        vdw_pair_styles,
        coul_pair_styles,
        kwargs["potential_kwargs"]["pair_style"],
    )
    pair_hybrid_flag = "hybrid" in pair_style

    # Fix n_eval
    n_eval = 1000

    # Bond styles
    bonds_styles = list(
        {
            a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"]
            for a in bonds_ff
        }
    )
    bonds_hybrid_flag = len(bonds_styles) > 1

    # Angle styles
    angles_styles = list(
        {
            a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"]
            for a in angles_ff
        }
    )
    angles_hybrid_flag = len(angles_styles) > 1

    # Dihedral styles
    dihedrals_styles = list(
        {
            a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"]
            for a in dihedrals_ff
        }
    )
    dihedrals_hybrid_flag = len(dihedrals_styles) > 1

    style_dict["pair_style"] = pair_style
    style_dict["bonds_style"] = bonds_styles
    style_dict["angles_style"] = angles_styles
    style_dict["dihedrals_style"] = dihedrals_styles

    hybrid_dict["pair_hybrid_flag"] = pair_hybrid_flag
    hybrid_dict["bonds_hybrid_flag"] = bonds_hybrid_flag
    hybrid_dict["angles_hybrid_flag"] = angles_hybrid_flag
    hybrid_dict["dihedrals_hybrid_flag"] = dihedrals_hybrid_flag

    return style_dict, hybrid_dict


def atoms_topology(system_ff: List[Dict[str, float | str]], software: str, **kwargs):
    """
    Generates a dictionary representing the topology of atoms in a system for different simulation software.

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

    """

    if software == "gromacs":
        atoms_dict = {"atoms": []}
    elif software == "lammps":
        # Check necessary kwargs
        KwargsError(
            ["do_mixing", "mixing_rule", "pair_hybrid_flag", "potential_kwargs"],
            kwargs,
        )

        atoms_dict = {"atoms": {"vdw": [], "coulomb": [], "masses": []}}

        # Add masses
        atoms_dict["atoms"]["masses"] = [
            [ffiatom["name"], ffiatom["mass"]] for ffiatom in system_ff
        ]

        # Add coulomb style if there is any.
        if len({p["coulomb_style"] for p in system_ff if p["coulomb_style"]}) == 1:
            atoms_dict["atoms"]["coulomb"] = [
                [
                    "*",
                    "*",
                    *{p["coulomb_style"] for p in system_ff if p["coulomb_style"]},
                ]
            ]
        elif len({p["coulomb_style"] for p in system_ff if p["coulomb_style"]}) > 1:
            atoms_dict["atoms"]["coulomb"] = [
                [
                    i,
                    i,
                    ffiatom["coulomb_style"],
                ]
                for i, ffiatom in enumerate(system_ff)
                if ffiatom["coulomb_style"]
            ]

    else:
        raise SoftwareError(software)

    for i, ffiatom in enumerate(system_ff):
        if software == "gromacs":
            atoms_dict["atoms"].append(
                [
                    ffiatom["name"],
                    ffiatom["atom_no"],
                    ffiatom["mass"],
                    ffiatom["charge"],
                    "A",
                    ffiatom["sigma"],
                    ffiatom["epsilon"],
                ]
            )
        elif software == "lammps":
            for ffjatom in system_ff[i:]:
                if not kwargs["do_mixing"] and ffiatom["name"] != ffjatom["name"]:
                    continue
                else:
                    sigma_ij, epsilon_ij, n_ij, m_ij = get_mixed_parameters(
                        ffiatom=ffiatom,
                        ffjatom=ffjatom,
                        mixing_rule=kwargs["mixing_rule"],
                        precision=4,
                    )

                # Get local variables and map them with potential kwargs
                local_variables = locals()

                atoms_dict["atoms"]["vdw"].append(
                    [ffiatom["name"], ffjatom["name"]]
                    + ([ffiatom["vdw_style"]] if kwargs["pair_hybrid_flag"] else [])
                    + [
                        local_variables[arg]
                        for arg in kwargs["potential_kwargs"]["vdw_style"][
                            ffiatom["vdw_style"]
                        ]
                    ]
                )

    return atoms_dict


def bonds_topology(bonds_ff: List[Dict[str, float | str]], software: str, **kwargs):
    """
    Generates a dictionary representing the topology of bonds based on the specified software and additional parameters.

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
    """

    bonds_dict = {"bonds": []}

    if software == "gromacs":
        for bond_ff in bonds_ff:
            bonds_dict["bonds"].append(
                [*sort_force_fields(bond_ff["list"]), bond_ff["style"], *bond_ff["p"]]
            )

    elif software == "lammps":
        # Check necessary kwargs
        KwargsError(["bonds_hybrid_flag"], kwargs)

        for bond_ff in bonds_ff:
            bonds_dict["bonds"].append(
                ["_".join(sort_force_fields(bond_ff["list"]))]
                + ([bond_ff["style"]] if kwargs["bonds_hybrid_flag"] else [])
                + bond_ff["p"]
            )
    else:
        raise SoftwareError(software)

    return bonds_dict


def angles_topology(angles_ff: List[Dict[str, float | str]], software: str, **kwargs):
    """
    Generates a dictionary representing the topology of angles based on the specified software and additional parameters.

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
    """
    angles_dict = {"angles": []}

    if software == "gromacs":
        for angle_ff in angles_ff:
            angles_dict["angles"].append(
                [
                    *sort_force_fields(angle_ff["list"]),
                    angle_ff["style"],
                    *angle_ff["p"],
                ]
            )

    elif software == "lammps":
        # Check necessary kwargs
        KwargsError(["angles_hybrid_flag"], kwargs)

        for angle_ff in angles_ff:
            angles_dict["angles"].append(
                ["_".join(sort_force_fields(angle_ff["list"]))]
                + ([angle_ff["style"]] if kwargs["angles_hybrid_flag"] else [])
                + angle_ff["p"]
            )
    else:
        raise SoftwareError(software)

    return angles_dict


def dihedrals_topology(
    dihedrals_ff: List[Dict[str, float | str]], software: str, **kwargs
):
    """
    Generates a dictionary representing the topology of dihedrals based on the specified software and additional parameters.

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
    """

    dihedrals_dict = {"dihedrals": []}

    if software == "gromacs":
        for dihedral_ff in dihedrals_ff:
            dihedrals_dict["dihedrals"].append(
                [
                    *sort_force_fields(dihedral_ff["list"]),
                    dihedral_ff["style"],
                    *dihedral_ff["p"],
                ]
            )

    elif software == "lammps":
        # Check necessary kwargs
        KwargsError(["dihedrals_hybrid_flag"], kwargs)

        for dihedral_ff in dihedrals_ff:
            dihedrals_dict["dihedrals"].append(
                ["_".join(sort_force_fields(dihedral_ff["list"]))]
                + ([dihedral_ff["style"]] if kwargs["dihedrals_hybrid_flag"] else [])
                + dihedral_ff["p"]
            )
    else:
        raise SoftwareError(software)

    return dihedrals_dict
