import os
import numpy as np
import yaml, toml, json

from typing import Dict, List
from jinja2 import Template
from ..tools.general import unique_by_key, find_key_by_value
from .convertfunctions import (
    convert_harmonic_bond,
    convert_harmonic_angle,
    convert_harmonic_dihedral,
    convert_opls_dihedral,
    source_destination_error,
)

from moleculegraph.molecule_utils import sort_force_fields, sort_graph_key
from .reader import (
    extract_data_file,
    extract_ff_file,
    extract_itp_file,
    extract_top_file,
)

# Define precision of converting
PRECISION = 4

# MAPPING DICTS

BOND_MAP = {
    "1": convert_harmonic_bond,
    "harmonic": convert_harmonic_bond,
}

ANGLE_MAP = {"1": convert_harmonic_angle, "harmonic": convert_harmonic_angle}

DIHEDRAL_MAP = {
    "1": convert_harmonic_dihedral,
    "harmonic": convert_harmonic_dihedral,
    "3": convert_opls_dihedral,
    "opls": convert_opls_dihedral,
}


def convert_atom(source: str, destination: str, atom: Dict[str, str | float | int]):
    new_atom = atom.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        new_atom["epsilon"] = round(
            atom["epsilon"] / 4.184, PRECISION
        )  # convert from kJ to kcal
        new_atom["sigma"] = round(atom["sigma"] * 10, PRECISION)  # convert from nm to A
        new_atom["vdw_style"] = "lj/cut"
        new_atom["coulomb_style"] = "coul/long" if atom["charge"] != 0 else ""
    elif source == "LAMMPS" and destination == "GROMACS":
        new_atom["epsilon"] = round(
            atom["epsilon"] * 4.184, PRECISION
        )  # convert from kcal to kJ
        new_atom["sigma"] = round(atom["sigma"] / 10, PRECISION)  # convert from A to nm
        new_atom.pop("vdw_style")
        new_atom.pop("coulomb_style")
        if "n" in new_atom:
            new_atom.pop("n")
            new_atom.pop("m")
    else:
        source_destination_error(source, destination)

    return new_atom


def convert_bond(source: str, destination: str, bond: Dict[str, str | float | int]):
    new_bond = BOND_MAP[bond["style"]](source, destination, bond)

    return new_bond


def convert_angle(source: str, destination: str, angle: Dict[str, str | float | int]):
    new_angle = ANGLE_MAP[angle["style"]](source, destination, angle)

    return new_angle


def convert_dihedral(
    source: str, destination: str, dihedral: Dict[str, str | float | int]
):
    new_dihedral = DIHEDRAL_MAP[dihedral["style"]](source, destination, dihedral)

    return new_dihedral


# Convert force field from LAMMPS to GROMACS / other way around
def convert_force_field(force_field_path: str, output_path: str):
    if not os.path.exists(force_field_path):
        raise FileExistsError(
            f"Specified force field file does not exist: '{force_field_path}'"
        )

    if ".yaml" in force_field_path:
        data = yaml.safe_load(open(force_field_path))
    elif ".json" in force_field_path:
        data = json.load(open(force_field_path))
    elif ".toml" in force_field_path:
        data = toml.load(open(force_field_path))

    source = data["format"]

    if source == "GROMACS":
        destination = "LAMMPS"
    elif source == "LAMMPS":
        destination = "GROMACS"
    else:
        raise KeyError(f"Specified source is unknown: '{source}'")

    print(f"Convert force field from '{source}' to '{destination}'")

    converted_ff = {
        "format": destination,
        "atoms": {},
        "bonds": {},
        "angles": {},
        "torsions": {},
    }

    for key, atom in data["atoms"].items():
        converted_ff["atoms"][key] = convert_atom(
            source=source, destination=destination, atom=atom
        )

    for key, bond in data["bonds"].items():
        converted_ff["bonds"][key] = convert_bond(
            source=source, destination=destination, bond=bond
        )

    for key, angle in data["angles"].items():
        converted_ff["angles"][key] = convert_angle(
            source=source, destination=destination, angle=angle
        )

    for key, dihedral in data["torsions"].items():
        converted_ff["torsions"][key] = convert_dihedral(
            source=source, destination=destination, dihedral=dihedral
        )

    output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if ".yaml" in output_path:
        yaml.dump(converted_ff, open(output_path, "w"))
    elif ".json" in output_path:
        json.dump(converted_ff, open(output_path, "w"), indent=2)
    elif ".toml" in output_path:
        toml.dump(converted_ff, open(output_path, "w"))


def extract_force_field_gromacs(itp_files: List[str], top_file: str, output_path: str):
    print("Starting to extract GROMACS force field...\n")

    # Define force field dictionary
    force_field = {
        "format": "GROMACS",
        "atoms": {},
        "bonds": {},
        "angles": {},
        "torsions": {},
    }

    # Read in topology
    (
        default_section,
        atom_section,
        bond_section,
        angle_section,
        dihedral_section,
    ) = extract_top_file(top_file)

    # Read in itp_files
    itp_atoms = []
    itp_bonds = []
    itp_angles = []
    itp_dihedrals = []

    for itp_file in itp_files:
        _, itp_atom, itp_bond, itp_angle, itp_dihedral = extract_itp_file(itp_file)
        itp_atoms.extend(itp_atom)
        itp_bonds.extend(itp_bond)
        itp_angles.extend(itp_angle)
        itp_dihedrals.extend(itp_dihedral)

    # Define mapping from atom no to force field key.
    atom_map = {atom[0]: [] for atom in atom_section}
    for atom in itp_atoms:
        atom_map[atom[1]].append(atom[0])

    # Define charge map, in case itp and topology has different charges
    # First use topology charges and then overwrite with itp charges
    charge_map = {atom[0]: float(atom[3]) for atom in atom_section}
    charge_map.update({atom[1]: float(atom[6]) for atom in itp_atoms})

    # If topology has no informaton about bonds, angles, dihedrals, but itp does, add the information.
    if any(itp_bonds) and not bond_section:
        bond_section = []
        for bond in itp_bonds:
            name1, name2, style, *p = bond
            name1 = find_key_by_value(atom_map, name1)
            name2 = find_key_by_value(atom_map, name2)

            bond_section.append([name1, name2, style, *p])

    if any(itp_angles) and not angle_section:
        angle_section = []
        for angle in itp_angles:
            name1, name2, name3, style, *p = angle
            name1 = find_key_by_value(atom_map, name1)
            name2 = find_key_by_value(atom_map, name2)
            name3 = find_key_by_value(atom_map, name3)

            angle_section.append([name1, name2, name3, style, *p])

    if any(itp_dihedrals) and not dihedral_section:
        dihedral_section = []
        for dihedral in itp_dihedrals:
            name1, name2, name3, name4, style, *p = dihedral
            name1 = find_key_by_value(atom_map, name1)
            name2 = find_key_by_value(atom_map, name2)
            name3 = find_key_by_value(atom_map, name3)
            name4 = find_key_by_value(atom_map, name4)

            dihedral_section.append([name1, name2, name3, name4, style, *p])

    # Write information

    for atom in atom_section:
        name, atom_no, mass, _, _, sigma, epsilon = atom
        charge = charge_map[name]
        force_field["atoms"][name] = {
            "name": name,
            "mass": round(float(mass), PRECISION),
            "charge": float(charge),
            "sigma": round(float(sigma), PRECISION),
            "epsilon": round(float(epsilon), PRECISION),
            "atom_no": int(atom_no),
        }

    for bond in bond_section:
        name1, name2, style, *p = bond

        graph_key = sort_graph_key("[" + "][".join([name1, name2]) + "]")
        name_list = sort_force_fields([name1, name2]).tolist()

        if "#" in p:
            p = p[: p.index("#")]

        force_field["bonds"][graph_key] = {
            "list": name_list,
            "p": [round(float(pp), PRECISION) for pp in p],
            "style": style,
        }

    for angle in angle_section:
        name1, name2, name3, style, *p = angle

        graph_key = sort_graph_key("[" + "][".join([name1, name2, name3]) + "]")
        name_list = sort_force_fields([name1, name2, name3]).tolist()

        if "#" in p:
            p = p[: p.index("#")]

        force_field["angles"][graph_key] = {
            "list": name_list,
            "p": [round(float(pp), PRECISION) for pp in p],
            "style": style,
        }

    for dihedral in dihedral_section:
        name1, name2, name3, name4, style, *p = dihedral

        graph_key = sort_graph_key("[" + "][".join([name1, name2, name3, name4]) + "]")
        name_list = sort_force_fields([name1, name2, name3, name4]).tolist()

        if "#" in p:
            p = p[: p.index("#")]

        force_field["torsions"][graph_key] = {
            "list": name_list,
            "p": [round(float(pp), PRECISION) for pp in p],
            "style": style,
        }

    # In case bonds, anlges, or dihedrals are not provided, add dummy entries
    if not force_field["bonds"]:
        force_field["bonds"]["dummy"] = {
            "list": ["dummy", "dummy"],
            "p": [],
            "style": 0,
        }
    if not force_field["angles"]:
        force_field["angles"]["dummy"] = {
            "list": ["dummy", "dummy", "dummy"],
            "p": [],
            "style": 0,
        }
    if not force_field["torsions"]:
        force_field["torsions"]["dummy"] = {
            "list": ["dummy", "dummy", "dummy", "dummy"],
            "p": [],
            "style": 0,
        }

    output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if ".yaml" in output_path:
        yaml.dump(force_field, open(output_path, "w"))
    elif ".json" in output_path:
        json.dump(force_field, open(output_path, "w"), indent=2)
    elif ".toml" in output_path:
        toml.dump(force_field, open(output_path, "w"))

    print("Success!")


class LMP2GRO:
    def __init__(
        self, data_file: str, ff_file: str, map_dict: Dict[str, Dict[str, str]]
    ):
        self.data_file = data_file
        self.ff_file = ff_file
        self.map_dict = map_dict
        self.atom_map = {residue: value["atoms"] for residue, value in map_dict.items()}

    def convert(
        self, itp_template: str, gro_template: str, top_template: str, destination: str
    ):
        # Convert itp files
        self.convert_data_in_itp(itp_template=itp_template, destination=destination)

        # Convert gro files
        self.convert_data_in_gro(gro_template=gro_template, destination=destination)

        # Convert topology
        destination = f'{destination}/{"_".join( self.map_dict.keys() )}.top'
        self.convert_ff_in_top(top_template=top_template, destination=destination)

    def convert_data_in_itp(self, itp_template: str, destination: str):
        self.itp_files = []

        print("Converting LAMMPS data file to .itp files for every residue\n")

        if not os.path.exists(itp_template):
            raise FileExistsError(
                f"Provided itp template does not exist:   '{itp_template}'"
            )

        with open(itp_template) as f:
            template = Template(f.read())

        # Extract relevant data from data file
        (
            _,
            mass_section,
            atom_section,
            bond_section,
            angle_section,
            dihedral_section,
        ) = extract_data_file(self.data_file)

        # Create for every residue a seperate itp file (# resiude name cant be longer than 5 tokens)
        residue_dict = {
            residue[:5]: {"atoms": [], "bonds": [], "angles": [], "dihedrals": []}
            for residue in self.atom_map
        }

        # Extract molecular mass of the atom types
        mass_map = {mass[0]: mass[1] for mass in mass_section}

        # Match the atoms to the resiude
        residue_atom_dict = {
            residue: [atom[0] for atom in atom_section if atom[2] in value]
            for residue, value in self.atom_map.items()
        }

        # Save all itp relevant information
        for atom in atom_section:
            runo, mol, attyp, q, x, y, z, *name = atom
            residue = find_key_by_value(residue_atom_dict, runo)
            gmx_type = self.atom_map[residue][attyp]
            # Provide: nr     type     resnr    residu     atom      cgnr      charge        mass
            residue_dict[residue]["atoms"].append(
                [runo, gmx_type, 1, residue, gmx_type, runo, q, mass_map[attyp]]
            )

        for bond in bond_section:
            runo, bdtyp, a1, a2, *name = bond
            residue = find_key_by_value(residue_atom_dict, a1)
            # If no name is provided parse empty string
            name = name[1:] if len(name) >= 2 else [""]

            # Provide GROMACS with  a1 and a2
            residue_dict[residue]["bonds"].append([a1, a2, "#", *name])

        for angle in angle_section:
            runo, agtyp, a1, a2, a3, *name = angle
            residue = find_key_by_value(residue_atom_dict, a1)
            # If no name is provided parse empty string
            name = name[1:] if len(name) >= 2 else [""]

            # Provide GROMACS with  a1, a2 and a3
            residue_dict[residue]["angles"].append([a1, a2, a3, "#", *name])

        for dihedral in dihedral_section:
            runo, dityp, a1, a2, a3, a4, *name = dihedral
            residue = find_key_by_value(residue_atom_dict, a1)
            # If no name is provided parse empty string
            name = name[1:] if len(name) >= 2 else [""]

            # Provide GROMACS with  a1, a2 and a3
            residue_dict[residue]["dihedrals"].append([a1, a2, a3, a4, "#", *name])

        # Get absolute path
        destination = os.path.abspath(destination)

        os.makedirs(destination, exist_ok=True)

        for residue, value in residue_dict.items():
            rendered = template.render(
                {
                    "residue": residue,
                    "nrexcl": self.map_dict[residue]["nrexcl"],
                    **value,
                }
            )
            with open(f"{destination}/{residue}.itp", "w") as f:
                f.write(rendered)
            self.itp_files.append(f"{destination}/{residue}.itp")
        print("Success!\n")

    def convert_data_in_gro(self, gro_template: str, destination: str):
        print("Converting LAMMPS data file to .gro files for every residue\n")

        if not os.path.exists(gro_template):
            raise FileExistsError(
                f"Provided gro template does not exist:   '{gro_template}'"
            )

        with open(gro_template) as f:
            template = Template(f.read())

        # Create for every residue a seperate gro file (# resiude name cant be longer than 5 tokens)
        residue_dict = {residue[:5]: {"atoms": []} for residue in self.atom_map}

        # Extract relevant data from data file
        box_section, _, atom_section, _, _, _ = extract_data_file(self.data_file)

        # Match the atoms to the residue
        residue_atom_dict = {
            residue: [atom[0] for atom in atom_section if atom[2] in value]
            for residue, value in self.atom_map.items()
        }

        for atom in atom_section:
            runo, mol, attyp, q, x, y, z, *name = atom
            residue = find_key_by_value(residue_atom_dict, runo)
            gmx_type = self.atom_map[residue][attyp]
            # Provide GROMACS with  RESno, RESNAME. attyp, running number, x, y ,z (optional velocities: %8.4f%8.4f%8.4f)
            residue_dict[residue]["atoms"].append(
                "%5d%-5s%5s%5d%8.3f%8.3f%8.3f"
                % (
                    1,
                    residue,
                    gmx_type,
                    int(runo),
                    float(x) / 10,
                    float(y) / 10,
                    float(z) / 10,
                )
            )

        # Get absolute path
        destination = os.path.abspath(destination)

        os.makedirs(destination, exist_ok=True)

        for residue, value in residue_dict.items():
            rendered = template.render(
                {
                    "name": residue,
                    "no_atoms": f"{len(value['atoms']):5d}",
                    "box_dimension": "%10.5f%10.5f%10.5f" % tuple(box_section),
                    **value,
                }
            )
            with open(f"{destination}/{residue}.gro", "w") as f:
                f.write(rendered)

        print("Success!\n")

    def convert_ff_in_top(self, top_template: str, destination: str):
        print("Converting LAMMPS force field file to GROMACS toppology file\n")

        if not os.path.exists(top_template):
            raise FileExistsError(
                f"Provided topology template does not exist:   '{top_template}'"
            )

        with open(top_template) as f:
            template = Template(f.read())

        # Extract relevant data from data file
        (
            _,
            mass_section,
            atom_section,
            bond_section,
            angle_section,
            dihedral_section,
        ) = extract_data_file(self.data_file)

        # Extract molecular mass of the atom types
        mass_map = {mass[0]: mass[1] for mass in mass_section}

        # Get the charge of the atom types
        charge_map = {atom[2]: atom[3] for atom in atom_section}

        # Get the atom types dict for all residues
        type_map = {
            k: v
            for key, value in self.map_dict.items()
            for k, v in value["atoms"].items()
        }

        # Get the atom no for all residues
        no_map = {
            k: v
            for key, value in self.map_dict.items()
            for k, v in value["atom_no"].items()
        }

        # Get bond, angle and torsion keys
        # Map the bonds with their force field type, needed for GROMACS
        atom_map = {key: [] for key in type_map}
        for atom in atom_section:
            atom_map[atom[2]].append(atom[0])

        # Bonds
        unqiue_bonds = unique_by_key(bond_section, 1)
        bond_map = {
            unique_bond[1]: [
                type_map[find_key_by_value(atom_map, a)] for a in unique_bond[2:4]
            ]
            for unique_bond in unqiue_bonds
        }

        # Angles
        unqiue_angles = unique_by_key(angle_section, 1)
        angle_map = {
            unique_angle[1]: [
                type_map[find_key_by_value(atom_map, a)] for a in unique_angle[2:5]
            ]
            for unique_angle in unqiue_angles
        }

        # Dihedrals
        unqiue_dihedrals = unique_by_key(dihedral_section, 1)
        dihedral_map = {
            unique_dihedral[1]: [
                type_map[find_key_by_value(atom_map, a)] for a in unique_dihedral[2:6]
            ]
            for unique_dihedral in unqiue_dihedrals
        }

        # Read out force field
        (
            mixing_rule,
            fudgeLJ,
            fudgeQQ,
            atom_section,
            bond_section,
            angle_section,
            dihedral_section,
        ) = extract_ff_file(self.ff_file)

        # Write topology file
        top_dict = {
            "atoms": [],
            "bonds": [],
            "angles": [],
            "dihedrals": [],
            "comb_rule": mixing_rule,
            "fudgeLJ": fudgeLJ,
            "fudgeQQ": fudgeQQ,
            "system_name": "_".join(self.map_dict.keys()),
            "residue_dict": {v: 1 for v in self.map_dict},
            "itp_files": self.itp_files,
        }

        for atom in atom_section:
            i, sigma, epsilon = atom
            top_dict["atoms"].append(
                [
                    type_map[i],
                    no_map[i],
                    mass_map[i],
                    charge_map[i],
                    "A",
                    round(float(sigma) / 10, PRECISION),
                    round(float(epsilon) * 4.184, PRECISION),
                ]
            )

        for bond in bond_section:
            i, style, p = bond
            bond_dict = {
                "style": style,
                "p": np.array(p).astype(float),
                "list": bond_map[i],
            }
            new_bond = convert_bond("LAMMPS", "GROMACS", bond_dict)
            top_dict["bonds"].append(
                [*new_bond["list"], new_bond["style"], *new_bond["p"]]
            )

        for angle in angle_section:
            i, style, p = angle
            angle_dict = {
                "style": style,
                "p": np.array(p).astype(float),
                "list": angle_map[i],
            }
            new_angle = convert_angle("LAMMPS", "GROMACS", angle_dict)
            top_dict["angles"].append(
                [*new_angle["list"], new_angle["style"], *new_angle["p"]]
            )

        for dihedral in dihedral_section:
            i, style, p = dihedral
            dihedral_dict = {
                "style": style,
                "p": np.array(p).astype(float),
                "list": dihedral_map[i],
            }
            new_dihedral = convert_dihedral("LAMMPS", "GROMACS", dihedral_dict)
            top_dict["dihedrals"].append(
                [*new_dihedral["list"], new_dihedral["style"], *new_dihedral["p"]]
            )

        destination = os.path.abspath(destination)

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with open(destination, "w") as f:
            f.write(template.render(top_dict))

        print("Success!\n")
