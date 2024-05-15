import os
import json, yaml, toml

from typing import List
from jinja2 import Template
from .molecule import get_forcefield_molecule_from_smiles
from moleculegraph.molecule_utils import sort_force_fields
from ..tools.general import (
    SOFTWARE_LIST,
    SoftwareError,
    KwargsError,
    flatten_list,
    merge_nested_dicts,
    unique_by_key,
)
from .writer import (
    atoms_molecule,
    bonds_molecule,
    angles_molecule,
    dihedrals_molecule,
    atoms_topology,
    bonds_topology,
    angles_topology,
    dihedrals_topology,
    style_topology,
    write_gro_file,
)


class forcefield:
    """
    This class writes a force field input for arbitrary mixtures using moleculegraph. This input can be used for GROMACS and LAMMPS.
    """

    def __init__(
        self, smiles: List[str], force_field_paths: List[str], verbose: bool = False
    ):

        # Check force field format
        self.software = self.ff["format"]

        if not self.software.lower() in SOFTWARE_LIST:
            raise SoftwareError(self.software)
        else:
            print(f"Force field provided for software '{self.software}'")

        # Read in force field files
        self.ff = {}
        for force_field_path in force_field_paths:
            if ".yaml" in force_field_path:
                data = yaml.safe_load(open(force_field_path))
            elif ".json" in force_field_path:
                data = json.load(open(force_field_path))
            elif ".toml" in force_field_path:
                data = toml.load(open(force_field_path))
            else:
                raise KeyError(
                    f"Force field file is not supported: '{force_field_path}'. Please provide 'YAML', 'JSON', or 'TOML' file."
                )
            # Update overall dict
            merge_nested_dicts(self.ff, data.copy())

        # Extract topology smarts for each type
        substructure_smarts = {
            atom["name"]: atom["topology"] for atom in self.ff["atoms"]
        }

        # Extract if united atoms are wanted
        UA_flag = self.ff["UA_flag"]

        # Match from SMILES to force field keys and get moleculegraph representation of the matched molecule
        self.mol_list = [
            get_forcefield_molecule_from_smiles(
                smiles=smile,
                substructure_smarts=substructure_smarts,
                UA_flag=UA_flag,
                verbose=verbose,
            )
            for smile in smiles
        ]

        # Define list for paths to molecule and topology files
        self.molecule_files = []
        self.gro_files = []
        self.topology_file = ""

        ## Map force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions) ##

        # Get (unique) atom types and parameters
        self.nonbonded = unique_by_key(
            flatten_list(
                molecule.map_molecule(molecule.unique_atom_keys, self.ff["atoms"])
                for molecule in self.mol_list
            ),
            "name",
        )

        # Get (unique) bond types and parameters
        self.bonds = unique_by_key(
            flatten_list(
                molecule.map_molecule(molecule.unique_bond_keys, self.ff["bonds"])
                for molecule in self.mol_list
            ),
            "list",
        )

        # Get (unique) angle types and parameters
        self.angles = unique_by_key(
            flatten_list(
                molecule.map_molecule(molecule.unique_angle_keys, self.ff["angles"])
                for molecule in self.mol_list
            ),
            "list",
        )

        # Get (unique) dihedrals types and parameters
        self.dihedrals = unique_by_key(
            flatten_list(
                molecule.map_molecule(
                    molecule.unique_torsion_keys, self.ff["dihedrals"]
                )
                for molecule in self.mol_list
            ),
            "list",
        )

        if not all(
            [
                all(self.nonbonded),
                all(self.bonds),
                all(self.angles),
                all(self.dihedrals),
            ]
        ):
            txt = (
                "nonbonded"
                if not all(self.nonbonded)
                else (
                    "bonds"
                    if not all(self.bonds)
                    else "angles" if not all(self.angles) else "dihedrals"
                )
            )
            raise ValueError(
                "Something went wrong during the force field mapping for key: %s" % txt
            )

        # Define labelmap for numeric numbers and the force field
        labelmap = {}
        labelmap["atoms"] = {
            i + 1: atom_ff["name"] for i, atom_ff in enumerate(self.nonbonded)
        }
        labelmap["bonds"] = {
            i + 1: "_".join(sort_force_fields(bond_ff["list"]))
            for i, bond_ff in enumerate(self.bonds)
        }
        labelmap["angles"] = {
            i + 1: "_".join(sort_force_fields(angle_ff["list"]))
            for i, angle_ff in enumerate(self.angles)
        }
        labelmap["dihedrals"] = {
            i + 1: "_".join(sort_force_fields(dihedral_ff["list"]))
            for i, dihedral_ff in enumerate(self.dihedrals)
        }

    def write_molecule_files(
        self, molecule_template: str, molecule_path: str, residues: List[str], **kwargs
    ):
        """
        Function that generates LAMMPS molecule files for all molecules.

        Parameters:
         - molecule_template (str): Path to the jinja2 template for the molecule file.
         - molecule_path (str): Path where the molecule files should be generated.

        Keyword Args:
         - gro_template (str): Template to write gro file for GROMACS
         - nrexcl (List[int]): A list of integers defining the number of bonds to exclude nonbonded interactions for each molecule for GROMACS.

        Return:
            - mol_files (List[str]): List with absolute paths of created mol files
        """

        if not os.path.exists(molecule_template):
            raise FileExistsError(
                f"Molecule template does not exists:\n   {molecule_template}"
            )

        with open(molecule_template) as file:
            template = Template(file.read())

        os.makedirs(os.path.abspath(molecule_path), exist_ok=True)

        file_suffix = (
            "itp"
            if self.software.lower() == "gromacs"
            else "mol" if self.software.lower() == "LAMMPS" else ""
        )

        renderdict = {**kwargs}

        for m, mol in enumerate(self.mol_list):
            # Update kwargs with current molecule residue
            kwargs.update({"residue": residues[m]})

            # Write gro files for GROMACS
            if self.software.lower() == "gromacs":
                KwargsError(["gro_template", "nrexcl"], kwargs.keys())

                # Add exclusion of nonbonded interactions for each molecule
                kwargs.update({"nrexcl": kwargs["nrecxl"][m]})
                self.gro_files.append(
                    write_gro_file(mol, destination=molecule_path, **kwargs)
                )

            # Update kwargs with current molecule coordinates
            kwargs.update({"coordinates": mol.coordinates})

            # Get molecule information
            molecule_ff = mol.map_molecule(mol.atom_names, self.ff["atoms"])

            renderdict.update(
                atoms_molecule(
                    molecule_ff=molecule_ff, software=self.software, **kwargs
                )
            )

            # Get bond information
            bond_list = mol.bond_list + 1
            bond_names = mol.bond_names

            renderdict.update(
                bonds_molecule(
                    bond_list=bond_list,
                    bond_names=bond_names,
                    software=self.software,
                    **kwargs,
                )
            )

            # Get angle information
            angle_list = mol.angle_list + 1
            angle_names = mol.angle_names

            renderdict.update(
                angles_molecule(
                    angle_list=angle_list,
                    angle_names=angle_names,
                    software=self.software,
                    **kwargs,
                )
            )

            # Get dihedrals information
            dihedral_list = mol.torsion_list + 1
            dihedral_names = mol.torsion_names

            renderdict.update(
                dihedrals_molecule(
                    dihedral_list=dihedral_list,
                    dihedral_names=dihedral_names,
                    software=self.software,
                    **kwargs,
                )
            )

            # Get number of atoms, bonds, angles and dihedrals
            renderdict.update(
                {
                    "atom_no": len(molecule_ff),
                    "bond_no": len(bond_list),
                    "angle_no": len(angle_list),
                    "dihedral_no": len(dihedral_list),
                }
            )

            rendered = template.render(renderdict)

            out_file = f"{molecule_path}/{residues[m]}.{file_suffix}"

            with open(out_file, "w") as fh:
                fh.write(rendered)

            self.molecule_files.append(out_file)

        return

    def write_topology_file(
        self, topology_template: str, topology_path: str, system_name: str, **kwargs
    ):
        """
        Function that generates topology for the system.

        Parameters:
         - topology_template (str): Path to the jinja2 template for the topology file.
         - topology_path (str): Path where the topology file should be generated.
         - system_name (str): Name of the system (topology filed gonna be named like it).

        Keyword Args:
         - rcut (float): Cutoff radius.
         - potential_kwargs (Dict[str,List[str]]): Additional keyword arguments specific to potential types and style for LAMMPS.
         - do_mixing (bool): Flag to determine if mixing rules should be applied for LAMMPS.
         - mixing_rule (str): The mixing rule to be used if do_mixing is True. For LAMMPS and GROMACS.
         - fudgeLJ (str): 1-4 pair vdW scaling for GROMACS.
         - fudgeQQ (str): 1-4 pair Coulomb scaling for GROMACS.
         - residue_dict (Dict[str,int]): Dictionary with residue names and the numbers for GROMACS.
        """

        if not os.path.exists(topology_template):
            raise FileExistsError(
                f"Topology template does not exists:\n   {topology_template}"
            )

        with open(topology_template) as file:
            template = Template(file.read())

        os.makedirs(topology_path, exist_ok=True)

        # Define file name on software and system name
        file_suffix = (
            "top"
            if self.software.lower() == "gromacs"
            else "params" if self.software.lower() == "LAMMPS" else ""
        )
        topology_path = f"{topology_path}/{system_name}.{file_suffix}"

        renderdict = {**kwargs, "system_name": system_name}

        if self.software.lower() == "lammps":
            # Check necessary kwargs
            KwargsError(
                ["potential_kwargs", "rcut", "do_mixing", "mixing_rule"], kwargs.keys()
            )

            # Provide label map
            kwargs["labelmap"] = self.labelmap

            # Get styles
            style_dict, hybrid_dict = style_topology(
                system_ff=self.nonbonded,
                bonds_ff=self.bonds,
                angles_ff=self.angles,
                dihedrals_ff=self.dihedrals,
                **kwargs,
            )
            renderdict["style"] = style_dict
            kwargs.update(hybrid_dict)

        elif self.software.lower() == "gromacs":
            # Check necessary kwargs
            KwargsError(
                ["mixing_rule", "fudgeLJ", "fudgeQQ", "residue_dict"], kwargs.keys()
            )

        renderdict.update(
            atoms_topology(system_ff=self.nonbonded, software=self.software, **kwargs)
        )
        renderdict.update(
            bonds_topology(bonds_ff=self.bonds, software=self.software, **kwargs)
        )
        renderdict.update(
            angles_topology(angles_ff=self.angles, software=self.software, **kwargs)
        )
        renderdict.update(
            dihedrals_topology(
                dihedrals_ff=self.dihedrals, software=self.software, **kwargs
            )
        )

        rendered = template.render(renderdict)

        with open(topology_path, "w") as fh:
            fh.write(rendered)

        self.topology_file = topology_path

        return
