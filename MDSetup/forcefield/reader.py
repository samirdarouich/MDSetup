from ..tools.general import flatten_list


def represents_number(s):
    try:
        float(s)
        return False
    except ValueError:
        return True


## Read out LAMMPS


def extract_data_file(data_file: str):
    with open(data_file) as f:
        lines = f.readlines()

    # Convert from angstrom in nm
    box_section = []
    for line in lines[: lines.index("Masses\n")]:
        if "xlo xhi" in line:
            xlo, xhi, _, _ = line.split()
            box_section.append((float(xhi) - float(xlo)) / 10)
        if "ylo yhi" in line:
            ylo, yhi, _, _ = line.split()
            box_section.append((float(yhi) - float(ylo)) / 10)
        if "zlo zhi" in line:
            zlo, zhi, _, _ = line.split()
            box_section.append((float(zhi) - float(zlo)) / 10)
            break

    mass_section = [
        line.split()
        for line in lines[lines.index("Masses\n") + 1 : lines.index("Atoms\n")]
        if line.strip() and not line.startswith("#")
    ]
    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    for line in lines:
        if in_atom_section:
            if line.startswith("Bonds"):
                in_atom_section = False
                in_bond_section = True
            elif line.strip() and not line.startswith("#"):
                atom_section.append(line.split())
        elif in_bond_section:
            if line.startswith("Angles"):
                in_bond_section = False
                in_angle_section = True
            elif line.strip() and not line.startswith("#"):
                bond_section.append(line.split())
        elif in_angle_section:
            if line.startswith("Dihedrals"):
                in_angle_section = False
                in_dihedral_section = True
            elif line.strip() and not line.startswith("#"):
                angle_section.append(line.split())
        elif in_dihedral_section:
            if line.strip() and not line.startswith("#"):
                dihedral_section.append(line.split())
        elif line.startswith("Atoms"):
            in_atom_section = True

    return (
        box_section,
        mass_section,
        atom_section,
        bond_section,
        angle_section,
        dihedral_section,
    )


def extract_ff_file(ff_file: str):
    with open(ff_file) as f:
        lines = f.readlines()

    # Extract mixing rule and scaling of 1-4 interactions
    mixing = [line for line in lines if "pair_modify" in line and "mix" in line]

    # Default LAMMPS mixing is geometric, this is combination rule n°3 in GROMACS
    if not mixing:
        mixing = "3"
    else:
        if "arithmetic" in mixing[-1]:
            mixing = "2"
        elif "geometric" in mixing[-1]:
            mixing = "3"
        else:
            raise KeyError("Mixing rule sixthpower is not available in GROMACS")

    special_bonds = [line for line in lines if "special_bonds" in line]

    # Default LAMMPS uses 0 scaling for 1-4 pair interactions
    if not special_bonds:
        fudgeLJ, fudgeQQ = "0.0", "0.0"
    else:
        if "lj/coul" in special_bonds[-1]:
            idx = special_bonds[-1].split().index("lj/coul")
            fudgeLJ, fudgeQQ = special_bonds[-1].split()[idx + 3] * 2
        elif "lj" in special_bonds[-1] and "coul" in special_bonds[-1]:
            idx = special_bonds[-1].split().index("lj")
            fudgeLJ = special_bonds[-1].split()[idx + 3]
            idx = special_bonds[-1].split().index("coul")
            fudgeQQ = special_bonds[-1].split()[idx + 3]
        else:
            fudgeLJ, fudgeQQ = "0.0", "0.0"

    # Check which format is provided (in input or data format)
    style_flag = any("pair_coeff" in line for line in lines)

    # Filter out numbers from the styles (cut offs etc)
    atom_styles = flatten_list(
        [line.split()[1:] for line in lines if line.startswith("pair_style")],
        represents_number,
    )
    bond_styles = flatten_list(
        [line.split()[1:] for line in lines if line.startswith("bond_style")],
        represents_number,
    )
    angle_styles = flatten_list(
        [line.split()[1:] for line in lines if line.startswith("angle_style")],
        represents_number,
    )
    dihedral_styles = flatten_list(
        [line.split()[1:] for line in lines if line.startswith("dihedral_style")],
        represents_number,
    )

    atom_hybrid = len(atom_styles) > 1
    bond_hybrid = len(bond_styles) > 1
    angle_hybrid = len(angle_styles) > 1
    dihedral_hybrid = len(dihedral_styles) > 1

    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    if style_flag:
        for line in lines:
            # Only care about vdW pair interactions
            if line.startswith("pair_coeff") and not "coul" in line:
                if not atom_hybrid:
                    _, i, j, epsilon, sigma, *params = line.split()
                else:
                    _, i, j, astyle, epsilon, sigma, *params = line.split()

                if i == j:
                    atom_section.append([i, sigma, epsilon])

            if line.startswith("bond_coeff"):
                if not bond_hybrid:
                    bstyle = bond_styles[0]
                    _, i, *params = line.split()
                else:
                    _, i, bstyle, *params = line.split()

                if "#" in params:
                    params = params[: params.index("#")]

                bond_section.append([i, bstyle, params])

            if line.startswith("angle_coeff"):
                if not angle_hybrid:
                    astyle = angle_styles[0]
                    _, i, *params = line.split()
                else:
                    _, i, astyle, *params = line.split()

                if "#" in params:
                    params = params[: params.index("#")]

                angle_section.append([i, astyle, params])

            if line.startswith("dihedral_coeff"):
                if not dihedral_hybrid:
                    dstyle = dihedral_styles[0]
                    _, i, *params = line.split()
                else:
                    _, i, dstyle, *params = line.split()

                if "#" in params:
                    params = params[: params.index("#")]

                dihedral_section.append([i, dstyle, params])

    else:
        for line in lines:
            if in_atom_section:
                if line.startswith("Bond Coeffs"):
                    in_atom_section = False
                    in_bond_section = True
                elif line.strip() and not line.startswith("#"):
                    if not atom_hybrid:
                        i, epsilon, sigma, *params = line.split()
                    else:
                        i, astyle, epsilon, sigma, *params = line.split()

                    atom_section.append([i, sigma, epsilon])

            elif in_bond_section:
                if line.startswith("Angle Coeffs"):
                    in_bond_section = False
                    in_angle_section = True
                elif line.strip() and not line.startswith("#"):
                    if not bond_hybrid:
                        bstyle = bond_styles[0]
                        _, i, *params = line.split()
                    else:
                        _, i, bstyle, *params = line.split()

                    if "#" in params:
                        params = params[: params.index("#")]

                    bond_section.append([i, bstyle, params])

            elif in_angle_section:
                if line.startswith("Dihedral Coeffs"):
                    in_angle_section = False
                    in_dihedral_section = True
                elif line.strip() and not line.startswith("#"):
                    if not angle_hybrid:
                        astyle = angle_styles[0]
                        _, i, *params = line.split()
                    else:
                        _, i, astyle, *params = line.split()

                    if "#" in params:
                        params = params[: params.index("#")]

                    angle_section.append([i, astyle, params])

            elif in_dihedral_section:
                if line.strip() and not line.startswith("#"):
                    if not dihedral_hybrid:
                        dstyle = dihedral_styles[0]
                        _, i, *params = line.split()
                    else:
                        _, i, dstyle, *params = line.split()

                    if "#" in params:
                        params = params[: params.index("#")]

                    dihedral_section.append([i, dstyle, params])

            elif line.startswith("Pair Coeffs"):
                in_atom_section = True

    return (
        mixing,
        fudgeLJ,
        fudgeQQ,
        atom_section,
        bond_section,
        angle_section,
        dihedral_section,
    )


## Read out GROMACS


def extract_itp_file(itp_file: str):
    with open(itp_file) as f:
        lines = f.readlines()

    in_molecule_section = False
    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    molecule_section = []
    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    for line in lines:
        if in_molecule_section:
            if line.startswith("[ atoms ]"):
                in_molecule_section = False
                in_atom_section = True
            elif line.strip() and not line.startswith(";"):
                molecule_section.append(line.split())
        elif in_atom_section:
            if line.startswith("[ bonds ]"):
                in_atom_section = False
                in_bond_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_atom_section = False
            elif line.strip() and not line.startswith(";"):
                atom_section.append(line.split())
        elif in_bond_section:
            if line.startswith("[ angles ]"):
                in_bond_section = False
                in_angle_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_bond_section = False
            elif line.strip() and not line.startswith(";"):
                bond_section.append(line.split())
        elif in_angle_section:
            if line.startswith("[ dihedrals ]"):
                in_angle_section = False
                in_dihedral_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_angle_section = False
            elif line.strip() and not line.startswith(";"):
                angle_section.append(line.split())
        elif in_dihedral_section:
            if line.startswith("[") or line.startswith("#"):
                in_dihedral_section = False
            elif line.strip() and not line.startswith(";"):
                dihedral_section.append(line.split())
        elif line.startswith("[ moleculetype ]"):
            in_molecule_section = True
        elif line.startswith("[ bonds ]"):
            in_bond_section = True

    return molecule_section, atom_section, bond_section, angle_section, dihedral_section


def extract_top_file(top_file: str):
    with open(top_file) as f:
        lines = f.readlines()

    in_default_section = False
    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    default_section = []
    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    for line in lines:
        if in_default_section:
            if line.startswith("[ atomtypes ]"):
                in_default_section = False
                in_atom_section = True
            elif line.strip() and not line.startswith(";"):
                default_section.append(line.split())
        elif in_atom_section:
            if line.startswith("[ bondtypes ]"):
                in_atom_section = False
                in_bond_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_atom_section = False
            elif line.strip() and not line.startswith(";"):
                atom_section.append(line.split())
        elif in_bond_section:
            if line.startswith("[ angletypes ]"):
                in_bond_section = False
                in_angle_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_bond_section = False
            elif line.strip() and not line.startswith(";"):
                bond_section.append(line.split())
        elif in_angle_section:
            if line.startswith("[ dihedraltypes ]"):
                in_angle_section = False
                in_dihedral_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_angle_section = False
            elif line.strip() and not line.startswith(";"):
                angle_section.append(line.split())
        elif in_dihedral_section:
            if line.startswith("[") or line.startswith("#"):
                in_dihedral_section = False
            elif line.strip() and not line.startswith(";"):
                dihedral_section.append(line.split())
        elif line.startswith("[ defaults ]"):
            in_default_section = True
        elif line.startswith("[ atomtypes ]"):
            in_atom_section = True

    return default_section, atom_section, bond_section, angle_section, dihedral_section
