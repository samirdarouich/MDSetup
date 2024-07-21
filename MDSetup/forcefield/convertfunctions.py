import numpy as np
from typing import Dict

# Define precision of converting
PRECISION = 4


def source_destination_error(source, destination):
    raise KeyError(
        (f"Specified transformation from '{source}' to '{destination}' is not implemented! "
         "Valid are from 'gromacs' to 'lammps', and from 'lammps' to 'gromacs'")
    )


def convert_harmonic_bond(
    source: str, destination: str, bond: Dict[str, str | float | int]
):
    """
    gromacs uses b0 and then K, lammps uses K then b0
    GOMACS uses 1/2*K and lammps K, hence include factor 1/2 in lammps
    gromacs uses kJ/nm^2 and lammps kcal/A^2
    gromacs uses b0 in nm and lammps in A
    """
    new_bond = bond.copy()

    if source == "gromacs" and destination == "lammps":
        # [r, K] to [K, r]
        new_bond["p"] = [
            round(bond["p"][1] / 2 / 4.184 / 100, PRECISION),
            round(bond["p"][0] * 10, PRECISION),
        ]
        new_bond["style"] = "harmonic"

    elif source == "lammps" and destination == "gromacs":
        # [K, r] to [r, K]
        new_bond["p"] = [
            round(bond["p"][1] / 10, PRECISION),
            round(bond["p"][0] * 2 * 4.184 * 100, PRECISION),
        ]
        new_bond["style"] = 1
    else:
        source_destination_error(source, destination)

    return new_bond


def convert_harmonic_angle(
    source: str, destination: str, angle: Dict[str, str | float | int]
):
    """
    gromacs uses theta0 and then K, lammps uses K then theta0
    GOMACS uses 1/2*K and lammps K, hence include factor 1/2 in lammps
    gromacs uses kJ/nm^2 and lammps kcal/A^2
    """
    new_angle = angle.copy()

    if source == "gromacs" and destination == "lammps":
        # [theta0, K] to [K, theta0]
        new_angle["p"] = [
            round(angle["p"][1] / 2 / 4.184, PRECISION),
            round(angle["p"][0], PRECISION),
        ]
        new_angle["style"] = "harmonic"

    elif source == "lammps" and destination == "gromacs":
        # [K, theta0] to [theta0, K]
        new_angle["p"] = [
            round(angle["p"][1], PRECISION),
            round(angle["p"][0] * 2 * 4.184, PRECISION),
        ]
        new_angle["style"] = 1
    else:
        source_destination_error(source, destination)

    return new_angle


def convert_harmonic_dihedral(
    source: str, destination: str, dihedral: Dict[str, str | float | int]
):
    """
    gromacs phi, k, n, lammps, k, d, n
    """
    new_dihedral = dihedral.copy()

    if source == "gromacs" and destination == "lammps":
        # gromacs uses trans = 0.0 while lammps trans = 180, hence d=-1
        new_dihedral["p"] = [
            round(dihedral["p"][1] * 4.184, PRECISION),
            -1,
            dihedral["p"][2],
        ]
        new_dihedral["style"] = "harmonic"

    elif source == "lammps" and destination == "gromacs":
        # gromacs uses trans = 0.0 while lammps trans = 180, but introduces a phi0 term.
        # If d=-1, phi0 = 0.0, if d=1 than phi0=180
        new_dihedral["p"] = [
            0.0 if dihedral["p"][1] == -1 else 180.0,
            round(dihedral["p"][0] / 4.184, PRECISION),
            dihedral["p"][2],
        ]
        new_dihedral["style"] = 1
    else:
        source_destination_error(source, destination)

    return new_dihedral


def convert_opls_dihedral(
    source: str, destination: str, dihedral: Dict[str, str | float | int]
):
    """ """
    new_dihedral = dihedral.copy()

    if source == "gromacs" and destination == "lammps":
        # Convert Ryckaert-Bellemans to nharmnoic (angle conversion has to be accounted with (-1)**n)
        new_dihedral["p"] = [
            6,
            *[
                round(p / 4.184 * (-1) ** i, PRECISION)
                for i, p in enumerate(dihedral["p"])
            ],
        ]
        new_dihedral["style"] = "nharmonic"

    elif source == "lammps" and destination == "gromacs":
        # Convert OPLS dihedral to Ryckaert-Bellemans (https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#equation-eqnoplsrbconversion)
        A = np.matrix(
            [
                [0.5, 1, 0.5, 0],
                [-0.5, 0, 3 / 2, 0],
                [0, -1, 0, 4],
                [0, 0, -2, 0],
                [0, 0, 0, -4],
                [0, 0, 0, 0],
            ]
        )
        opls = np.array(dihedral["p"]).reshape(-1, 1) * 4.184
        new_dihedral["p"] = np.round((A * opls).flatten(), PRECISION).tolist()[0]
        new_dihedral["style"] = 3
    else:
        source_destination_error(source, destination)

    return new_dihedral
