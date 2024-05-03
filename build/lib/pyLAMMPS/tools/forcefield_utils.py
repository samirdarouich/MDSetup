import os
import numpy as np
import yaml, toml, json
from typing import Dict

# Define precision of converting
PRECISION = 4

def source_destination_error(source,destination):
    raise KeyError(f"Specified transformation from '{source}' to '{destination}' is not implemented! Valid are from 'GROMACS' to 'LAMMPS', and from 'LAMMPS' to 'GROMACS'")

def convert_harmonic_bond( source: str, destination: str, bond: Dict[str,str|float|int] ):
    """
    GROMACS uses b0 and then K, LAMMPS uses K then b0
    GOMACS uses 1/2*K and LAMMPS K, hence include factor 1/2 in LAMMPS
    GROMACS uses kJ/nm^2 and LAMMPS kcal/A^2
    GROMACS uses b0 in nm and LAMMPS in A
    """
    new_bond = bond.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        # [r, K] to [K, r]
        new_bond["p"] = [ round(bond["p"][1] / 2 / 4.184 / 100, PRECISION), round(bond["p"][0] * 10, PRECISION) ]
        new_bond["style"] = "harmonic"

    elif source == "LAMMPS" and destination == "GROMACS":
        # [K, r] to [r, K]
        new_bond["p"] = [ round(bond["p"][1] / 10, PRECISION), round(bond["p"][0] * 2* 4.184 * 100, PRECISION) ]
        new_bond["style"] = 1
    else:
        source_destination_error( source ,destination )

    return new_bond

def convert_harmonic_angle( source: str, destination: str, angle: Dict[str,str|float|int] ):
    """
    GROMACS uses theta0 and then K, LAMMPS uses K then theta0
    GOMACS uses 1/2*K and LAMMPS K, hence include factor 1/2 in LAMMPS
    GROMACS uses kJ/nm^2 and LAMMPS kcal/A^2
    """
    new_angle = angle.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        # [theta0, K] to [K, theta0]
        new_angle["p"] = [ round(angle["p"][1] / 2 / 4.184, PRECISION), round(angle["p"][0], PRECISION) ]
        new_angle["style"] = "harmonic"

    elif source == "LAMMPS" and destination == "GROMACS":
        # [K, theta0] to [theta0, K]
        new_angle["p"] = [ round(angle["p"][1], PRECISION), round(angle["p"][0] * 2* 4.184, PRECISION) ]
        new_angle["style"] = 1
    else:
        source_destination_error( source , destination )
    
    return new_angle

def convert_harmonic_dihedral( source: str, destination: str, dihedral: Dict[str,str|float|int] ):
    """
    GROMACS phi, k, n, LAMMPS, k, d, n
    """
    new_dihedral = dihedral.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        # GROMACS uses trans = 0.0 while LAMMPS trans = 180, hence d=-1
        new_dihedral["p"] = [ round(dihedral["p"][1] * 4.184, PRECISION) , -1, dihedral["p"][2] ]
        new_dihedral["style"] = "harmonic"

    elif source == "LAMMPS" and destination == "GROMACS":
        # GROMACS uses trans = 0.0 while LAMMPS trans = 180, but introduces a phi0 term.
        # If d=-1, phi0 = 0.0, if d=1 than phi0=180
        new_dihedral["p"] = [ 0.0 if dihedral["p"][1]==-1 else 180.0, round(dihedral["p"][0] / 4.184, PRECISION), dihedral["p"][2] ]
        new_dihedral["style"] = 1
    else:
        source_destination_error( source ,destination )

    return new_dihedral

def convert_opls_dihedral( source: str, destination: str, dihedral: Dict[str,str|float|int] ):
    """
    """
    new_dihedral = dihedral.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        # Convert Ryckaert-Bellemans to nharmnoic (angle conversion has to be accounted with (-1)**n)
        new_dihedral["p"] = [6, *[ round(p / 4.184*(-1)**i, PRECISION) for i,p in enumerate(dihedral["p"]) ] ]
        new_dihedral["style"] = "nharmonic"

    elif source == "LAMMPS" and destination == "GROMACS":
        # Convert OPLS dihedral to Ryckaert-Bellemans (https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#equation-eqnoplsrbconversion)
        A = np.matrix( [ [0.5, 1, 0.5, 0], [-0.5, 0, 3/2, 0], [0,-1,0,4], [0,0,-2,0], [0,0,0,-4], [0,0,0,0] ] )
        opls = np.array(dihedral["p"][1:]).reshape(-1,1) * 4.184
        new_dihedral["p"] = np.round((A*opls).flatten(), PRECISION).tolist()[0]
        new_dihedral["style"] = 1
    else:
        source_destination_error( source ,destination )

    return new_dihedral

# MAPPING DICTS

BOND_MAP = { 1: convert_harmonic_bond, "harmonic": convert_harmonic_bond, 
            }

ANGLE_MAP = { 1: convert_harmonic_angle, "harmonic": convert_harmonic_angle 
             }

DIHEDRAL_MAP = { 1: convert_harmonic_dihedral, "harmonic": convert_harmonic_dihedral,
                 3: convert_opls_dihedral, "opls": convert_opls_dihedral 
                }


def convert_atom( source: str, destination: str, atom: Dict[str,str|float|int] ):

    new_atom = atom.copy()

    if source == "GROMACS" and destination == "LAMMPS":
        new_atom["epsilon"] = round( atom["epsilon"] / 4.184, PRECISION ) # convert from kJ to kcal
        new_atom["sigma"] = round( atom["sigma"] * 10, PRECISION ) # convert from nm to A 
        new_atom["vdw_style"] = "lj/cut"
        new_atom["coul_style"] = "coul/long" if atom["charge"] != 0 else ""
    elif source == "LAMMPS" and destination == "GROMACS":
        new_atom["epsilon"] = round( atom["epsilon"] * 4.184, PRECISION ) # convert from kcal to kJ
        new_atom["sigma"] = round( atom["sigma"] / 10, PRECISION ) # convert from A to nm 
        new_atom.pop( "vdw_style" )
        new_atom.pop( "coul_style" )
    else:
        source_destination_error( source ,destination )
    
    return new_atom

def convert_bond( source: str, destination: str, bond: Dict[str,str|float|int] ):
    new_bond = BOND_MAP[bond["style"]]( source, destination, bond )

    return new_bond

def convert_angle( source: str, destination: str, angle: Dict[str,str|float|int] ):
    new_angle = ANGLE_MAP[angle["style"]]( source, destination, angle )

    return new_angle

def convert_dihedral( source: str, destination: str, dihedral: Dict[str,str|float|int] ):
    new_dihedral = DIHEDRAL_MAP[dihedral["style"]]( source, destination, dihedral )

    return new_dihedral

def convert_force_field( force_field_path: str, output_path: str ):

    if not "/" in output_path:
        output_path = "./"+output_path
    
    if not os.path.exists(force_field_path):
        raise FileExistsError(f"Specified force field file does not exist: '{force_field_path}'")

    if ".yaml" in force_field_path:
        data = yaml.safe_load( open(force_field_path) )
    elif ".json" in force_field_path:
        data = json.load( open(force_field_path) )
    elif ".toml" in force_field_path:
        data = toml.load( open(force_field_path) )

    source = data["format"]

    if source == "GROMACS":
        destination = "LAMMPS"
    elif source == "LAMMPS":
        destination == "GROMACS"
    else:
        raise KeyError(f"Specified source is unknown: '{source}'")

    print(f"Convert force field from '{source}' to '{destination}'")

    converted_ff = { "format": destination, "atoms": {}, "bonds": {}, "angles": {}, "torsions": {} }

    for key, atom in data["atoms"].items():
        converted_ff["atoms"][key] = convert_atom( source = source, destination = destination, atom = atom )

    for key, bond in data["bonds"].items():
        converted_ff["bonds"][key] = convert_bond( source = source, destination = destination, bond = bond )

    for key, angle in data["angles"].items():
        converted_ff["angles"][key] = convert_angle( source = source, destination = destination, angle = angle )

    for key, dihedral in data["torsions"].items():
        converted_ff["torsions"][key] = convert_dihedral( source = source, destination = destination, dihedral = dihedral )

    os.makedirs( os.path.dirname(output_path), exist_ok = True )

    if ".yaml" in output_path:
        yaml.dump( converted_ff, open(output_path,"w") )
    elif ".json" in output_path:
        json.dump( converted_ff, open(output_path,"w"), indent=2 )
    elif ".toml" in output_path:
        toml.dump( converted_ff, open(output_path,"w") ) 
