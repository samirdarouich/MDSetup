import os
import moleculegraph
import json, yaml, toml

from jinja2 import Template
from typing import List, Dict, Any
from ..tools.molecule import get_molecule_coordinates
from ..tools.general import flatten_list, merge_nested_dicts, unique_by_key

class GROMACS_molecules():
    """
    This class writes GROMACS data input for arbitrary mixtures using moleculegraph.
    """
    def __init__(self, mol_str: List[str], force_field_paths: List[str] ):

        # Save moleclue graphs of both components class wide
        self.mol_str    = mol_str
        self.mol_list   = [ moleculegraph.molecule(mol) for mol in self.mol_str ]

        # Read in force field files
        self.ff = {}
        for force_field_path in force_field_paths:
            if ".yaml" in force_field_path:
                data = yaml.safe_load( open(force_field_path) )
            elif ".json" in force_field_path:
                data = json.load( open(force_field_path) )
            elif ".toml" in force_field_path:
                data = toml.load( open(force_field_path) )
            else:
                raise KeyError(f"Force field file is not supported: '{force_field_path}'. Please provide 'YAML', 'JSON', or 'TOML' file.")
            # Update overall dict
            merge_nested_dicts( self.ff, data.copy() )

        ## Map force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions) ##

        # Get (unique) atom types and parameters
        self.nonbonded =  unique_by_key( flatten_list( molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in self.mol_list ), "name" )
        
        # Get (unique) bond types and parameters
        self.bonds     =  unique_by_key( flatten_list( molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in self.mol_list ), "list" )
        
        # Get (unique) angle types and parameters
        self.angles    =  unique_by_key( flatten_list( molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in self.mol_list ), "list" )
        
        # Get (unique) torsion types and parameters 
        self.torsions  =  unique_by_key( flatten_list( molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in self.mol_list ), "list" ) 
        
        if not all( [ all(self.nonbonded), all(self.bonds), all(self.angles), all(self.torsions) ] ):
            txt = "nonbonded" if not all(self.nonbonded) else "bonds" if not all(self.bonds) else "angles" if not all(self.angles) else "torsions"
            raise ValueError("Something went wrong during the force field mapping for key: %s"%txt)

    def write_gromacs_itp( self, itp_template: str, itp_path: str, residue: List[str], nrexcl: List[int] ):
        """
        Function that generates GROMACS itp files for all molecules.

        Parameters:
            itp_template (str): Path to the jinja2 template for the GROMACS itp file.
            itp_path (str): Path where the GROMACS itp files should be generated.
            residue (List[str]): List with residue names.
            nrexcl (List[int]): List with excluding non-bonded interactions between atoms that are no further than 'nrexcl' bonds away.

        Return:
         - itp_files (List[str]): List with absolute paths of created itp files
        """

        if not os.path.exists(itp_template):
            raise FileExistsError(f"Itp template does not exists:\n   {itp_template}")

        with open(itp_template) as file_:
            template = Template(file_.read())
            
        os.makedirs( itp_path, exist_ok=True )

        itp_files = []

        for m,mol in enumerate(self.mol_list):

            # Define atoms
            # GROMACS INPUT: Atom n°     type     resnr    residu     atom      cgnr      charge        mass
            nb_ff = mol.map_molecule( mol.atom_names, self.ff["atoms"] )
            gmx_atom_list = [ [ i+1, ff_atom["name"], 1, residue[m], ff_atom["name"], i+1, ff_atom["charge"], ff_atom["mass"] ] for i,ff_atom in enumerate(nb_ff) ]

            # Define bonds 
            # GROMACS INPUT: Atom n° of atoms in the bond
            bond_numbers = mol.bond_list + 1 
            bond_names = [ [mol.atom_names[i] for i in bl] for bl in mol.bond_list ] 
            gmx_bond_list = [ [ *bond, "#", " ".join(bond_name) ] for bond,bond_name in zip( bond_numbers, bond_names ) ]

            # Define angles
            # GROMACS INPUT: Atom n° of atoms in the angle
            angle_numbers = mol.angle_list + 1 
            angle_names = [ [mol.atom_names[i] for i in al] for al in mol.angle_list ] 
            gmx_angle_list = [ [ *angle, "#", " ".join(angle_name) ] for angle,angle_name in zip( angle_numbers, angle_names ) ]

            # Define torsions
            # GROMACS INPUT: Atom n° of atoms in the torsion
            torsion_numbers = mol.torsion_list + 1 
            torsion_names = [ [mol.atom_names[i] for i in tl] for tl in mol.torsion_list ] 
            gmx_torsion_list = [ [ *torsion, "#", " ".join(torsion_name) ] for torsion,torsion_name in zip(torsion_numbers, torsion_names) ]


            renderdict = { "residue": residue[m],
                           "nrexcl": nrexcl[m],
                           "atoms": gmx_atom_list,
                           "bonds": gmx_bond_list,
                           "angles": gmx_angle_list,
                           "dihedrals": gmx_torsion_list  
                        }
            
            rendered = template.render( renderdict )

            with open(f"{itp_path}/{residue[m]}.itp", "w") as fh:
                fh.write(rendered)

            itp_files.append( f"{itp_path}/{residue[m]}.itp" )

        return itp_files
    
    def write_gromacs_top( self, top_template: str, top_path: str, comb_rule: int,
                           system_name: str, itp_files: List[str],
                           residue_dict: Dict[str,int], fudgeLJ: float=1.0, 
                           fudgeQQ: float=1.0, ):
        """
        Function that generates GROMACS top file for the system.

        Parameters:
            top_template (str): Path to the jinja2 template for the GROMACS top file.
            top_path (str): Path where the GROMACS top file should be generated.
            comb_rule (int): Combination rule to use. Possible are 1, 2, or 3.
                             If 1: Vii, Wii = 4epsilon_i*sigma_i^6, 4epsilon_i*sigma_i^12
                             If 2 or 3: Vii, Wii = sigma_i, epsilon_i
                             1 and 3: geometric mixing
                             2: arithmetic mixing.
            system_name (str): Name of the system.
            itp_files (List[str]): List with (absolute/relative) paths to itp files.
            residue_dict (Dict[str,int]): Dict with residue names and their corresponding number of molecules
            fudgeQQ (float, optional): Factor by which to multiply Lennard-Jones 1-4 interactions. Defaults to 1.0
            fudgeLJ (float, optional): Factor by which to multiply Lennard-Jones 1-4 interactions. Defaults to 1.0
        """

        if not os.path.exists(top_template):
            raise FileExistsError(f"Itp template does not exists:\n   {top_template}")

        with open(top_template) as file_:
            template = Template(file_.read())
            
        os.makedirs( os.path.dirname(top_path), exist_ok=True )

        atom_paras = [ [ nb["name"], nb["atom_no"], nb["mass"], nb["charge"], "A", nb["sigma"], nb["epsilon"] ] for nb in self.nonbonded ]
        bond_paras = [ [ *bonded["list"], bonded["style"], *bonded["p"] ] for bonded in self.bonds ]
        angles_paras = [ [ *bonded["list"], bonded["style"], *bonded["p"] ] for bonded in self.angles ]
        torsion_paras = [ [ *bonded["list"], bonded["style"], *bonded["p"] ] for bonded in self.torsions ]
        
        renderdict = { "comb_rule": comb_rule,
                        "system_name": system_name,
                        "itp_files": itp_files,
                        "residue_dict": residue_dict,
                        "fudgeLJ": fudgeLJ,
                        "fudgeQQ": fudgeQQ,
                        "atoms": atom_paras,
                        "bonds": bond_paras,
                        "angles": angles_paras,
                        "dihedrals": torsion_paras  
                    }
            
        rendered = template.render( renderdict )

        with open(top_path, "w") as fh:
            fh.write(rendered)

        return top_path

def write_gro_files( destination: str, molecules_dict_list: List[Dict[str,str|float]], gro_template: str ):
    """
    Function that generates GROMACS gro file for a list of molecules based on their SMILES and their moleculegraph representation.

    Parameters:
     - destination (str): Path where the .gro files are writen to
     - molecules_dict_list (List[Dict[str,str|float]]): List with dictionaries of each molecule, containing the keys: smiles, graph, name, and number
     - gro_template (str): Path to the gro template file.
    """

    if not os.path.exists(gro_template):
        raise FileExistsError(f"Gro file template does not exists:\n   {gro_template}")
    
    coordinate_paths = [ f'{destination}/{mol["name"]}.gro' for mol in molecules_dict_list ]

    os.makedirs( destination, exist_ok = True)

    (raw_atom_numbers, final_atomtyps, 
    final_atomsymbols, final_coordinates) = get_molecule_coordinates( molecule_name_list = [ mol["name"] for mol in molecules_dict_list ], 
                                                                      molecule_graph_list = [ mol["graph"] for mol in molecules_dict_list ],
                                                                      molecule_smiles_list = [ mol["smiles"] for mol in molecules_dict_list ],
                                                                      verbose = False
                                                                    )

    # Write coordinates to gro files.
    for i,(coord_destination, raw_atom_number, final_atomtyp, final_atomsymbol, final_coordinate) in enumerate( zip( coordinate_paths, raw_atom_numbers, final_atomtyps, final_atomsymbols, final_coordinates ) ):
        
        with open(gro_template) as file:
            template = Template(file.read())
        
        gromacs = { "name": molecules_dict_list[i]["name"], 
                    "no_atoms": f"{len(raw_atom_number):5d}", 
                    "atoms": [],
                    "box_dimension": "%10.5f%10.5f%10.5f"%(0.0, 0.0, 0.0)
                    }
        
        # Gromacs guesses the atomic radii for the insert based on the name in the gro file, hence it makes sense to name the atoms with their element

        # Xyz coordinates are given in Angstrom, convert in nm
        # Provide GROMACS with  RESno, RESNAME. attyp, running number, x, y ,z (optional velocities)
        gromacs["atoms"] = [ "%5d%-5s%5s%5d%8.3f%8.3f%8.3f # %s"%( 1, molecules_dict_list[i]["name"][:5], atsym[:5], j+1, float(x) / 10, float(y) / 10, float(z) / 10, attyp ) 
                                for j,(attyp, atsym, (x,y,z)) in enumerate(zip(final_atomtyp,final_atomsymbol, final_coordinate)) ]

        rendered = template.render( gromacs )

        with open(coord_destination, "w") as fh:
            fh.write( rendered )

    return coordinate_paths