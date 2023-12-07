import os
import re
import toml
import numpy as np
import moleculegraph
from typing import List, Dict
from jinja2 import Template
from scipy.constants import Avogadro


from .utils import get_local_dipol, calc_mie, calc_coulomb, calc_bond

class LAMMPS_input():
    """
    This class can be used to build initial systems using Playmol (Packmol) and simple xyz for each component. Furthermore, it writes LAMMPS data files from the produced system xyz, 
    as welll as LAMMPS input files. For every writing task, jinja2 templates are utilized. The LAMMPS input template can be adjusted as necessary.
    """

    def __init__(self, mol_str: List[str], ff_path: str, decoupling: bool=False, table_path: str="", charge_group_approach: bool=False):
        """
        Initilizing LAMMPS input class. Save system independent force field parameters (bonds, angles, torsions).

        Args:
            mol_list (List[str]): List containing moleculegraph strings for the component(s). These will be transalted into moleculegraph objects
            ff_path (str): Path to toml file containing used force-field readable format by moleculegraph.
            decoupling (bool,optional): This will decouple the first component (should just be one molecule) from all other components.
                                        This mean that the intermolecular interactions with all other components will be coupled using a lambda parameter. Defaults to False.
            charge_group_approach (bool, optional): If the charge group approach should be utilized. Defaults to False.
            table_path (str, optional): Provide a (relative) path to the table containing all tabled bonds (these include the nonbonded interactions evaluated in the charge group approach).                      
        """

        # Save moleclue graphs of both components class wide
        self.mol_str  = mol_str
        self.mol_list = [ moleculegraph.molecule(mol) for mol in mol_str ]

        # Read in force field toml file
        with open(ff_path) as ff_toml_file:
            self.ff = toml.load(ff_toml_file)

        # If the charge group approach should be utilized, the force field as well as the molecule graph objects are altered to consider two local dipols within a molecule
        if charge_group_approach: 
            print("\nCharge group approach is utilized. Scan for local dipols:\n")
            self.apply_charge_group_approach( table_path )


        ## Specify force field parameters for all interactions seperately (nonbonded, bonds, angles and torsions)

        # Get (unique) atom types and parameters #
        self.nonbonded = [j for sub in [molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) bond types and parameters #
        self.bonds     = [j for sub in [molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) angle types and parameters #     
        self.angles    = [j for sub in [molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in self.mol_list] for j in sub]
        
        # Get (unique) torsion types and parameters #
        self.torsions  = [j for sub in [molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in self.mol_list] for j in sub]
        
        if not all( [ all(self.nonbonded), all(self.bonds), all(self.angles), all(self.torsions) ] ):
            txt = "nonbonded" if not all(self.nonbonded) else "bonds" if not all(self.bonds) else "angles" if not all(self.angles) else "torsions"
            raise ValueError("Something went wrong during the force field mapping for key: %s"%txt)
        
        # Get nonbonded force field for all atom types not only the unique one. This is later used to extract the charge of each atom in the system while writing the LAMMPS data file.
        self.ff_all    = np.array([j for sub in [molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) for molecule in self.mol_list] for j in sub])


        #### Define general settings that are not system size dependent ####

        self.renderdict              = {}


        ## Definitions for atoms in the system ##

        # This is a helper list, that higher the unique atom indexes in such a way that the atom indexes of the 2nd component 
        # start after the indexes of the first component (e.g.: Component 1: 1 cH_alcohol, 2 OH_alcohol 3 CH2_alcohol --> Component 2: 4 CH3_alkane, ...)
        # As molecule graph just gives for each component the indexes starting from 0 --> Component 1: 0 cH_alcohol, 1 OH_alcohol 2 CH2_alcohol; Component 2: 0 CH3_alkane, ... 
        # one needs to add the index of all preceding molecules.
        add_atoms                    = [1] + [ sum(len(mol.unique_atom_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        # Get the force field type identifiers for each atom. This is used in the #Atoms section in the data file.
        # Therefore take the identifier number of every atom in each moleculegraph and add the preceeding number of atom types.
        # 1 1 1 0.404 -10.69874 -8.710742 1.59434903 # cH_alcohol1 --> First index is a running LAMMPS number for the atom. The 2nd index is the running numner for the molecule.
        # The third index is the force field type identifier (here it is 1, thus it is type the cH_alcohol type, as defined in the input file header)
        # The forth number is the partial charge. The following 3 numbers are the x, y and z coordinate of the atom in the system.
        self.atoms_running_number    = np.concatenate( [mol.unique_atom_inverse + add_atoms[i] for i,mol in enumerate(self.mol_list)], axis=0 )
        
        # These are the !unique! atom force field types from "atoms_running_number". This is used in the atom definition section of the data file.
        # As just the unique atom force field types are defined in LAMMPS. 
        self.atom_numbers_ges        = np.unique( self.atoms_running_number )

        # Get the number of atoms per component. This will later be multiplied with the number of molecules per component to get the total number of atoms in the system
        self.number_of_atoms         = [ mol.atom_number for mol in self.mol_list ]

        # Define the total number of atom tpyes
        self.renderdict["atom_type_number"] = len( self.atom_numbers_ges )

        ## Definitions for bonds in the system ##

        # This is the same as for the unique atom types, just for the unique bond types. One need to add the amount of all preceeding unique
        # bond types to every bond type of each component.
        add_bonds = [1] + [ sum(len(mol.unique_bond_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        # Get the identifiers for each bond with there corresponding force field type. 
        # (e.g.: Ethanediol: 1 2 3 2 1: where 1 is the cH_alc - OH_alc bond, 2 is the OH_alc - CH2_alc bond, and 3 the CH2_alc - CH2_alc bond)
        # These types will be written in the # Bonds section. Where every bond for every atom in the system is defined. 
        # (e.g.: 1 1 1 5 # cH_alc OH_alc --> the first number is a running index for LAMMPS, the 2nd is the force field bond type index (as defined in bonds_running_number),
        # and the 3th and 4th are the atoms in this bond (as defined in self.bond_numbers)
        self.bonds_running_number    = np.concatenate( [mol.unique_bond_inverse + add_bonds[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")

        # These are the !unique! bond force field types from "bonds_running_number". This is used in the bonds definition section of the data file.
        # As just the unique bond force field types are defined in LAMMPS. 
        self.bond_numbers_ges        = np.unique( self.bonds_running_number )
        
        # This identify the atoms of each bond in the molecule. This is used in #Bonds section, where each Atom is assigned a bond, as well as the bond force field type. 
        # To these indicies the number of preceeding atoms in the system will be added (in the write_lammps_data function).
        # Thus the direct list from molecule graph can be used, without further refinement. 
        # (E.g.: Bond_list for ethanediol: [ [1,2], [2,3], [3,4], [4,5], [5,6] ] )
        self.bond_numbers            = np.concatenate( [mol.bond_list + 1 for mol in self.mol_list if mol.bond_list.size > 0], axis=0 ).astype("int")

        # This is just the name of each the bonds defined above. This is written as information that one knows what which bond is defined in the data file.
        self.bond_names              = np.concatenate( [[[mol.atom_names[i] for i in bl] for bl in mol.bond_list] for mol in self.mol_list if mol.bond_list.size > 0], axis=0 )

        # Get the number of bonds per component. This will later be multiplied with the number of molecules per component to get the total number of bonds in the system
        self.number_of_bonds         = [ len(mol.bond_keys) for mol in self.mol_list ]

        # If several bond styles are used, these needs to be added in the data file, as well as the "hybrid" style.
        self.renderdict["bond_styles"]        = list( np.unique( [ p["style"] for p in self.bonds ] ) )

        # Defines the total number of bond types
        self.renderdict["bond_type_number"]   = len( self.bond_numbers_ges )


        ## Definitions for angles in the system --> (all elements here have the same meaning as above for bonds just for angles) ##
        
        add_angles = [1] + [ sum(len(mol.unique_angle_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]

        self.angles_running_number   = np.concatenate( [mol.unique_angle_inverse + add_angles[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")
        self.angle_numbers_ges       = np.unique( self.angles_running_number )
        self.angle_numbers           = np.concatenate( [mol.angle_list + 1 for mol in self.mol_list if mol.angle_list.size > 0], axis=0 ).astype("int")
        self.angle_names             = np.concatenate( [[[mol.atom_names[i] for i in al] for al in mol.angle_list] for mol in self.mol_list if mol.angle_list.size > 0], axis=0 )
        self.number_of_angles        = [ len(mol.angle_keys) for mol in self.mol_list ]

        self.renderdict["angle_styles"]        = list( np.unique( [ p["style"] for p in self.angles] ) )
        self.renderdict["angle_type_number"]   = len(self.angle_numbers_ges)


        ## Definitions for torsions in the system --> (all elements here have the same meaning as above for bonds just for torsions) ##
        
        add_torsions = [1] + [ sum(len(mol.unique_torsion_keys) for mol in self.mol_list[:(i+1)]) + 1 for i in range( len(self.mol_list[1:]) ) ]
        
        self.torsions_running_number = np.concatenate( [mol.unique_torsion_inverse + add_torsions[i] for i,mol in enumerate(self.mol_list)], axis=0 ).astype("int")
        self.torsion_numbers_ges     = np.unique( self.torsions_running_number )
        self.torsion_numbers         = np.concatenate( [mol.torsion_list + 1 for mol in self.mol_list if mol.torsion_list.size > 0], axis=0 ).astype("int")
        self.torsion_names           = np.concatenate( [[[mol.atom_names[i] for i in tl] for tl in mol.torsion_list] for mol in self.mol_list if mol.torsion_list.size > 0], axis=0 )
        self.number_of_torsions      = [ len(mol.torsion_keys) for mol in self.mol_list ]

        self.renderdict["torsion_styles"]      = list( np.unique( [ p["style"] for p in self.torsions ] ) )
        self.renderdict["torsion_type_number"] = len(self.torsion_numbers_ges)
        
        return

    def apply_charge_group_approach(self, table_path: str):
        """
        Function that applies the charge group approach to the components. This function alters the bond entries in the moleculegraph objects, as well as the force field saved within this class.

        Args:
            table_path (str): Path where the tabled potential are written to or already exist.
        """

        self.dipol_lists = [ get_local_dipol(mol,self.ff) for mol in self.mol_list ]

        for j,(dipol_list, mol) in enumerate(zip( self.dipol_lists, self.mol_list )):
            
            # If dipol list is empty or only contains one local dipol, skip !
            if len(dipol_list) < 2: continue

            print("\nCharge group approach is applied to molecule: %s"%(self.mol_str[j]))

            dipol_names = [ "Local dipol n°%d: "%i + " ".join(mol.atom_names[np.array(dipol)]) for i,dipol in enumerate(dipol_list)] 
            
            print("\nThese are the local dipols identified:\n%s\n"%("\n".join(dipol_names)))
            
            # Use the dipol list, to create a special bond list, as well as the corresponding moleculegraph representation of it.
            # This means matching the correct force field types to each of the atoms in the special bonds list.
            special_bond_indexes = np.array( np.meshgrid( dipol_list[0], dipol_list[1] ) ).T.reshape(-1, 2)
            special_bond_names   = mol.atom_names[special_bond_indexes]
            special_bond_keys    = [ "[special]"+ moleculegraph.make_graph( moleculegraph.sort_force_fields(x) ) for x in special_bond_names ]

            # Delete unnecessary standard bonds. Checks which standard bonds also exist in the special bonds and therefore remove them
            idx             = np.array( [ i for i,entry in enumerate(mol.bond_list) if tuple(entry) not in set(map(tuple,special_bond_indexes)) ] )

            # Overwrite the bonding information of the moleculegraph object with the new bonds including the special bonds!
            mol.bond_list   = np.concatenate( [mol.bond_list[idx], special_bond_indexes], axis=0 )
            mol.bond_names  = np.concatenate( [mol.bond_names[idx], special_bond_names], axis=0 )
            mol.bond_keys   = np.concatenate( [mol.bond_keys[idx], special_bond_keys], axis=0 )

            mol.unique_bond_keys, mol.unique_bond_indexes, mol.unique_bond_inverse = moleculegraph.molecule_utils.unique_sort( mol.bond_keys, return_inverse=True )
            mol.unique_bond_names   = mol.bond_names[ mol.unique_bond_indexes ]
            mol.unique_bond_numbers = mol.bond_list[ mol.unique_bond_indexes ]    

            # Add the force field information for special bonds
            special_bond_keys_unique, uidx = np.unique(special_bond_keys, return_index=True)
            special_bond_names_unique      = special_bond_names[uidx]

            print("\nThese are the unique special bonds that are added for intramolecular interaction:\n%s\n"%("\n".join( " ".join(sb) for sb in special_bond_names_unique )))
            
            # Jinja2 template uses p[1] as first entry and p[0] as second --> thats why at first the key then the table name.
            for ( bkey, bname ) in zip(special_bond_keys_unique, special_bond_names_unique):
                self.ff["bonds"][bkey] = { "list": list(bname), "p":  [ bkey, table_path ], "style": "table", "type": -1}
        
        return 

    def prepare_playmol_input(self, playmol_template: str, playmol_ff_path: str):
        """
        Function that writes playmol force field using a jinja2 template.

        Args:
            playmol_template (str): Path to playmol template for system building.
            playmol_ff_path (str): Path were the new playmol force field file should be writen to.
        """

        # Playmol can't handle the charge group approach, thus the playmol system is build disregarding this approach
        # and hence all the special coulombic and vdW interactions. To do this, the original (unaltered) moleculegraph representations are used to produce playmol input.
        mol_list  =  [ moleculegraph.molecule(mol) for mol in self.mol_str ]

        # Get (unique) atom types and parameters #
        nonbonded = np.array([j for sub in [molecule.map_molecule( molecule.unique_atom_keys, self.ff["atoms"] ) for molecule in mol_list] for j in sub])
        
        # Get (unique) bond types and parameters #
        bonds     = [j for sub in [molecule.map_molecule( molecule.unique_bond_keys, self.ff["bonds"] ) for molecule in mol_list] for j in sub]
        
        # Get (unique) angle types and parameters #     
        angles    = [j for sub in [molecule.map_molecule( molecule.unique_angle_keys, self.ff["angles"] ) for molecule in mol_list] for j in sub]
        
        # Get (unique) torsion types and parameters #
        torsions  = [j for sub in [molecule.map_molecule( molecule.unique_torsion_keys, self.ff["torsions"] ) for molecule in mol_list] for j in sub]
     

        ## Prepare dictionary for jinja2 template to write force field input for Playmol ##

        renderdict              = {}
        renderdict["nonbonded"] = list( zip( [j for sub in [molecule.unique_atom_keys for molecule in mol_list] for j in sub], nonbonded ) )
        renderdict["bonds"]     = list( zip( [j for sub in [molecule.unique_bond_names for molecule in mol_list] for j in sub], bonds ) )
        renderdict["angles"]    = list( zip( [j for sub in [molecule.unique_angle_names for molecule in mol_list] for j in sub], angles ) )
        renderdict["torsions"]  = list( zip( [j for sub in [molecule.unique_torsion_names for molecule in mol_list] for j in sub], torsions ) )
        
        # Generate force field file for playmol using jinja2 template
        os.makedirs( os.path.dirname(playmol_ff_path), exist_ok=True )

        with open(playmol_template) as file_:
            template = Template(file_.read())
        
        rendered = template.render( rd=renderdict )

        with open(playmol_ff_path, "w") as fh:
            fh.write(rendered) 
    
        return
    
    def write_playmol_input(self, playmol_template: str, playmol_path: str, playmol_ff_path: str, xyz_paths: List[str], playmol_executeable: str="~/.local/bin/playmol"):
        """
        Function that generates input file for playmol to build the specified system, as well as execute playmol to build the system

        Args:
            playmol_template (str): Path to playmol input template.
            playmol_path (str): Path where the playmol .mol file is writen and executed.
            playmol_ff_path (str): Path to the playmol force field file.
            xyz_paths (List[str]): List with the path(s) to the xyz file(s) for each component.
            playmol_executeable (str, optional): Path to playmol executeable. Defaults to "~/.local/bin/playmol"
        """

        # Playmol can't handle the charge group approach, thus the playmol system is build disregarding this approach
        # and hence all the special coulombic and vdW interactions. To do this, the original (unaltered) moleculegraph representations are used to produce playmol input.
        mol_list      =  [ moleculegraph.molecule(mol) for mol in self.mol_str ]
                    
        moldict       = {}

        # Get running atom numbers --> These are the numbers of all atoms in each molecule. Here the index of each atom needs to be 
        # lifted by the index of all preceeding atoms.
        add_atom      = [1] + [ sum(mol.atom_number for mol in mol_list[:(i+1)]) + 1 for i in range( len(mol_list[1:]) ) ]
        atom_numbers  = list( np.concatenate( [ mol.atom_numbers + add_atom[i] for i,mol in enumerate(mol_list) ] ) )
        
        # Get running bond numbers --> These are the corresponding atoms of each bond in each molecule. (In order to use the right atom index, the add_atom list is used)
        bond_numbers  = list( np.concatenate( [mol.bond_list + add_atom[i] for i,mol in enumerate(mol_list) if mol.bond_list.size > 0], axis=0 ) )

        # Get the force field type of each atom in each molecule.
        atom_names    = [j for sub in [molecule.atom_names for molecule in mol_list] for j in sub]

        # Get the names of each atom in each bond of each molecule. This is done to explain playmol which bond type they should use for this bond
        playmol_bond_names = list(np.concatenate( [ [ [mol.atom_names[i] for i in bl] for bl in mol.bond_list ] for mol in mol_list if mol.bond_list.size > 0], axis=0 ))

        # Playmol uses as atom input: atom_name force_field_type charge --> the atom_name is "force_field_type+atom_index"
        moldict["atoms"]   = list(zip( atom_numbers, atom_names, [self.ff_all[i]["charge"] for i,_ in enumerate(atom_names)] ) )

        # Playmol uses as bond input: atom_name atom_name --> the atom_name is "force_field_type+atom_index"
        moldict["bonds"]   = list(zip( bond_numbers, playmol_bond_names))

        # Provide the number of molecules per component, as well as the starting atom of each molecule (e.g.: molecule1 = {C1, C2, C3}, molecule2 = {C4, C5, C6} --> Provide C1 and C4 )
        molecule_indices   = [a-1 for a in add_atom]
        moldict["mol"]     = list( zip( self.nmol_list, [ str(moldict["atoms"][i][1])+str(moldict["atoms"][i][0]) for i in molecule_indices ] ) )

        # Add path to force field
        moldict["force_field"] = playmol_ff_path

        # Add path to xyz of one molecule of each component
        moldict["xyz"]         = xyz_paths

        # Add name of the final xyz file and log file
        moldict["final_xyz"]   = ".".join(os.path.basename(playmol_path).split(".")[:-1]) + ".xyz"
        moldict["final_log"]   = ".".join(os.path.basename(playmol_path).split(".")[:-1]) + ".log"


        ## Write playmol input file to build the system with specified number of molecules for each component ##

        with open(playmol_template) as file:
            template = Template(file.read())

        # Playmol template needs density in g/cm^3; rho given in kg/m^3 to convert in g/cm^3 divide by 1000
        rendered = template.render( rd   = moldict,
                                    rho  = str(self.density / 1000),
                                    seed = np.random.randint(1,1e6) )
        
        with open(playmol_path, "w") as fh:
            fh.write(rendered) 

        # Save current folder of notebook
        maindir = os.getcwd()

        # Move in the specified folder
        os.chdir( os.path.dirname(playmol_path) )

        # Execute playmol to build the system
        log = os.system( "%s -i %s"%( playmol_executeable, os.path.basename(playmol_path) ) )

        print(log)

        print( "\nDONE: %s -i %s\n"%( playmol_executeable, os.path.basename(playmol_path) ) )

        os.chdir( maindir)
        
        return 
    
    def prepare_lammps_data(self, nmol_list: List, densitiy: float, decoupling: bool=False):
        """
        Function that prepares the LAMMPS data file at a given density for a given force field and system. In the case decoupling is wanted, then the first component will be decoupled
        to all other mixture components.

        Args:
            nmol_list (list): List containing the number of molecules per component
            densitiy (float): Mass density of the component/mixture at that state [kg/m^3]
        """
        
        #### System specific settings ####

        # Variables defined here are used class wide 
        self.nmol_list      = nmol_list
        self.density        = densitiy
        self.decoupling     = decoupling

        # This is used to write the header of the data file for the 
        # Zip objects has to be refreshed for every system since its only possible to loop over them once
        self.renderdict["atom_paras"]    = zip(self.atom_numbers_ges, self.nonbonded)
        self.renderdict["bond_paras"]    = zip(self.bond_numbers_ges, self.bonds)
        self.renderdict["angle_paras"]   = zip(self.angle_numbers_ges, self.angles)
        self.renderdict["torsion_paras"] = zip(self.torsion_numbers_ges, self.torsions)

        # Total atoms in system (summation of the number of atoms of each component times the number of molecules of each component)
        self.total_number_of_atoms    = np.dot( self.number_of_atoms, nmol_list )

        # Total bonds in system 
        self.total_number_of_bonds    = np.dot( self.number_of_bonds, nmol_list )
        
        # Total angles in system 
        self.total_number_of_angles   = np.dot( self.number_of_angles, nmol_list) 
        
        # Total torsions in system 
        self.total_number_of_torsions = np.dot( self.number_of_torsions, nmol_list )


        ## Mass, mol, volume and box size of the system ##

        # Molar masses of each species [g/mol]
        Mol_masses = np.array( [ np.sum( [ a["mass"] for a in molecule.map_molecule( molecule.atom_names, self.ff["atoms"] ) ] ) for molecule in self.mol_list ] )

        # Account for mixture density --> in case of pure component this will not alter anything

        # mole fraction of mixture (== numberfraction)
        x = np.array( nmol_list ) / np.sum( nmol_list )

        # Average molar weight of mixture [g/mol]
        M_avg = np.dot( x, Mol_masses )

        # Total mole n = N/NA [mol] #
        n = np.sum( nmol_list ) / Avogadro

        # Total mass m = n*M [kg]
        mass = n * M_avg / 1000

        # Compute box volume V=m/rho and with it the box lenght L (in Angstrom) --> assuming orthogonal box
        # With mass (kg) and rho (kg/m^3 --> convert in g/A^3 necessary as lammps input)

        # Volume = mass / mass_density = mol / mol_density [A^3]
        volume = mass / self.density * 1e30

        boxlen = volume**(1/3) / 2

        box = [ -boxlen, boxlen ]

        self.renderdict["box_x"] = box
        self.renderdict["box_y"] = box
        self.renderdict["box_z"] = box
        
        return

    def write_lammps_data(self, xyz_path: str, data_template: str, data_path: str):
        """
        Function that generates a LAMMPS data file, containing bond, angle and torsion parameters, as well as all the coordinates etc.

        Args:
            xyz_path (str): Path to the xyz file for this system.
            data_template (str): Path to the jinja2 template for the LAMMPS data file.
            data_path (str): Path where the LAMMPS data file should be generated.
        """

        # Running counts of atoms, bonds, angles, and torsions.
        atom_count       = 0
        bond_count       = 0
        angle_count      = 0
        torsion_count    = 0

        # Introduce a count for the number of molecules of each component. This will be updated during the loop and at the end have the same numbers as the nmol_list.
        mol_count        = np.zeros(len(self.mol_list)).astype("int")

        # Lists containing all the lines for atoms, bonds, angles, and torsions that will be written into the #Atoms, #Bonds, #Angles and #Dihedrals section in the data file.
        lmp_atom_list    = []
        lmp_bond_list    = []
        lmp_angle_list   = []
        lmp_torsion_list = []

        # Read in the specified coordinates --> the atom names are also read in (these are the atom names given in playmol setup. Double check if they match the force field type you expect!)
        coordinates      = moleculegraph.funcs.read_xyz(xyz_path)
        
        # Get the number of atoms per component as list
        component_atom_numbers = [mol.atom_number for mol in self.mol_list] 

        for m,mol in enumerate(self.mol_list):

            # All the lists produced yet are one flat 1D list, containing the information of every atom of every component.
            # As we here loop through every component individually, the correct entries of all the lists need to be taken.
            # (E.g.: For the correct atoms_running_number of component 1, one needs to take the list entries from 0 to n° of atoms of component 1
            # The same needs to be done for component 2: from n° of atoms of component 1 to n° of atoms of component 1 + component 2, and so on...)
            # The same is true for the bonds, angles and torsions
            idx  = mol.atom_numbers + sum( mole.atom_number for mole in self.mol_list[:m] )
            idx1 = np.arange( len(mol.bond_keys) ) + sum( len(mole.bond_keys) for mole in self.mol_list[:m] )
            idx2 = np.arange( len(mol.angle_keys) ) + sum( len(mole.angle_keys) for mole in self.mol_list[:m] )
            idx3 = np.arange( len(mol.torsion_keys) ) + sum( len(mole.torsion_keys) for mole in self.mol_list[:m] )
            

            ## Now write LAMMPS input for every molecule of each component ##

            for mn in range(self.nmol_list[m]):
                
                # The atom index in the bond, angle, torsion list starts always at 1. Thus to write the correct atoms for each bond, angle, torsion one need to 
                # know the indcies of the atoms of the current molecule. To do so, add the dot product of all atom numbers and molecules of each component
                # to the atom index in the current bond, angle, torsion.
                add_atom_count = np.dot( mol_count, component_atom_numbers ).astype("int")

                # Define atoms
                for atomtype,ff_atom in zip( self.atoms_running_number[idx], self.ff_all[idx] ):
                    
                    atom_count +=1

                    # LAMMPS INPUT: total n° of atom in system, mol n° in system, atomtype, partial charges,coordinates
                    line = [ atom_count, sum(mol_count)+1, atomtype, ff_atom["charge"],*coordinates[atom_count-1]["xyz"], coordinates[atom_count-1]["atom"] ]

                    lmp_atom_list.append(line)


                # Define bonds 
                for bondtype,bond,bond_name in zip( self.bonds_running_number[idx1], self.bond_numbers[idx1], self.bond_names[idx1] ):

                    bond_count += 1

                    # Higher the indices of the bond to match the atoms of the current molecule.
                    dummy = bond + add_atom_count
      
                    # LAMMPS INPUT: total n° of bond in system, bond force field type, atom n° in this bond
                    line  = [ bond_count, bondtype, *dummy, " ".join(bond_name) ]

                    lmp_bond_list.append(line)

                # Define angles
                for angletype,angle,angle_name in zip (self.angles_running_number[idx2], self.angle_numbers[idx2], self.angle_names[idx2] ):

                    angle_count += 1

                    # Higher the indices of the angle to match the atoms of the current molecule.
                    dummy = angle + add_atom_count

                    # LAMMPS INPUT: total n° of angles in system, angle force field type, atom n° in this angle
                    line = [ angle_count, angletype, *dummy, " ".join(angle_name) ]

                    lmp_angle_list.append(line)

                # Define torsions
                for torsiontype,torsion,torsion_name in zip( self.torsions_running_number[idx3], self.torsion_numbers[idx3], self.torsion_names[idx3] ):

                    torsion_count += 1

                    # Higher the indices of the dihedral to match the atoms of the current molecule.
                    dummy = torsion + add_atom_count

                    # LAMMPS INPUT: total n° of torsions in system, torsion force field type, atom n° in this torsion
                    line = [ torsion_count, torsiontype, *dummy, " ".join(torsion_name) ]

                    lmp_torsion_list.append(line)

                # Increase the molecule count of the current component by one.
                mol_count[m] += 1

                
            self.renderdict["atoms"]    = lmp_atom_list
            self.renderdict["bonds"]    = lmp_bond_list
            self.renderdict["angles"]   = lmp_angle_list
            self.renderdict["torsions"] = lmp_torsion_list

            self.renderdict["atom_number"]    = atom_count
            self.renderdict["bond_number"]    = bond_count
            self.renderdict["angle_number"]   = angle_count
            self.renderdict["torsion_number"] = torsion_count
            
        ## Write to jinja2 template file to create LAMMPS data file ##

        os.makedirs( os.path.dirname(data_path), exist_ok=True )

        with open(data_template) as file_:
            template = Template(file_.read())
            
        rendered = template.render( rd = self.renderdict )

        with open(data_path, "w") as fh:
            fh.write(rendered)

        return


    def write_tabled_bond( self, torsion_pairs: List[List], table_path: str, table_template: str, evaluation_steps: int = 1000 ):
        """
        This function writes an input table for the LAMMPS bond style "table". The charge group approach is used, to include the 1.4-Mie interaction of the special torsion pair provided,
        as well as the evaluation of all coulombic interactions between local dipols. In the case two atoms of different local dipols are bonded, also evaluate the bonded potential.

        Args:
            torsion_pairs (List[List]): List with Lists containing the force field keys for the special torsion pair under investigation. Should match the lenght of components provided.
            table_path (str): Path where the table should be written to.
            table_template (str): Path to the Jinja2 template to write the table.
            steps_nonbonded (int, optional): Number of evaluation points for all special bonds. Defaults to 1000.
        """

        tabled_dict_list = []

        # As the gradient of the Coulomb and Mie potential is quiete high at low distances, use more 50% of the intermediates in the first 30% of the cut off range. 
        # And less intermediates for the distances higher than 30% of the cut off.
        n1              = int( 0.5 * evaluation_steps )
        n2              = int( evaluation_steps - n1 )

        # Get overall cutoff used in the simulation
        cut_off         = max( p["cut"] for _,p in self.ff["atoms"].items() )

        # Loop through every molecule and check if there are special bonds
        for mol, pair in zip( self.mol_list, torsion_pairs ):
            for sbi,sbk in zip( mol.bond_list, mol.bond_keys ):

                # Check if it is a special bond and if the special bond is already evaluted. In that case, skip the reevaluation.
                if "special" in sbk and not sbk in [ td["list"] for td in tabled_dict_list ]:

                    # Extract the force field keys of the atoms in this bond.
                    ff_keys = re.findall(r'\[(.*?)\]', sbk)[1:]

                    # Evalute the bonded interaction for every bonded special bond and the Coulomb interaction
                    if mol.get_distance(*sbi) == 1:

                        # K is given in Kcal/mol/Angstrom^2, r0 in Angstrom
                        r0, K    = self.ff["bonds"][moleculegraph.make_graph( ff_keys )]["p"]

                        # Get Coulomb parameter (charges and cut off)
                        charges  = [ self.ff["atoms"][atom_key]["charge"] for atom_key in ff_keys ]
                        
                        # Distances evaluated ±70% around r0.
                        r_eval   = np.linspace( 0.3*r0, 1.7*r0, evaluation_steps )
                        
                        # Compute bonded interaction 
                        u_bond, f_bond  = calc_bond( r_eval, r0, K, "energy"), calc_bond( r_eval, r0, K, "force")

                        # Compute Coulomb interaction
                        u_coulomb, f_coulomb = calc_coulomb( r_eval, *charges, "energy"), calc_coulomb( r_eval, *charges, "force")

                        # Get total energy and force
                        u_total, f_total     = u_bond + u_coulomb, f_bond + f_coulomb


                    # Evaluate the Mie interaction for the special 1-4 pair in the reparametrized torsion, as well as Coulomb.
                    elif mol.get_distance(*sbi) == 3 and all( p in ff_keys for p in pair ):

                        # Get Coulomb and vdW parameter (charges, epsilon, sigma and repulsive exponent)
                        charges  = [ self.ff["atoms"][atom_key]["charge"] for atom_key in ff_keys ]
                        epsilons = [ self.ff["atoms"][atom_key]["epsilon"] for atom_key in ff_keys ]
                        sigmas   = [ self.ff["atoms"][atom_key]["sigma"] for atom_key in ff_keys ]
                        ns       = [ self.ff["atoms"][atom_key]["m"] for atom_key in ff_keys ]

                        # Use mixing Lorentz-Berthelot mixing rule for sigma and epsilon, as well as an arithmetic mean for the repulsive exponent
                        n        = np.mean( ns )
                        sigma    = np.mean( sigmas )
                        epsilon  = np.sqrt(np.prod(epsilons))

                        # Evaluated distances 
                        r_eval   = np.concatenate( [np.linspace( 0.01*cut_off, 0.3*cut_off, n1), np.linspace( 0.301*cut_off, cut_off, n2)] )
                        
                        # Compute Mie interaction up to the cut off radius
                        u_mie, f_mie         = calc_mie( r_eval, sigma, epsilon, n, 6, "energy"), calc_mie( r_eval, sigma, epsilon, n, 6, "force")

                        # Compute Coulomb interaction up to the cut off radius. 
                        u_coulomb, f_coulomb = calc_coulomb( r_eval, *charges, "energy"), calc_coulomb( r_eval, *charges, "force")

                        # Get total energy and force
                        u_total, f_total     = u_mie + u_coulomb, f_mie + f_coulomb
                    

                    # Evaluate only the Coulomb interaction
                    else:
                        
                        # Get Coulomb parameter (charges and cut off)
                        charges  = [ self.ff["atoms"][atom_key]["charge"] for atom_key in ff_keys ]

                        # Evaluated distances 
                        r_eval   = np.concatenate( [np.linspace( 0.01*cut_off, 0.3*cut_off, n1), np.linspace( 0.301*cut_off, cut_off, n2)] )

                        # Compute Coulomb interaction up to the cut off radius. 
                        u_coulomb, f_coulomb = calc_coulomb( r_eval, *charges, "energy"), calc_coulomb( r_eval, *charges, "force")

                        # Get total energy and force
                        u_total, f_total     = u_coulomb, f_coulomb

                    # Save the evaluated distance, energy and force into the table dict
                    tabled_dict_list.append( { "list": sbk, 
                                               "N": len(r_eval), 
                                               "p": list( zip( range(1, evaluation_steps + 1), r_eval, u_total, f_total ) )
                                             } )
  
        # Write the table using Jinja2 template
        with open(table_template) as file_: 
            template = Template(file_.read())

        rendered = template.render( rd = tabled_dict_list )

        # Create folder (if necessary) where table should be writen to
        os.makedirs( os.path.dirname(table_path), exist_ok=True )

        with open(table_path, "w") as fh:
            fh.write(rendered) 
        
        return

    def prepare_lammps_input(self, timestep: int=1, sample_frequency: int=20, sample_number: int=50, 
                             pair_style: str="hybrid/overlay mie/cut 14 coul/long 14", mixing_rule: str="arithmetic", 
                             tail_correction: bool=True, sb_dict: Dict={"vdw":[0,0,0],"coulomb":[0,0,0]}, 
                             shake_dict: Dict={"atoms":[],"bonds":[],"angles":[]}, n_eval: int=1000 ):
        """
        This function initialize the needed input for LAMMPS. Here special bonds to scale 1-2, 1-3, and 1-4 pair vdW / Coulomb interactions can be defined, as well as possible force field types for the
        shake algorithm. Furthermore, this function defines the types of bonds, angles, and dihedrals used, as well as the pair style.

        Args:
            timestep (int, optional): Timestep in fs. Defaults to 1.
            sample_frequency (int, optional): Frequency fix ave/time output is computed. Defaults to 200.
            sample_number (int, optional): Number of samples that are averaged in the fix ave/time output. The final fix output will be every sample_frequency*sample_number times. Defaults to 5.
            pair_style (str, optional): Pair style that should be utilized. Defaults to hybrid/overlay mie/cut 14 coul/cut 14.
            mixing_rule (str, optional): Which mixing rule should be utilized, LAMMPS options are "geometric, arithmetic, and sixthpower". Defaults to arithmetic.
            tail_correction (bool, optional): If tail correction should be applied for vdW interactions. Defaults to True.
            sb_dict (dict, optional): Dictionary containing the special bonds coefficients to scale 1-2, 1-3, and 1-4 pair vdW / Coulomb interactions.
                                    The keys are "vdw" and "coulomb", defining the type of interaction that is scaled. Each key has a list with three entries,
                                    for each pair interaction, 1-2, 1-3, 1-4, respectively. Defaults to {"vdw":[0,0,0],"coulomb":[0,0,0]}.
            shake_dict (dict, optional): Keys for atoms, bonds or angles that should be constrained using the SHAKE algorithm. Input arguments are the force field types, which will be mapped back to the
                                        corresponding force field index. Defaults to {"atoms":[],"bonds":[],"angles":[]}.
            n_eval (int, optional): If tabled bond potentials are used, how many node points should be utilized for the spline interpolation.
        """
        
        
        # Settings dictionary containing all necessary information for the simulation
        self.settings = {}

        ## Define general simulation settings ##

        self.settings["timestep"]         = int(timestep)
        self.settings["sample_frequency"] = int(sample_frequency)
        self.settings["sample_number"]    = int(sample_number)

        ## Define used pair, bond, angle, and dihedrals types ##

        self.settings["style"] = {}
        
        # If several styles are utilized, add "hybrid" infront spline %d"%n_eval, 
        bond_styles     = np.unique( [a["style"] + f" spline {n_eval}" if a["style"] == "table" else a["style"] for a in self.bonds ] ).tolist()
        angle_styles    = np.unique( [a["style"] for a in self.angles] ).tolist()
        dihedral_styles = np.unique( [a["style"] for a in self.torsions] ).tolist()

        if len(bond_styles) >1: bond_styles = ["hybrid"] + bond_styles
        if len(angle_styles) >1: angle_styles = ["hybrid"] + angle_styles
        if len(dihedral_styles) >1: dihedral_styles = ["hybrid"] + dihedral_styles

        self.settings["style"]["pair_style"] = pair_style
        self.settings["style"]["bond"]       = bond_styles
        self.settings["style"]["angle"]      = angle_styles
        self.settings["style"]["dihedral"]   = dihedral_styles

        # Tail corrections
        self.settings["style"]["tail"]       = "yes" if tail_correction else "no"

        # Mixing rule
        self.settings["style"]["mixing"]     = mixing_rule

        # Charged system
        self.settings["style"]["uncharged"] = all( [charge==0 for charge in np.unique([p["charge"] for p in self.nonbonded])] )
        if "coul" in  pair_style and self.settings["style"]["uncharged"]: 
            raise KeyError("\n!!! Coulomb pair style is choosen, eventhough system is uncharged !!!\n")

        
        ## Define special bonds and SHAKE algorithm ##

        # Special bonds for LAMMPS. This is used to scale all 1-2, 1-3, 1-4 interactions if wanted.
        self.settings["sp_bond"] = sb_dict

        # Shake algorithm
        # Define used input atom, bond and angle index
        atom_paras  = zip(self.atom_numbers_ges, self.nonbonded)
        bond_paras  = zip(self.bond_numbers_ges, self.bonds)
        angle_paras = zip(self.angle_numbers_ges, self.angles)

        # Search the index of the given force field types
        key_at  = [item for sublist in [[a[0] for a in atom_paras if a_key == a[1]["name"]] for a_key in shake_dict["atoms"]] for item in sublist]
        key_b   = [item for sublist in [[a[0] for a in bond_paras if a_key == a[1]["list"]] for a_key in shake_dict["bonds"]] for item in sublist]
        key_an  = [item for sublist in [[a[0] for a in angle_paras if a_key == a[1]["list"]] for a_key in shake_dict["angles"]] for item in sublist]

        self.settings["shake"] = {"t":key_at, "b":key_b, "a":key_an}

        return

    def write_lammps_input(self, input_path: str, template_path: str, data_file: str, temperature: float, pressure: float=0.0, equilibration_time: int=5e6, production_time: int=3e6,
                           restart: bool=False, restart_file = "equil.restart", lambda_vdw: float=1.0, lambda_coulomb: float=1.0,
                           free_energy_method: str="TI", dlambda: List=[-0.001,0.001], free_energy_output_files: List=["fep.fep01","fep.fep12"]):
        """
        Function that writes a LAMMPS input file using Jinja2 template

        Args:
            input_path (str): Path where the input file will be created.
            template_path (str): Path to Jinja2 template.
            data_file (str): Path to data file (Relative path specified from input file location)
            temperature (float): Temperature of the simulation system. Unit is Kelvin.
            pressure (float, optional): Pressure of the simulation system. Unit is bar. If a NVT ensemble is wanted, use 0.0 as pressure. Defaults to 0.0.
            equilibration_time (int, optional): Equilibration time of the system. Defaults to 5e6.
            production_time (int, optional): Production time of the system. All fix commands will output their values in this time. Defaults to 3e6.
            restart (bool, optional): If simulation should start from a restart rather than from a data file. Defaults to False.
            restart_file (str, optional): File name of restart file, either for writing a restart file it self, or for reading a restart file in. Defaults to "equil.restart".
        """

        ## Define general settings ##

        # Set simulation time
        self.settings["equiltime"] = int(equilibration_time)
        self.settings["runtime"]   = int(production_time)

        # Set the ensemble (NPT if pressure and temperature is defined, else NVT)
        self.settings["ensemble"] = "NPT" if pressure else "NVT"

        # Set thermodynamic conditions (convert bar into atm for LAMMPS)
        self.settings["temperature"] = temperature
        self.settings["pressure"]    = pressure / 1.01325

        # Set restart / data / sampling file options
        self.settings["restart"]      = int(restart)
        self.settings["restart_file"] = restart_file
        self.settings["data_file"]    = data_file


        ## Define pair interactions ##

        # Van der Waals
        pair_interactions = []

        # Define the atom numbers of the coupling molecule (this is always component one)
        atom_list_coulped_molecule = np.arange( len(self.mol_list[0].unique_atom_keys) ) + 1

        for i,iatom in zip(self.atom_numbers_ges, self.nonbonded):
            for j,jatom in zip(self.atom_numbers_ges[i-1:], self.nonbonded[i-1:]):

                # If decoupling is wanted, no mixture rule can be utilized --> the i j pair interactions need to defined in here
                # Otherwise only the i i pair interactions need to be defined, and thus any i j pair can be skipped.
                if i != j and not self.decoupling:
                    continue
                
                # If decoupling is wanted, the intramolecular interactions of the coupling molecule need to be activated,
                # while the intermolecular interactions need to be scaled. This is done using a lambda_ij.
                # Activated: lambda_ij = 1.0, coupled: 0.0 < lambda_ij < 1.0, deactiavted: lambda_ij = 0.0
                if all( np.isin( np.array([i,j]), atom_list_coulped_molecule) ):
                    lambda_ij = 1.0 
                elif (i in atom_list_coulped_molecule and not j in atom_list_coulped_molecule) or (not i in atom_list_coulped_molecule and j in atom_list_coulped_molecule):
                    lambda_ij = lambda_vdw
                else:
                    lambda_ij = None

                name_ij   = "%s  %s"%( iatom["name"], jatom["name"] ) 

                if self.settings["style"]["mixing"] == "arithmetic": 
                    sigma_ij   = ( iatom["sigma"] + jatom["sigma"] ) / 2
                    epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

                elif self.settings["style"]["mixing"] == "geometric":
                    sigma_ij   = np.sqrt( iatom["sigma"] * jatom["sigma"] )
                    epsilon_ij = np.sqrt( iatom["epsilon"] * jatom["epsilon"] )

                elif self.settings["style"]["mixing"] == "sixthpower": 
                    sigma_ij   = ( 0.5 * ( iatom["sigma"]**6 + jatom["sigma"]**6 ) )**( 1 / 6 ) 
                    epsilon_ij = 2 * np.sqrt( iatom["epsilon"] * jatom["epsilon"] ) * iatom["sigma"]**3 * jatom["sigma"]**3 / ( iatom["sigma"]**6 + jatom["sigma"]**6 )

                n_ij  = ( iatom["m"] + jatom["m"] ) / 2
                
                pair_interactions.append( { "i": i, "j": j, "sigma": round( sigma_ij, 4 ) , "epsilon": round( epsilon_ij, 4 ),  "m": n_ij, "name": name_ij, "lambda_vdw": lambda_ij } ) 

        self.settings["style"]["pairs"] = pair_interactions


        ## Save free energy (coupling) related settings ##

        self.settings["free_energy"]                        = {}

        # Save the atom types of the coupled molecule
        self.settings["free_energy"]["coupled_atom_list"]   = atom_list_coulped_molecule

        # Coulomb
        # If coupling is not wanted, long range coulomb interactions are evaluated for every charged atom in the system. 
        # If coupling is wanted, the charges of all all atoms in the coupled molecule need to be scaled accordingly (with lambda_coulomb).
        # In case vdW interactions are investiagted, scale the charges to closely 0 (1e-9), and overlay a 2nd Coulomb potential to maintain unaltered intramolecular Coulomb interactions.
        # In case Coulomb interactions are investiaged, scale the charges accordignly, and also overlay the 2nd Coulomb potential.
        self.settings["free_energy"]["charge_list"]         = [ iatom["charge"] * lambda_coulomb for iatom in np.array( self.nonbonded )[ atom_list_coulped_molecule-1 ] ]

        # Define the overlay between all atoms of the coupled molecule (defined in https://doi.org/10.1007/s10822-020-00303-3)
        self.settings["free_energy"]["lamda_overlay"]       = ( 1 - lambda_coulomb**2 ) / lambda_coulomb**2

        # Define if vdW or Coulomb interactions is decoupled.
        self.settings["free_energy"]["couple_lambda"]       = lambda_vdw if np.isclose( lambda_coulomb, 0 ) else lambda_coulomb
        self.settings["free_energy"]["couple_interaction"]  = "vdw" if np.isclose( lambda_coulomb, 0 ) else "coulomb"

        # Define free energy method
        self.settings["free_energy"]["method"]              = free_energy_method

        # Define the lambda perpurtation that should be performed (backwards and forwards)
        self.settings["free_energy"]["lambda_perturbation"] = dlambda

        # Define output files for free energy computation
        self.settings["free_energy"]["output_files"]        = free_energy_output_files

        # Write the input file using Jinja2 template
        with open(template_path) as file_: 
            template = Template(file_.read())

        rendered = template.render( set = self.settings )

        # Create folder (if necessary) where the input file should be writen to
        os.makedirs( os.path.dirname(input_path), exist_ok=True )

        with open(input_path, "w") as fh:
            fh.write(rendered) 
        
        return