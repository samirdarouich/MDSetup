Module MDSetup.forcefield.forcefield
====================================

Classes
-------

`forcefield(smiles: List[str], force_field_paths: List[str])`
:   This class writes a force field input for arbitrary mixtures using moleculegraph. This input can be used for GROMACS and LAMMPS.

    ### Methods

    `write_molecule_files(self, molecule_template: str, molecule_path: str, residues: List[str], **kwargs)`
    :   Function that generates LAMMPS molecule files for all molecules.
        
        Parameters:
         - molecule_template (str): Path to the jinja2 template for the molecule file.
         - molecule_path (str): Path where the molecule files should be generated.
        
        Keyword Args:
         - gro_template (str): Template to write gro file for GROMACS
         - nrexcl (List[int]): A list of integers defining the number of bonds to exclude nonbonded interactions for each molecule for GROMACS.
        
        Return:
            - mol_files (List[str]): List with absolute paths of created mol files

    `write_topology_file(self, topology_template: str, topology_path: str, system_name: str, **kwargs)`
    :   Function that generates topology for the system.
        
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