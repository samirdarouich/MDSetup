Module MDSetup.forcefield.converter
===================================

Functions
---------

    
`convert_angle(source: str, destination: str, angle: Dict[str, str | float | int])`
:   

    
`convert_atom(source: str, destination: str, atom: Dict[str, str | float | int])`
:   

    
`convert_bond(source: str, destination: str, bond: Dict[str, str | float | int])`
:   

    
`convert_dihedral(source: str, destination: str, dihedral: Dict[str, str | float | int])`
:   

    
`convert_force_field(force_field_path: str, output_path: str)`
:   

    
`extract_force_field_gromacs(itp_files: List[str], top_file: str, output_path: str)`
:   

Classes
-------

`LMP2GRO(data_file: str, ff_file: str, map_dict: Dict[str, Dict[str, str]])`
:   

    ### Methods

    `convert(self, itp_template: str, gro_template: str, top_template: str, destination: str)`
    :

    `convert_data_in_gro(self, gro_template: str, destination: str)`
    :

    `convert_data_in_itp(self, itp_template: str, destination: str)`
    :

    `convert_ff_in_top(self, top_template: str, destination: str)`
    :