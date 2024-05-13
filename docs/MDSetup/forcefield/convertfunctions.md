Module MDSetup.forcefield.convertfunctions
==========================================

Functions
---------

    
`convert_harmonic_angle(source: str, destination: str, angle: Dict[str, str | float | int])`
:   GROMACS uses theta0 and then K, LAMMPS uses K then theta0
    GOMACS uses 1/2*K and LAMMPS K, hence include factor 1/2 in LAMMPS
    GROMACS uses kJ/nm^2 and LAMMPS kcal/A^2

    
`convert_harmonic_bond(source: str, destination: str, bond: Dict[str, str | float | int])`
:   GROMACS uses b0 and then K, LAMMPS uses K then b0
    GOMACS uses 1/2*K and LAMMPS K, hence include factor 1/2 in LAMMPS
    GROMACS uses kJ/nm^2 and LAMMPS kcal/A^2
    GROMACS uses b0 in nm and LAMMPS in A

    
`convert_harmonic_dihedral(source: str, destination: str, dihedral: Dict[str, str | float | int])`
:   GROMACS phi, k, n, LAMMPS, k, d, n

    
`convert_opls_dihedral(source: str, destination: str, dihedral: Dict[str, str | float | int])`
:   

    
`source_destination_error(source, destination)`
: