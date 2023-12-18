<h1 align="center">
  TAMie force field
</h1>
<p align="center">This repository enables users to perform molecular dynamics simulations utilizing LAMMPS with any force field. The process begins with a SMILES and graph representation of each component, where PLAYMOL constructs initial systems at specific densities. The moleculegraph software and supplementary Python code are then used to generate a LAMMPS data and input file via jinja2 templates. </p>


## 🚀 Getting Started

Get started by running the following command to install:

1. moleculegraph
```
git clone https://github.com/maxfleck/moleculegraph
cd moleculegraph
pip install -I .
```
2. pyLAMMPS
```
git clone https://github.com/samirdarouich/pyLAMMPS.git
cd pyLAMMPS
pip install -I .
```
3. PLAYMOL
```
git clone git clone https://github.com/atoms-ufrj/playmol
cd playmol
make
```


## 🐍 Example program

The following example will demonstrate how to setup MD simulations (this is also demonstrated in the example.ipynb). Essentially, the workflow can be summarized as follows:
1. Get intial molecule coordinates using SMILES and PubChem and provide a graph representation.
2. Use PLAYMOL to construct a system of any mixture.
3. Use the moleculegraph software and pyLAMMPS tools to generate LAMMPS data and input files via jinja2 templates.

## 1. Define general settings ##

1. Define path to force field toml (in a moleculegraph understandable format),
2. Define name, SMILES, and graph strings of the molecules under investigation.
3. Call the pyLAMMPS LAMMPS_input class

```python
# 1: Path tho force field toml
force_field_path     = "force-fields/forcefield_UA_TAMie_alkanes.toml"

# 2: Names, SMILES, and graphs of molecules (further examples on constructing molecule graphs available at https://github.com/maxfleck/moleculegraph)

system_name          = "mixture_butane_hexane"

molecule_name1       = "butane"
molecule_graph1      = "[CH3_alkane][CH2_alkane][CH2_alkane][CH3_alkane]"
molecule_smiles1     = "CCCC"

molecule_name2       = "hexane"
molecule_graph2      = "[CH3_alkane][CH2_alkane][CH2_alkane][CH2_alkane][CH2_alkane][CH3_alkane]"
molecule_smiles2     = "CCCCCC"

molecule_name_list   = [ molecule_name1, molecule_name2 ]
molecule_graph_list  = [ molecule_graph1, molecule_graph2 ]
molecule_smiles_list = [ molecule_smiles1, molecule_smiles2 ]

# Template paths
# xyz templates
template_xyz         = "templates/template_write_xyz.xyz"

# PLAYMOL templates
playmol_force_field_template = "templates/template_playmol_forcefield.playmol"
playmol_input_template       = "templates/template_playmol_input.mol"

# LAMMPS templates
LAMMPS_data_template         = "templates/template_lammps_data.data"
LAMMPS_input_template        = "templates/template_lammps_input.in"
LAMMPS_ff_template           = "templates/template_lammps_ff.params"

# System paths
# Path to working folder
working_folder       = f"example/{system_name}"

# Define output path for final xyz files
xyz_destinations     = [ f"{working_folder}/%s.xyz"%name for name in molecule_name_list ]

# Get the single molecule coordinates for each component
get_molecule_coordinates( molecule_name_list = molecule_name_list, molecule_graph_list = molecule_graph_list, molecule_smiles_list = molecule_smiles_list,
                          xyz_destinations = xyz_destinations, template_xyz = template_xyz, verbose = False )

# 3: Call the LAMMPS input class
LAMMPS_class        = LAMMPS_input( mol_str = molecule_graph_list, ff_path = force_field_path )
```

## 2. Write system size independent files ##

1. The PLAYMOL force field file, which is used to build the initial configurations

```python
# 1: PLAYMOL force field file
playmol_force_field_destination = f"{working_folder}/playmol_ff.playmol"

LAMMPS_class.prepare_playmol_input( playmol_template = playmol_force_field_template, playmol_ff_path = playmol_force_field_destination )
```

## 3. Write system size dependent files ##

1. Define general (thermodynamic) settings for each system.
2. Write PLAYMOL input file and execute it to build the system (if wanted).
3. Write LAMMPS data file using the from PLAYMOL generated xyz file
4. Write LAMMPS input file for each system

```python
# 1: Define general settings for each system
# Temperatures [K], pressures [bar] (if wanted, otherwise use 0.0 for a NVT ensemble) and initial denisties [kg/m^3] for each system.

temperatures     = [ 343.0 ]
pressures        = [ 4.311 ]
densities        = [ 573.282 ]

# Define the total number of molecules in the simulation and the compositions per statepoint
molecule_number  = 500
compositions     = [ [ 0.5, 0.5 ] ]

# Simulation path
simulation_path  = f"{working_folder}/sim_%d"

# Define if PLAYMOL should be executed and further settings
build_playmol    = True

# Define additional input (as list of dictionaries) that should be passed to the settings dictionary used in the jinja2 template (will be accessed with: settings.*)
external_input   = [ {} ]

# Define additional functions that can be parsed to the LAMMPS input class. They can operate with class atributes, if the input arguments have the same name as the class argument.
# Furthermore define possible external function inputs that are also passed to the functions (per function define a new dictionary with inputs).
# In this example the write_pair_ff is a function that writes the van der Waals pair interactions. external arguments can be, that the force field is writen to an external file instead within the 
# input file.
external_functions = [ write_pair_ff ]
lammps_ff_path     = f"{working_folder}/lammps_ff.params"

for i, (temp, press, dens, composition) in enumerate( zip( temperatures, pressures, densities, compositions ) ):

    # Create the simulation folder (if not already done)
    os.makedirs( simulation_path%i, exist_ok = True )

    # Prepare LAMMPS with molecules numbers and density of the system
    # Get the molecule numbers according to the mixture composition (utilize closing condition for first component) 
    remaining_numbers = ( np.array(composition[1:]) * molecule_number ).astype("int")
    molecule_numbers  = [ molecule_number - sum(remaining_numbers), *(remaining_numbers if sum(remaining_numbers) > 0 else []) ]

    LAMMPS_class.prepare_lammps_data (nmol_list = molecule_numbers, density = dens )


    # 2: Write PLAYMOL input and execute (if wanted.)
    if build_playmol:
        playmol_input_destination = simulation_path%i + f"/build/{system_name}_{i}.mol"
        playmol_relative_ff_path  = os.path.relpath(playmol_force_field_destination, os.path.dirname(playmol_input_destination))
        playmol_relative_xyz_path = [ os.path.relpath(xyz, os.path.dirname(playmol_input_destination)) for xyz in xyz_destinations ]

        LAMMPS_class.write_playmol_input( playmol_template = playmol_input_template, playmol_path = playmol_input_destination, 
                                          playmol_ff_path = playmol_relative_ff_path, xyz_paths = playmol_relative_xyz_path )


    # 3: Write LAMMPS data file from generated xyz file
    system_xyz              = playmol_input_destination.replace( ".mol", ".xyz" )
    LAMMPS_data_destination = simulation_path%i + "/build/lammps.data"

    LAMMPS_class.write_lammps_data( xyz_path = system_xyz, data_template = LAMMPS_data_template, data_path = LAMMPS_data_destination )


    # 4: Write LAMMPS input file
    LAMMPS_input_destination  = simulation_path%i + "/simulation/lammps.input"
    relative_LAMMPS_data_path = os.path.relpath(LAMMPS_data_destination, os.path.dirname(LAMMPS_input_destination))
   
    LAMMPS_class.prepare_lammps_input( pair_style = "hybrid/overlay mie/cut 14", mixing_rule = "arithmetic", sb_dict = {"vdw":[0,0,0],"coulomb":[0,0,0]},
                                       external_input = external_input )


    # Write the input file, also pass external functions and arguments if wanted
    external_function_input = [ { "ff_template": LAMMPS_ff_template, "lammps_ff_path": lammps_ff_path, 
                                  "relative_lammps_ff_path": os.path.relpath(lammps_ff_path, os.path.dirname(LAMMPS_input_destination)) } ]

    LAMMPS_class.write_lammps_input( input_path = LAMMPS_input_destination, template_path = LAMMPS_input_template, data_file = relative_LAMMPS_data_path,
                                     temperature = temp, pressure = press, equilibration_time = 4e6, production_time = 2e6,
                                     external_functions = external_functions, external_function_input = external_function_input )
```

## 🚑 Help

Help will arrive soon ...

## 👫 Authors

Samir Darouich - University of Stuttgart, Maximilian Fleck - University of Stuttgart

## 📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details
