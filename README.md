<h1 align="center">
  pyLAMMPS
</h1>
<p align="center">This repository enables users to perform molecular dynamics simulations utilizing LAMMPS. It can either start from given parameter and data file, or create them itself by using moleculegraph, PLAYMOL and supplementary Python code. YAML files are utilized to parse simulation settings, such as ensemble definitions, sampling properties and system specific input. </p>


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

# pyLAMMPS

This module enables users to perform molecular dynamics simulations utilizing LAMMPS with any force field provided as toml/json file. 
There is the possiblity to provide data and LAMMPS compatible parameter files, or to build a system and write all necessary input using pyLAMMPS, PLAYMOL, and moleculegraph.

1) Read in the YAML files to define the system and simulation/sampling settings.

```python
lammps_setup = LAMMPS_setup( system_setup = "input/setup.yaml", 
                             simulation_default = "input/defaults.yaml",
                             simulation_ensemble = "input/ensemble.yaml",
                             simulation_sampling = "input/sampling.yaml",
                             submission_command = "qsub"
                            )
```

## Setting up a simulation pipeline

In this section the possibility to setup a simulation folder, along with a simulation pipeline using several ensembles, is provided.

1) Setup simulation and build initial system (if not provided)

```python
# Define the simulation folder
simulation_folder = "md_thermo"

# Define the ensembles that should be simulated (definition what each ensemble means is provided in yaml file)
ensembles = [ "em", "npt" ] 

# Define the simulation time per ensemble in nano seconds (for em the number of iterations is provided in the ensemble yaml)
simulation_times = [ 0, 10.0 ]

# Define initial systems, in case the simulation should be continued from a prior simulation.
# In that case, provide one initial structure for each temperature & pressure state.
# If the simulation should start from an initial configuration, provide an empty list.
initial_systems = [ "/home/st/st_st/st_ac137577/workspace/software/pyLAMMPS/example/butane_hexane/md_thermo/temp_343_pres_4/build/system.data" ]

# Define if there is already a force field file
ff_file = ""

# Provide kwargs that should be passed into the input template directly
input_kwargs = {  }

# Define number of copies
copies = 2

# Define if the inital system should build locally or with the cluster
on_cluster = False

# Define the starting number for the first ensemble ( 0{off_set}_ensemble )
off_set    = 0

lammps_setup.prepare_simulation( folder_name = simulation_folder, ensembles = ensembles, simulation_times = simulation_times,
                                 initial_systems = initial_systems, input_kwargs = input_kwargs, copies = copies,
                                 ff_file = ff_file, on_cluster = on_cluster,  off_set = off_set )
```

2) Submit jobs to cluster

```python
# Submit the simulations
lammps_setup.submit_simulation()
```

## 🚑 Help

Help will arrive soon ...

## 👫 Authors

Samir Darouich - University of Stuttgart, Maximilian Fleck - University of Stuttgart

## 📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details
