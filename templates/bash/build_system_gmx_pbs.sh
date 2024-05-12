#!/bin/bash
#PBS -q tiny
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -N build_system
#PBS -o {{folder}}/build_log.o 
#PBS -e {{folder}}/build_log.e 
#PBS -l mem=3000mb

# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2022.4

# Go to working folder
cd {{folder}}

# Use GROMACS to build the box

{%- for coord, mol, nmol in coord_mol_no %}

# Add molecule: {{ mol }}
{%- if loop.index0 == 0 and not initial_system %}
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -box {{ box_lengths|join(" ") }} -o temp{{ loop.index0 }}.gro
{%- elif loop.index0 == 0 %}
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f {{ initial_system }} -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- else %} 
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f temp{{ loop.index0-1 }}.gro -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- endif %}

{%- endfor %}

# Correctly rename the final configuration
mv temp{{ coord_mol_no | length - 1 }}.gro init_conf.gro

# Delete old .gro files
rm -f \#*.gro.*#