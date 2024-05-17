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

# Define the main working path 
WORKING_PATH={{folder}}
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

# Use GROMACS to build the box

{%- for coord, mol, nmol in coord_mol_no %}

# Add molecule: {{ mol }}
{%- if loop.index0 == 0 and not initial_system %}
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -box {{ box_lengths.x[0]|abs + box_lengths.x[1]|abs box_lengths.y[0]|abs + box_lengths.y[1]|abs box_lengths.z[0]|abs + box_lengths.z[1]|abs }} -o temp{{ loop.index0 }}.gro
{%- elif loop.index0 == 0 %}
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f {{ initial_system }} -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- else %} 
gmx insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f temp{{ loop.index0-1 }}.gro -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- endif %}

{%- endfor %}

# Correctly rename the final configuration
mv temp{{ coord_mol_no | length - 1 }}.gro {{output_coord}}

# Delete old .gro files
rm -f \#*.gro.*#

# End
echo "Ending. Job completed."