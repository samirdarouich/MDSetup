#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p single
#SBATCH -J build_system
#SBATCH -o {{folder}}/build_log.o
#SBATCH --mem-per-cpu=2000

# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2023.3

# Define the main working path 
WORKING_PATH={{folder}}
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

# Use GROMACS to build the box

{%- for coord, mol, nmol in coord_mol_no %}

# Add molecule: {{ mol }}
{%- if loop.index0 == 0 and not initial_system %}
gmx_mpi insert-molecules -ci {{ coord }} -nmol {{ nmol }} -box {{ box_lengths.x[0]|abs + box_lengths.x[1]|abs box_lengths.y[0]|abs + box_lengths.y[1]|abs box_lengths.z[0]|abs + box_lengths.z[1]|abs }} -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- elif loop.index0 == 0 %}
gmx_mpi insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f {{ initial_system }} -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- else %} 
gmx_mpi insert-molecules -ci {{ coord }} -nmol {{ nmol }} -f temp{{ loop.index0-1 }}.gro -try {{ n_try }} -o temp{{ loop.index0 }}.gro
{%- endif %}

{%- endfor %}

# Correctly rename the final configuration
mv temp{{ coord_mol_no | length - 1 }}.gro {{output_coord}}

# Delete old .gro files
rm -f \#*.gro.*#


# End
echo "Ending. Job completed."