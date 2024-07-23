#!/bin/bash
#PBS -q tiny
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -N build_system
#PBS -o gromacs/build_log.o 
#PBS -e gromacs/build_log.e 
#PBS -l mem=3000mb

# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2022.4

# Define the main working path 
WORKING_PATH=gromacs
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

# Use GROMACS to build the box

# Add molecule: propane
gmx insert-molecules -ci propane.gro -nmol 100 -box 3.384 3.384 3.384 -o temp0.gro

# Add molecule: ethene 
gmx insert-molecules -ci ethene.gro -nmol 300 -f temp0.gro -try 10000 -o temp1.gro

# Correctly rename the final configuration
mv temp1.gro init_conf.gro

# Delete old .gro files
rm -f \#*.gro.*#

# End
echo "Ending. Job completed."