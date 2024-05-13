#!/bin/bash
#PBS -q long
#PBS -l nodes=1:ppn=28
#PBS -l walltime=20:00:00
#PBS -N anhydrite_298_1
#PBS -o /Users/samir/Documents/Coding_libaries/MDSetup/example/lammps/mechanical_properties/anhydrite/equilibration/temp_298.1_pres_1.0/copy_2/LOG.o 
#PBS -e /Users/samir/Documents/Coding_libaries/MDSetup/example/lammps/mechanical_properties/anhydrite/equilibration/temp_298.1_pres_1.0/copy_2/LOG.e 
#PBS -l mem=3000mb

module purge
module load mpi/openmpi/3.1-gnu-9.2

# Define the main working path 
WORKING_PATH=/Users/samir/Documents/Coding_libaries/MDSetup/example/lammps/mechanical_properties/anhydrite/equilibration/temp_298.1_pres_1.0/copy_2
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"


# Define the names of each simulation step taken. The folder as well as the output files will be named like this


################################# 
#       00_em       #
#################################
echo ""
echo "Starting ensemble: 00_em"
echo ""

mkdir -p 00_em
cd 00_em

mpirun --bind-to core --map-by core -report-bindings lmp -i em.input

echo "Completed ensemble: 00_em"

cd ../
sleep 10



################################# 
#       01_npt       #
#################################
echo ""
echo "Starting ensemble: 01_npt"
echo ""

mkdir -p 01_npt
cd 01_npt

mpirun --bind-to core --map-by core -report-bindings lmp -i npt.input

echo "Completed ensemble: 01_npt"

cd ../
sleep 10




# End
echo "Ending. Job completed."