#!/bin/bash
#PBS -q long
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N build_system
#PBS -o lammps/build_log.o 
#PBS -e lammps/build_log.e 
#PBS -l mem=3000mb

module purge
module load mpi/openmpi/3.1-gnu-9.2

# Define the main working path 
WORKING_PATH=lammps
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

mpirun --bind-to core --map-by core -report-bindings lmp -i build_box.in

# End
echo "Ending. Job completed."