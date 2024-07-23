#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 01:00:00
#SBATCH -p single
#SBATCH -J build_system
#SBATCH -o {{folder}}/build_log.o
#SBATCH --mem-per-cpu=500

module purge
module load compiler/gnu/10.2
module load mpi/openmpi/4.1

# Define the main working path 
WORKING_PATH={{folder}}
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

mpirun --bind-to core --map-by core -report-bindings lmp -i {{build_input_file}}

# End
echo "Ending. Job completed."
