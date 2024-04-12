#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00
#SBATCH -p single
#SBATCH -J {{job_name}}
#SBATCH -o {{log_path}}.o
#SBATCH --mem-per-cpu=150

module purge
module load compiler/gnu/10.2
module load mpi/openmpi/4.1

# Define the main working path 
WORKING_PATH={{working_path}}
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"


# Define the names of each simulation step taken. The folder as well as the output files will be named like this
{% for ensemble_name, ensemble in ensembles.items() %}

################################# 
#       {{ensemble_name}}       #
#################################
echo ""
echo "Starting ensemble: {{ensemble_name}}"
echo ""

mkdir -p {{ensemble_name}}
cd {{ensemble_name}}

mpirun --bind-to core --map-by core -report-bindings lmp -i {{ensemble.mdrun}}

echo "Completed ensemble: {{ensemble_name}}"

cd ../
sleep 10

{% endfor %}


# End
echo "Ending. Job completed."
