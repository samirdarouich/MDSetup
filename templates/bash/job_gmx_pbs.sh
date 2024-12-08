#!/bin/bash
#PBS -q short
#PBS -l nodes=1:ppn=14
#PBS -l walltime=48:00:00
#PBS -N {{job_name}}
#PBS -o {{log_path}}.o 
#PBS -e {{log_path}}.e 
#PBS -l mem=3000mb

module purge
module load chem/gromacs/2022.4

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

gmx {{ensemble.grompp}}

gmx {{ensemble.mdrun}}

echo "Completed ensemble: {{ensemble_name}}"

cd ../
sleep 10

{% endfor %}


# End
echo "Ending. Job completed."
