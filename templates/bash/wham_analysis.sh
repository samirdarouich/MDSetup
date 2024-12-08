#!/bin/bash
#PBS -q short
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00
#PBS -N extract_properties
#PBS -o {{folder}}/index_log.o 
#PBS -e {{folder}}/index_log.e 
#PBS -l mem=3000mb

# Bash script to seperate GROMACS pull trajectory and compute distance CV. Automaticaly created by MDSetup.
module purge
module load chem/gromacs/2022.4

# Define the main working path 
WORKING_PATH={{folder}}
cd $WORKING_PATH

echo "This is the working path: $WORKING_PATH"

if [ -f "{{output_prefix}}_pmf.xvg" ] && [ -f "{{output_prefix}}_hist.xvg" ]; then
    echo "{{output_prefix}}_pmf.xvg and {{output_prefix}}_hist.xvg files exist, skipping WHAM analysis."
    exit 0
fi

gmx wham -it {{tpr_files}} -ix {{pullx_files}} -o {{output_prefix}}_pmf.xvg -hist {{output_prefix}}_hist.xvg -unit {{unit}} -temp {{temperature}} {% for key, value in kwargs.items() %}-{{key}} {{value}} {% endfor %}