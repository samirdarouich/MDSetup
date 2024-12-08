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

if ls conf*.gro 1> /dev/null 2>&1; then
    echo "conf*.gro files exist, skipping extraction."
else
    echo "No conf*.gro files found, proceeding with extraction."
    # Split the trajectory of the whole system into frames
    echo 0 | gmx trjconv -s {{ensemble}}.tpr -f {{ensemble}}.xtc -o conf.gro -sep
fi

# Number of frames
no_frames=$(ls conf*.gro 2>/dev/null | grep -oP '(?<=conf)\d+(?=\.gro)' | sort -n | tail -1)

if [[ -z "$no_frames" ]]; then
    echo "No trajectory frames found."
    exit 1
fi

# compute collective variable
if ls dist*.xvg 1> /dev/null 2>&1 || [ -f collective_variable.csv ]; then
    
    if [ -f collective_variable.csv ]; then
        echo "dist*.xvg and cv.csv files exist, ending job."
        exit 0
    else
        echo "dist*.xvg files exist, skipping cv computation."
    fi
else
    for (( i=0; i<${no_frames}+1; i++ ))
    do
    gmx distance -s {{ensemble}}.tpr -f conf${i}.gro -n {{index_file}} -select '{{selection}}' -oall dist${i}.xvg
    done
fi

# Get output files
output_files=$(ls *.xvg 2>/dev/null | head -1)

if [[ -z "$output_files" ]]; then
    echo "No *.xvg output files found."
    exit 1
fi

# Get base name of output files (assuming all are the same and only one type of xvg file is present)
base_name=$(echo "$output_files" | awk -F '[0-9]+' '{print $1}')

# compile summary
touch collective_variable.csv
echo "frame,cv" >> collective_variable.csv
for (( i=0; i<${no_frames}+1; i++ ))
do
    d=$(tail -n 1 ${base_name}${i}.xvg | awk '{print $2}')
    echo "${i},${d}" >> collective_variable.csv
    rm ${base_name}${i}.xvg
done