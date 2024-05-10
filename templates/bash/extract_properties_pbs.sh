#!/bin/bash
#PBS -q short
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00
#PBS -N extract_properties
#PBS -o {{folder}}/extract_log.o 
#PBS -e {{folder}}/extract_log.e 
#PBS -l mem=3000mb

# Bash script to extract GROMACS properties. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2022.4

# Go to working folder
cd {{folder}}

# List of properties to extract
properties=( {% for item in extracted_properties %}'{{ item }}' {% endfor %})

# Initialize an empty string to store the numbers
numbers_string=""

# Run the command and capture its output using the script command
output=$(script -q -c "gmx {{gmx_command}} <<< ''" /dev/null)

for property in "${properties[@]}"; do
    # Use grep and awk to extract the number corresponding to the property
    number=$(echo "$output" | grep -E "([0-9]+)\s+$property" | awk -v prop="$property" '{ for (i=1; i<=NF; i++) { if ($i == prop) { print $(i-1); exit } } }')
    numbers_string+="$number "
done

# Check if no number is extracted. Then simply use the property list as input
if [[ "$numbers_string" =~ ^[[:space:]]*$ ]]; then
    echo "Warning no numbers were extracted. This might results unexpected behavior."
    echo "Instead use the specified properties directly"
    
    for property in "${properties[@]}"; do
        numbers_string+="$property\n"
    done
fi

# Call GROMACS
echo -e "$numbers_string" | gmx {{gmx_command}}

# Delete old .xvg files
rm -f \#*.xvg.*#