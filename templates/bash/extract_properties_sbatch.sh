#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -p single
#SBATCH -J extract_properties
#SBATCH -o {{folder}}/extract_log.o
#SBATCH --mem-per-cpu=2000

# Bash script to extract GROMACS properties. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2023.3

# Go to working folder
cd {{folder}}

# List of properties to extract
properties=( {% for item in extracted_properties %}'{{ item }}' {% endfor %})

# Initialize an empty string to store the numbers
numbers_string=""

# Run the command and capture its output using the script command
output=$(script -q -c "gmx_mpi {{gmx_command}} <<< ''" /dev/null)

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
echo -e "$numbers_string" | gmx_mpi {{gmx_command}}

# Delete old .xvg files
rm -f \#*.xvg.*#