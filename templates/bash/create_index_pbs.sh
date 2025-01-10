#!/bin/bash

# Bash script to extract GROMACS index file. Automaticaly created by MDSetup.
module purge
module load chem/gromacs/2022.4


# Run the command and capture its output using the script command
output=$(script -q -c "gmx make_ndx -f {{structure}} <<< ''" /dev/null)

# Extract the highest selection number
offset=$(echo "$output" | grep -oP '^\s*\d+\s' | awk '{print $1}' | sort -n | tail -1)

# Define the lists of selections and names
selections=( {% for item in selections %}'{{ item }}' {% endfor %})
names=( {% for item in names %}'{{ item }}' {% endfor %})

# Initialize an empty string for the commands
commands=""

# Loop through the selections and names
for i in "${!selections[@]}"; do
  # Add the selection command
  commands+="${selections[i]}\n"
  # Add the name command using the current index + 17 (starting offset)
  commands+="name $((offset + 1 + i)) ${names[i]}\n"
done

# Add the quit command
commands+="q"

# Call GROMACS
echo -e "$commands" | gmx  make_ndx -f {{structure}} -o {{index_file}}  {% if not old_index_file is none %} -n {{old_index_file}} {% endif %}
# Delete old .ndx files
rm -f \#*.ndx.*#