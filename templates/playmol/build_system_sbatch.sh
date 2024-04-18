#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 00:30:00
#SBATCH -p dev_single
#SBATCH -J build_system
#SBATCH -o {{folder}}/build_log.o 
#SBATCH --mem-per-cpu=150

# Bash script to generate PLAYMOL box. Automaticaly created by pyLAMMPS.

# Go to working folder
cd {{folder}}

# Use PLAYMOL to build the box
playmol {{file}}
