#!/bin/bash
#PBS -q tiny
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -N build_system
#PBS -o {{folder}}/build_log.o 
#PBS -e {{folder}}/build_log.e 
#PBS -l mem=3000mb

# Bash script to generate PLAYMOL box. Automaticaly created by pyLAMMPS.

# Go to working folder
cd {{folder}}

# Use PLAYMOL to build the box
playmol {{file}}
