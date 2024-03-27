#!/bin/bash
#PBS -q tiny
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -N build_system
#PBS -o /home/st/st_st/st_ac137577/workspace/software/pyLAMMPS/example/butane_hexane/md_thermo/temp_343_pres_4/build/build_log.o 
#PBS -e /home/st/st_st/st_ac137577/workspace/software/pyLAMMPS/example/butane_hexane/md_thermo/temp_343_pres_4/build/build_log.e 
#PBS -l mem=3000mb

# Bash script to generate PLAYMOL box. Automaticaly created by pyLAMMPS.

# Go to working folder
cd /home/st/st_st/st_ac137577/workspace/software/pyLAMMPS/example/butane_hexane/md_thermo/temp_343_pres_4/build

# Use PLAYMOL to build the box
playmol /home/st/st_st/st_ac137577/workspace/software/pyLAMMPS/example/butane_hexane/md_thermo/temp_343_pres_4/build/build_script.mol