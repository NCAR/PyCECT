#!/bin/bash
### Job Name 
#PBS -N ect_postrun_xnutr
### Project Code
#PBS -A NTDD0005
#PBS -l walltime=08:00:00
#PBS -q regular
#PBS -l select=4:ncpus=4:mpiprocs=4:mem=45GB
#PBS -M teo.pricebroncucia@colorado.edu
#PBS -m abe

module load conda
conda activate npl

python /glade/work/teopb/PyCECT/MPAS_true_failure_testing/post_run_script.py /glade/work/teopb/PyCECT/MPAS_true_failure_testing/test_params_xnutr.json
