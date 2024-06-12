#!/bin/bash
### Job Name
#PBS -N ect_postrun_xnutr
### Project Code
#PBS -A YOUR_PROJECT_CODE
#PBS -l walltime=00:30:00
#PBS -q regular
#PBS -l select=4:ncpus=4:mpiprocs=4:mem=45GB
#PBS -M YOUR_EMAIL_ADDRESS
#PBS -m abe

module load conda
conda activate npl

python PyCECT/MPAS_true_failure_testing/post_run_script.py MPAS_true_failure_testing/test_params_chey_tests.json
