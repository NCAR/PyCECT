#!/bin/bash -l
#PBS -A JOB_CODE
#PBS -N ensSumMPAS
#PBS -q main
#PBS -l select=4:ncpus=4:mpiprocs=4
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M YOUR_EMAIL

module load conda
conda activate npl

for ((i=0; i<=16; i++))
do
    mpiexec python ../../pyEnsSumMPAS.py --esize 430 --indir PATH_TO_MODEL_RUNS  --sumfile WHERE_TO_PLACE_SUMMARY_FILES/MPAS_no_pv_430_sums/mpas_sum_430_slice_3x$i.nc  --tslice $i --tag v7.1 --model mpas  --mach MACHINE_USED_TO_CREATE_RUNS --verbose --jsonfile PATH_TO_EXCLUDE_FILE/mpas_ex_with_pv_vars.json
done
