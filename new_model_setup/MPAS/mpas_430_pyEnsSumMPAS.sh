#!/bin/bash -l
#PBS -A NTDD0005
#PBS -N ensSumMPAS
#PBS -q main
#PBS -l select=4:ncpus=4:mpiprocs=4
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M teopb@ucar.edu

module load conda
conda activate npl

for ((i=0; i<=16; i++))
do
    mpiexec python pyEnsSumMPAS.py --esize 430 --indir /glade/campaign/cisl/asap/abaker/mpas/large_ens  --sumfile /glade/work/teopb/PyCECT/MPAS_no_pv_430_sums/mpas_sum_430_slice_3x$i.nc  --tslice $i --tag v7.1 --model mpas  --mach cheyenne --verbose --jsonfile /glade/work/teopb/PyCECT/mpas_ex_with_pv_vars.json
done
