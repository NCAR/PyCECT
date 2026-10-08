#!/bin/bash
#PBS -A NCGD0067
#PBS -N ensSumMom
#PBS -q main
#PBS -l select=1:ncpus=12:mpiprocs=12
#PBS -l walltime=0:20:00
#PBS -j oe


module load conda
conda activate npl

mpiexec -n 12 -ppn 12 python pyEnsSumMom6.py --verbose --indir /glade/derecho/scratch/abaker/mom6/ensfiles --sumfile mom_sumfile.nc --nyear 1 --nmonth 12 --esize 15 --jsonfile mom6_ensemble.json  --mach derecho
