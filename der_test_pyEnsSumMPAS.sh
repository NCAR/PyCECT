#!/bin/bash -l
#PBS -A NTDD0004
#PBS -N ensSumM
#PBS -q main
#PBS -l select=1:ncpus=36:mpiprocs=36
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

mpiexec python pyEnsSumMPAS.py --esize 100 --indir /glade/p/cisl/asap/abaker/mpas/large_ens  --sumfile mpas_sumt4.nc  --tslice 4 --tag v7.1 --model mpas  --mach cheyenne --verbose --jsonfile empty_excluded.json
