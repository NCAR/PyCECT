#!/bin/sh
#PBS -A NTDD0004
#PBS -N ensSumPop
#PBS -q main
#PBS -l select=1:ncpus=12:mpiprocs=12
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

mpiexec -n 12 -ppn 12 python pyEnsSumPop.py --verbose --indir  /glade/p/cisl/asap/pycect_sample_data/pop_c2.0.b10/pop_ens_files --sumfile pop.cesm2.0.b10.nc --tslice 0 --nyear 1 --nmonth 12 --esize 40 --jsonfile pop_ensemble.json  --mach cheyenne --compset G --tag cesm2_0_beta10 --res T62_g17
