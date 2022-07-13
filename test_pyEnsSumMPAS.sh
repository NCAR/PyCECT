#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N ensSum
#PBS -q regular
#PBS -l select=2:ncpus=9:mpiprocs=9
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl


setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

mpiexec python pyEnsSumMPAS.py --esize 1 --indir /glade/work/abaker/mpas_data/ensemble --sumfile PAR_mpas_sum.nc  --tslice 0 --tag v7.1 --model mpas  --mach cheyenne --verbose
