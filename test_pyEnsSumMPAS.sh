#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N ensSum
#PBS -q regular
#PBS -l select=4:ncpus=9:mpiprocs=9
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl


setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

mpiexec python pyEnsSumMPAS.py --esize 100 --indir /glade/scratch/abaker/longer_mpas_hist --sumfile mpas_sum_t24_new.nc  --tslice 8 --tag v7.1 --model mpas  --mach cheyenne --verbose --jsonfile mpas_ex.json
