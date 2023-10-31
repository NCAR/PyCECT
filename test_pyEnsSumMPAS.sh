#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N MensSum
#PBS -q premium
#PBS -l select=4:ncpus=9:mpiprocs=9
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl


setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

mpiexec python pyEnsSumMPAS.py --esize 100 --indir /glade/p/cisl/asap/abaker/mpas/large_ens  --sumfile mpas_sumt4.nc  --tslice 4 --tag v7.1 --model mpas  --mach cheyenne --verbose --jsonfile test.json
