#!/bin/tcsh
#PBS -A NTDD0004
#PBS -N mpas-cect
#PBS -q regular
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl


setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python pyCECT.py --sumfile /glade/work/abaker/mpas_data/100_ens_summary/mpas_sum_ts24.nc --indir /glade/scratch/abaker/longer_mpas_hist --tslice 8 --mpas
