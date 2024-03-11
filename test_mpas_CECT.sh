#!/bin/sh
#PBS -A NTDD0004
#PBS -N mpas-cect
#PBS -q main
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=0:15:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

python pyCECT.py --sumfile /glade/campaign/cisl/asap/pycect_sample_data/mpas_a.v7.3/summary_files/mpas_sum.nc --indir  /glade/campaign/cisl/asap/pycect_sample_data/mpas_a.v7.3/mpas_test_files --tslice 3 --mpas
