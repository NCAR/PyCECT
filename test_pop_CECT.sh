#!/bin/sh

#PBS -A NTDD0004
#PBS -N pop-CECT
#PBS -q main
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=0:15:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

python pyCECT.py --popens --sumfile  /glade/campaign/cisl/asap/pycect_sample_data/pop_c2.0.b10/summary_files/pop.cesm2.0.b10.nc --indir /glade/campaign/cisl/asap/pycect_sample_data/pop_c2.0.b10/pop_test_files/lw-lim --tslice 0 --jsonfile pop_ensemble.json --input_glob lw-lim.pop.000.pop.h.0001-12

python pyCECT.py --popens --sumfile  /glade/campaign/cisl/asap/pycect_sample_data/pop_c2.0.b10/summary_files/pop.cesm2.0.b10.nc --indir /glade/campaign/cisl/asap/pycect_sample_data/pop_c2.0.b10/pop_test_files/C96 --tslice 0 --jsonfile pop_ensemble.json --input_glob C96.pop.000.pop.h.0001-12
