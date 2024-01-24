#!/bin/sh
#PBS -A NTDD0004
#PBS -N MPASSum
#PBS -q main
#PBS -l select=1:ncpus=36:mpiprocs=36
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

export TMPDIR=/glade/scratch/derecho/$USER/temp
mkdir -p $TMPDIR

mpiexec -n 36 -ppn 36 python pyEnsSumMPAS.py --esize 200 --indir /glade/campaign/cisl/asap/pycect_sample_data/mpas_a.v7.3/mpas_ens_files  --sumfile mpas_sum.nc  --tslice 3 --tag v7.3 --model mpas  --mach cheyenne --verbose --jsonfile empty_excluded.json
