#!/bin/sh
#PBS -A NTDD0004
#PBS -N ensSum
#PBS -q main
#PBS -l select=1:ncpus=36:mpiprocs=36
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

mpiexec -n 36 -ppn 36 python pyEnsSum.py --esize 350 --indir /glade/campaign/cisl/asap/pycect_sample_data/cam_c1.2.2.1/uf_cam_ens_files --sumfile cam_sum.nc  --tslice 1 --tag cesm1.2.2.1 --compset FC5 --res ne30_ne30 --mach cheyenne --verbose --jsonfile empty_excluded.json
