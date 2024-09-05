#!/bin/sh
#PBS -A NTDD0004
#PBS -N ensSum
#PBS -q main
#PBS -l select=20:ncpus=36:mpiprocs=36
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

module load conda
conda activate npl

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

mpiexec -n 720 -ppn 36 python pyEnsSum.py --esize 1800 --indir /glade/campaign/cisl/asap/pycect_sample_data/cam7_ne30/uf_cam_ens_files --sumfile cam_sum.nc  --tslice 0 --tag cam6_4_019 --compset F2000 --res ne30_ne30 --mach derecho --verbose --jsonfile empty_excluded.json
