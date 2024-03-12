#!/bin/sh
#PBS -A NTDD0004
#PBS -N UF-CAM-ECT
#PBS -q main
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=0:15:00
#PBS -j oe
#PBS -M abaker@ucar.edu


module load conda
conda activate npl

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

python pyCECT.py --sumfile /glade/campaign/cisl/asap/pycect_sample_data/cam_c1.2.2.1/summary_files/uf.ens.c1.2.2.1_fc5.ne30.nc --indir /glade/campaign/cisl/asap/pycect_sample_data/cam_c1.2.2.1/uf_cam_test_files --tslice 1 --sigMul 2.0 --nPC 50
