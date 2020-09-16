#!/bin/tcsh
#PBS -A NIOW0001
#PBS -N ensSum
#PBS -q regular
#PBS -l select=8:ncpus=9:mpiprocs=9
#PBS -l walltime=0:20:00
#PBS -j oe
#PBS -M abaker@ucar.edu

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

python pyCECT.py --sumfile uf.ens.c1.2.2.1_fc5.ne30.nc --indir /glade/p/cisl/asap/pycect_sample_data/cam_c1.2.2.1/uf_cam_ens_files --tslice 1
 
