#!/bin/csh -f
#PBS -A P93300606
#PBS -N pyEnsSumPop
#PBS -q regular
#PBS -l select=1:ncpus=12:mpiprocs=12
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -M abaker@ucar.edu


    mpiexec_mpt -np 12 python pyEnsSumPop.py --verbose --indir  /glade/p/cisl/iowa/pop_verification/cesm2_0_beta10/ensembles --sumfile new.pop.ens.sum.cesm2.0.b10.nc --tslice 0 --nyear 1 --nmonth 12 --npert 40 --jsonfile pop_ensemble.json  --mpi_enable --zscoreonly --mach cheyenne --compset G --tag cesm2_0_beta10 --res T62_g17
