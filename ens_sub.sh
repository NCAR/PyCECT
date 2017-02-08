#! /bin/tcsh -f
#BSUB -n 23 
#BSUB -R "span[ptile=4]"
#BSUB -q regular
#BSUB -N
#BSUB -o ens.%J.stdout
#BSUB -e ens.%J.stdout
#BSUB -a poe
#BSUB -x
#BSUB -J pyEnsSum
#BSUB -W 00:20
#BSUB -P STDD0002

#mpirun.lsf python pyEnsSum.py --indir /glade/u/tdd/asap/verification/cesm1_3_beta06/intel.yellowstone.151 --esize 151  --verbose --tslice 1  --tag cesm1_3_beta06 --sumfile /glade/scratch/haiyingx/intel.151.beta06.nc --jsonfile beta06_ens_excluded_varlist.json --gmonly --mpi_enable 
#mpirun.lsf python pyEnsSum.py --indir /glade/p/work/jshollen/validation/cesm1_4_beta06/ensemble --esize 151   --tslice 0  --tag cesm1_2_2 --sumfile /glade/scratch/haiyingx/intel.151.cesm1_2_2_summary.2.nc --mpi_enable --mach yellowstone --compset B1850 --res f19_g16 --jsonfile ens_excluded_varlist.json --gmonly
#mpirun.lsf python pyEnsSum.py --indir /glade/u/tdd/asap/verification/work/jshollen/validation/cesm1_4_beta06/ensemble --esize 151   --tslice 0  --tag cesm1_2_1 --sumfile /glade/scratch/haiyingx/intel.151.cesm1_2_1_summary.nc --mpi_enable --mach yellowstone --compset B1850 --res f19_g16 --jsonfile ens_excluded_varlist.json --gmonly

#generate a new ens sum of mira_all which will have standardized_gm 
mpirun.lsf python pyEnsSum.py --indir /glade/p/tdd/asap/verification/cesm1_3_beta11/sz151-yellowstone-intel --esize 151   --tslice 1  --tag cesm1_3_beta11 --sumfile /glade/scratch/haiyingx/cesm1.3.b11.ne30_ne30.FC5_V6.nc --mpi_enable --mach yellowstone --compset FC5 --res ne30_ne30 --jsonfile ens_excluded_sz151.json --gmonly
