
pyEnsSumPop
==================

The verification tools in the CESM-ECT suite all require an *ensemble
summary file*, which contains statistics describing the ensemble distribution.
pyEnsSumPop can be used to create a POP (ocean component) ensemble summary file.


Note that an ensemble summary files for existing CESM tags for POP-ECT
that were created by CSEG (CESM Software Engineering Group)
are located in the CESM input data directory:

$CESMDATAROOT/inputdata/validation/pop_ensembles

Alternatively, pyEnsSumPop can be used to create a summary file for POP-ECT
given the location of appropriate ensemble history files (which should
be generated in CIME via $CIME/tools/statistical_ensemble_test/ensemble.py).

(Note: to generate a summary file for UF-CAM-ECT/CAM-ECT or MPAS-ECT, you must use
pyEnsSum.py or PyEnsSUmMPAS.py, each of which have their own corresponding instructions.)

To use pyEnsSumPop:
--------------------------

1. On NCAR's Derecho machine:

   An example script is given in ``test_pyEnsSumPop.sh``.  Modify as needed and do:

   ``qsub test_pyEnsSumPop.sh``

   Note that the python environment is loaded in the script:
   ``module load conda``
   ``conda activate npl``

2.  Otherwise you need these packages (see ``requirements.txt`):

         * numpy
         * scipy
         * netcdf4
         * mpi4py

3. To see all options (and defaults):

   ``python pyEnsSumPop.py -h``::

         Creates the summary file for an ensemble of POP data.

         ------------------------
	 Args for pyEnsSumPop :
         ------------------------
	 pyEnsSumPop.py
	 -h                   : prints out this usage message
	 --verbose            : prints out in verbose mode (off by default)
	 --sumfile    <ofile> : the output summary data file (default = pop.ens.summary.nc)
	 --indir      <path>  : directory containing all of the ensemble runs (default = ./)
	 --esize <num>        : Number of ensemble members (default = 40)
	                        (Note: backwards compatible with --npert)
         --tag <name>         : Tag name used in metadata (default = tag)
	 --compset <name>     : Compset used in metadata (default = G)
	 --res <name>         : Resolution (used in metadata) (default = T62_g17)
	 --mach <name>        : Machine name used in the metadata (default = derecho)
	 --tslice <num>       : the time slice of the variable that we will use (default = 0)
	 --nyear  <num>       : Number of years (default = 1)
	 --nmonth  <num>      : Number of months (default = 12)
	 --jsonfile <fname>   : Jsonfile to provide that a list of variables that will be
                            	 included  (RECOMMENDED: default = pop_ensemble.json)
	 --mpi_disable        : Disable mpi mode to run in serial (off by default)


Notes:
----------------

1. POP-ECT uses monthly average files. Therefore, one typically needs
    to set ``--tslice 0``.

2.  Note that ``--res``, ``--tag``, ``--compset``, and --mach only affect the
    metadata in the summary file.

3.  The sample script test_pyEnsSumPop.sh gives a recommended parallel
    configuration for Derecho.  We recommend one core per month (and make
    sure each core has sufficient memory).

4.  The json file indicates variables from the output files that you want
    to include in the summary files statistics.  We recommend using the
    default pop_ensemble.json, which contains only 5 variables.



Example:
----------------------------------------
(Note: this example is in test_pyEnsSumPop.sh)

*To generate a summary file for 40 POP-ECT simulations runs (1 year of monthly output):*

* We specify the size and data location:

  ``--esize 40``

  ``--indir /glade/campaign/cisl/asap/pop_verification/cesm2_0_beta10/ensembles``

*  We also specify the name of file to create for the summary:

   ``--sumfile pop.ens.sum.cesm2.0.nc``

* Since these are monthly average files:

  ``--tslice 0``

* We also specify the number of years, the number of months (optional, as 1 and 12 are the defaults):

   ``--nyear 1``

   ``--nmonth 12``

*  We also can specify the tag, resolution, machine and compset
   information (that will be written to the  metadata of the summary file):

   ``--tag cesm2.0_beta10``

   ``--res T62_g16``

   ``--mach derecho``

   ``--compset G``

* We include a recommended subset of variables (5) for the
  analysis by specifying them in a json file (optional, as
  this is the defaut):

  ``--jsonfile pop_ensemble.json``

 * This yields the following command for your job submission script:

 ``python pyEnsSumPop.py  --indir  /glade/campaign/cisl/asap/pycect_sample_data/pop_c2.0.b10/pop_ens_files  --sumfile pop.cesm2.0.b10.nc --tslice 0 --nyear 1 --nmonth 12 --esize 40 --jsonfile pop_ensemble.json   --mach derecho --compset G --tag cesm2_0_beta10 --res T62_g17``
