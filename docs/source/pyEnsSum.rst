
pyEnsSum
==============

The verification tools in the CESM-ECT suite all require an *ensemble
summary file*, which contains statistics describing the ensemble distribution.
pyEnsSum can be used to create a CAM (atmospheric component) ensemble summary file.

Note that an ensemble summary files for existing CESM tags for CAM-ECT and UF-CAM-ECT
that were created by CSEG (CESM Software Engineering Group)
are located (respectively) in the CESM input data directories:

$CESMDATAROOT/inputdata/validation/ensembles
$CESMDATAROOT/inputdata/validation/uf_ensembles

Alternatively, pyEnsSum.py be used to create a summary file for CAM-ECT or
UF-CAM-ECT, given the location of appropriate ensemble history files (which should
be generated via CESM,  https://github.com/ESCOMP/CESM).

(Note: to generate a summary file for POP-ECT or MPAS-ECT, you must use pyEnsSumPop.py
or PyEnsSumMPAS.py, respectively, each of which have their own corresponding instructions.)

To use pyEnsSum:
--------------------

1. On NCAR's Derecho machine:

   An example script is given in ``test_pyEnsSum.sh``. Modify as needed and do:

   ``qsub test_pyEnsSum.sh``

   Note that the python environment is loaded in the script:

   ``module load conda``

   ``conda activate npl``

2.  Otherwise you need these packages (see ``requirements.txt``):

         * numpy
         * scipy
         * netcdf4
         * mpi4py

3. To see all options (and defaults):

   ``python pyEnsSum.py -h``::

       Creates the summary file for an ensemble of CAM data.

       ------------------------
        Args for pyEnsSum :
       ------------------------
        pyEnsSum.py
        -h                   : prints out this usage message
        --verbose            : prints out in verbose mode (off by default)
        --sumfile <ofile>    : the output summary data file (default = ens.summary.nc)
        --indir <path>       : directory containing all of the ensemble runs (default = ./)
        --esize  <num>       : Number of ensemble members (default = 1800)
        --tag <name>         : Tag name used in metadata (default = cesm_version)
        --compset <name>     : Compset used in metadata (default = compset)
        --res <name>         : Resolution used in metadata (default = res)
        --mach <name>        : Machine name used in the metadata (default = derecho)
        --tslice <num>       : the index into the time dimension (default = 0)
        --jsonfile <fname>   : Jsonfile to provide that a list of variables that will
                               be excluded (default = exclude_empty.json)
        --mpi_disable        : Disable mpi mode to run in serial (off by default)



Notes:
------------------

1. CAM-ECT uses yearly average files, which by default (in the ensemble.py
   generation script in CESM) also contain the initial conditions.  Therefore,
   one typically needs to set ``--tslice 1`` to use the yearly average (because
   slice 0 is the initial conditions.)

2.  UF-CAM-ECT uses an early timestep such as 7 or 9.  By default (in the ensemble.py
    generation script in CESM) the ouput file no longer contains the initial conditions.
    Therefore, one typically needs to set ``--tslice 0``, assuming that only one timestep
    is written to the file.

3. There is no need to indicate UF-CAM-ECT vs. CAM-ECT to this routine.  It
   simply creates statistics for the supplied history files at the specified
   time slice. For example, if you want to look at monthly files, simply
   supply their location.  Monthly files typically do not contain an initial
   condition and would require ``--tslice 0``.

4. The ``--esize``  (the ensemble size) can be less than or equal to the number of files
   in ``--indir``.  Ensembles numbered 0000-(esize-1) will be included.  UF-CAM-ECT
   typically uses at least 1800 members, whereas CAM-ECT does not require as many.

5. Note that ``--res``, ``--tag``, ``--compset``, and ``--mach``
   parameters only affect the metadata written to the summary file.

6. When running in parallel, the recommended number of cores to use is one
   for each 3D variable. The default is to run in parallel (recommended).

7. You must specify a json file (via ``--jsonfile``) that indicates
   the variables in the ensemble
   output files that you want to exclude from the summary file
   statistics (see the example json files).  The default is the provided
   empty_excluded,json, which is does not contain any variables.
   The pyEnsSum routine will let you know if you have not
   listed variables that need to be excluded (see more in next note).

8. *IMPORTANT:* If there are variables that need to be excluded (that are not in
   the .json file  already) for the summary to be generated, pyEnsSum will list these
   variables in the output.  These variables will also be added to a copy of
   your exclude variable list (prefixed with "NEW.") for future reference and use.
   The summary file will be generated with all listed variables excluded.
   Note that the following types of variables will be removed:  any variables that
   are constant across the ensemble, are not floating-point (e.g., integer),
   are linearly dependant, or have very few (< 3%) unique values.

9. The pyEnsSum.py program parallelizes over the number of files in the ensemble
   directory. This allows for the use of a large number of processes, up to the ensemble size if needed. Our experience on Derecho has also shown that per-node bandwidth is a limiting factor. For this reason, we recommend using fewer processes than possible cores. With sufficient nodes, (for example 20 nodes and 36 cores per node) an 1800 member ensemble should be processed in less than 10 minutes on Derecho.

Example:
--------------------------------------
(Note: This example is in test_pyEnsSum.sh)

*To generate a summary file for 1800 UF-CAM-ECT simulations runs (time step 7):*

* we specify the size and data location:

  ``--esize 1800``

  ``--indir /glade/campaign/cisl/asap/pycect_sample_data/cam_c1.2.2.1/uf_cam_ens_files``

* We also specify the name of file to create for the summary:

  ``--sumfile uf.ens.c1.2.2.1_fc5.ne30.nc``

* If the ensemble files do not contain the initial conditions as a timeslice, the desired values at time step 7 are the only timeslice. So we set:

  ``--tslice 0``

* We also specify the CESM tag, compset and resolution and machine of our ensemble data so that it can be written to the metadata of the summary file:

  ``--tag cesm1.2.2.1 --compset FC5 --res ne30_ne30 --mach derecho``

* We can exclude variables from the analysis by specifying them in a json file:

  ``--jsonfile excluded_varlist.json``

* This yields the following command for your job submission script:

  ``python pyCECT.py --esize 1800 --indir /glade/campaign/cisl/asap/pycect_sample_data/cam_c1.2.2.1/uf_cam_ens_files  --sumfile uf.ens.c1.2.2.1_fc5.ne30.nc  --tslice 0 --tag cesm1.2.2.1 --compset FC5 --res ne30_ne30 --jsonfile excluded_varlist.json``
