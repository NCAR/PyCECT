
pyEnsSumMPAS
==============

The verification tools in the CECT suite all require an *ensemble
summary file*, which contains statistics describing the ensemble distribution.
pyEnsSumMPAS can be used to create an MPAS (atmospheric component) ensemble summary file.

Note that an ensemble summary files for existing MPAS tags are not yet available as this
functionality is new.  Therefore, pyEnsSum.py be used to create a summary file for MPAS-ECT,
given the location of appropriate ensemble history files (which should be generated
via MPAS-A, https://github.com/MPAS-Dev/MPAS-Model).

(Note: to generate a summary file for POP-ECT or MPAS-ECT, you must use pyEnsSumPop.py
or PyEnsSumMPAS.py, respectively, each of which have their own corresponding instructions.)


o use pyEnsSumMPAS:
--------------------

1. On NCAR's Derecho machine:

   An example script is given in ``test_pyEnsSumMPAS.sh``.  Modify as needed and do:

   ``qsub test_pyEnsSumMPAS.sh``

   Note that the python environment is loaded in the script:
   ``module load conda``
   ``conda activate npl``

2.  Otherwise you need these packages (see ``requirements.txt`):

         * numpy
         * scipy
         * netcdf4
         * mpi4py

3. To see all options (and defaults):

   ``python pyEnsSumMPAS.py -h*``::

        Creates the summary file for an ensemble of MPAS data. 

	------------------------
	Args for pyEnsSumMPAS : 
	------------------------
	pyEnsSumMPAS.py
	-h                   : prints out this usage message
	--verbose            : prints out in verbose mode (off by default)
	--sumfile <ofile>    : the output summary data file (default = mpas.ens.summary.nc)
	--indir <path>       : directory containing all of the ensemble runs (default = ./)
	--esize  <num>       : Number of ensemble members (default = 200)
	--tag <name>         : Tag name for the summary metadata (default = tag)
	--core <name>        : Core name for the summary metadata (default = atmosphere)
	--mesh <name>        : Mesh name for the summary metadata (default = mesh)
	--model <name>       : Model name for the summary metadata (default = mpas)
	--mach <name>        : Machine name used in the metadata (default = derecho)
	--tslice <num>       : the index into the time dimension (default = 0)
	--jsonfile <fname>   : Jsonfile to provide that a list of variables that will 
                        	be excluded  (default = empty_excluded.json)
        --mpi_disable        : Disable mpi mode to run in serial (mpi is enabled by default)
   

     

Notes:
------------------

1. MPAS-ECT typically uses data after several timeteps, and the output file may contain
   multiple timeslice and may or may not
   contain initial conditions.   Therefore, just be aware when choosing which time to use
   to generate the summary that this same time slice is used for testing with pyCECT. Specify
   the time slice with ``--tslice 0`, for example.

2. The ``--esize``  (the ensemble size) can be less than or equal to the number of files
   in ``--indir``.  Ensembles numbered 0000-(esize-1) will be included unless ``--fIndex``
   is specified.  MPAS-ECT typically uses at least 200 members.

3. Note that ``--core``, ``--tag``, ``--mesh``, ``--model``, and ``--mach``
   parameters only affect the metadata written to the summary file.

4. When running in parallel, the recommended number of cores to use is one
   for each 3D variable. The default is to run in parallel (recommended).

5. You must specify a json file (via ``--jsonfile``) that indicates
   the variables in the ensembleoutput files that you want to exclude from the summary file
   statistics (see the example json files).  The default is the provided
   empty_excluded,json, which is does not contain any variables.
   The pyEnsSumMPAS routine will let you know if you have not
   listed variables that need to be excluded (see more in next note).
   
6. *IMPORTANT:* If there are variables that need to be excluded (that are not in
   the .json file  already) for the summary to be generated, pyEnsSumMPAS will list these
   variables in the output.  These variables will also be added to a copy of
   your exclude variable list (prefixed with "NEW.") for future reference and use.
   The summary file will be generated with all listed variables excluded.
   Note that the following types of variables will be removed:  any variables that
   are constant across the ensemble, are not floating-point (e.g., integer),
   are linearly dependant, or have very few (< 3%) unique values.


Example:
--------------------------------------
(Note: This example is in test_pyEnsSumMPAS.sh)

*To generate a summary file for 200 MPAS-ECT simulations runs (from time slice 3 in the file):*

* we specify the size and data location:

  ``--esize 200``

  ``--indir /glade/campaign/cisl/asap/pycect_sample_data/mpas_a.v7.3/mpas_ens_files``

* We also specify the name of file to create for the summary:

  ``--sumfile mpas_sum.nc.nc``

* Since the ensemble files could contain more than one time steps (in this example,
  starting a 3 and output every 3), then we specify a timeslice corresponding to timestep 12 with:

``--tslice 3``

* We can also specify the MPAS tag, model, mesh, core and machine of our ensemble data so that it can be written to the metadata of the summary file:

  ``--tag v7.3 --model mpas --mach cheyenne``

* We can exclude variables from the analysis by specifying them in a json file:

  ``--jsonfile empty_excluded.json``

* This yields the following command for your job submission script:

  ``python pyEnsSumMPAS.py --esize 200 --indir /glade/campaign/cisl/asap/pycect_sample_data/mpas_a.v7.3/mpas_ens_files  --sumfile mpas_sum.nc --tslice 3 --tag v7.3 --model mpas  --mach cheyenne --verbose --jsonfile empty_excluded.json``
