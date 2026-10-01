#!/usr/bin/env python
import getopt
import os
import sys
import time

import netCDF4 as nc
import numpy as np

import pyEnsLib
import pyTools
from pyTools import EqualStride

#files should have member, then year, then month in the filename (in that order) to be recognized by this script
# we want 1 rank per timeslice (so months*years)

def main(argv):
    # Get command line stuff and store in a dictionary
    s = 'tag= mach= nyear= nmonth= esize= nbin= minrange= maxrange= res= sumfile= indir= jsonfile= verbose mpi_enable mpi_disable'
    optkeys = s.split()

    try:
        opts, args = getopt.getopt(argv, 'h', optkeys)
    except getopt.GetoptError:
        pyEnsLib.EnsSumMom_usage()
        sys.exit(2)

    # Put command line options in a dictionary - also set defaults
    opts_dict = {}

    # Defaults
    opts_dict['model'] = 'MOM6'
    opts_dict['tag'] = 'tag'
    opts_dict['mach'] = 'derecho'
    opts_dict['nyear'] = 1
    opts_dict['nmonth'] = 12
    opts_dict['esize'] = 40
    opts_dict['nbin'] = 40
    opts_dict['minrange'] = 0.0
    opts_dict['maxrange'] = 4.0
    opts_dict['res'] = 'res'
    opts_dict['sumfile'] = 'mom6.ens.summary.nc'
    opts_dict['indir'] = './'
    opts_dict['jsonfile'] = 'mom6_ensemble.json'
    opts_dict['verbose'] = False
    opts_dict['mpi_enable'] = True
    opts_dict['mpi_disable'] = False
    
    # TO DO: why not listed in help: seq, minrange, maxrange, nbin, mpi_enable

    # This creates the dictionary of input arguments
    opts_dict = pyEnsLib.getopt_parseconfig(opts, optkeys, 'ES_MOM', opts_dict)

    verbose = opts_dict['verbose']
    nbin = opts_dict['nbin']

    if opts_dict['mpi_disable']:
        opts_dict['mpi_enable'] = False

    st = opts_dict['esize']
    esize = int(st)

    # Now find file names in indir
    input_dir = opts_dict['indir']

    # Create a mpi simplecomm object
    if opts_dict['mpi_enable']:
        me = pyTools.create_comm()
    else:
        me = pyTools.create_comm(False)

        # Z, Y,X
    if opts_dict['jsonfile']:
        # Read in the included var list
        json_vars = pyEnsLib.read_jsonlist(opts_dict['jsonfile'], 'ES_MOM')
        if len(json_vars) != 4:
            if me.get_rank() == 0:
                print('ERROR: could not read MOM6 variable lists from ', opts_dict['jsonfile'])
            sys.exit(2)
        Var_lhh, Var_ihh, Var_lhq, Var_lqh = json_vars


        # get max size of var names
        str_size = 0
        str_size = max(len(v) for v in Var_lhh + Var_ihh + Var_lhq + Var_lqh)


    # get number of each variable type
    n_var_lhh = len(Var_lhh)
    n_var_ihh = len(Var_ihh)
    n_var_lqh = len(Var_lqh)
    n_var_lhq = len(Var_lhq)

    nvars = n_var_lhh + n_var_ihh + n_var_lqh + n_var_lhq

    if me.get_rank() == 0:
        print('STATUS: Running pyEnsSumMom6!')

        if verbose:
            print('VERBOSE: opts_dict = ')
            print(opts_dict)

    in_files = []
    if os.path.exists(input_dir):
        # Get the list of files
        in_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.nc'))
        num_files = len(in_files)
    else:
        if me.get_rank() == 0:
            print('ERROR: Input directory: ', input_dir, ' not found => EXITING....')
        sys.exit(2)

    # make sure we have enough files
    files_needed = opts_dict['nmonth'] * esize * opts_dict['nyear']
    if num_files != files_needed:
        if me.get_rank() == 0:
            print(
                'ERROR: Input directory must contain exactly esize*nyear*nmonth = ',
                files_needed,
                ' ( but it has',
                num_files,
                ' files).',
            )
        sys.exit(2)

    # if using parallel, then we want exactly one timeslice per process
    # (serial handles all timeslices)
    ntslices = opts_dict['nmonth'] * opts_dict['nyear']
    if opts_dict['mpi_enable'] and me.get_size() != ntslices:
        if me.get_rank() == 0:
            print(
                'ERROR: number of processors (',
                me.get_size(),
                ') must equal nmonth*nyear (',
                ntslices,
                ') => EXITING....',
            )
        sys.exit(2)

    # Partition the timeslices (serial gets all of them, MPI gets one per rank)
    my_slices = me.partition(list(range(ntslices)), func=EqualStride(), involved=True)

    # Files for timeslice k are in_files[k::ntslices] (assumes the sorted file
    # names are member-major: all months for member 0, then member 1, ...)
    slice_files = []
    for k in my_slices:
        full_in_files = []
        for onefile in in_files[k::ntslices]:
            fname = input_dir + '/' + onefile
            if os.path.isfile(fname):
                full_in_files.append(fname)
            else:
                print('ERROR: Could not locate file: ' + fname + ' => EXITING....')
                if opts_dict['mpi_enable']:
                    me.abort()
                sys.exit(2)
        slice_files.append(full_in_files)

    # open just the first file (all procs) to get metadata
    first_file = nc.Dataset(slice_files[0][0], 'r')

    # Store dimensions of the input fields
    if verbose and me.get_rank() == 0:
        print('VERBOSE: Getting spatial dimensions')
    z_l = -1
    z_i = -1
    yq = -1
    yh = -1
    xq = -1
    xh = -1

    # Look at first file and get dims
    input_dims = first_file.dimensions
    # All files should have the same dimensions
    if verbose and me.get_rank() == 0:
        print('VERBOSE: Checking dimensions ...')
    for key in input_dims:
        if key == 'z_l':
            z_l = len(input_dims['z_l'])
        elif key == 'z_i':
            z_i = len(input_dims['z_i'])
        elif key == 'yq':
            yq = len(input_dims['yq'])
        elif key == 'yh':
            yh = len(input_dims['yh'])
        elif key == 'xq':
            xq = len(input_dims['xq'])
        elif key == 'xh':
            xh = len(input_dims['xh'])

    if z_i == -1 or z_l == -1 or xq == -1 or xh == -1 or yh == -1 or yq == -1:
        if me.get_rank() == 0:
            print('ERROR: Need dimensions z_i, z_l, xq, xh, yh and yq => EXITING....')
        sys.exit(2)

    if verbose and me.get_rank() == 0:
        print('z_i = ', z_i)
        print('z_l = ', z_l)
        print('yq = ', yq)
        print('yh = ', yh)
        print('xq = ', xq)
        print('xh = ', xh)

    # Rank 0: prepare new summary ensemble file
    this_sumfile = opts_dict['sumfile']
    if me.get_rank() == 0:
        if os.path.exists(this_sumfile):
            os.unlink(this_sumfile)

        if verbose:
            print('VERBOSE: Creating ', this_sumfile, '  ...')

        nc_sumfile = nc.Dataset(this_sumfile, 'w', format='NETCDF4_CLASSIC')

        # Set dimensions
        if verbose:
            print('VERBOSE: Setting dimensions .....')
        nc_sumfile.createDimension('z_i', z_i)
        nc_sumfile.createDimension('z_l', z_l)
        nc_sumfile.createDimension('yq', yq)
        nc_sumfile.createDimension('yh', yh)
        nc_sumfile.createDimension('xq', xq)
        nc_sumfile.createDimension('xh', xh)

        nc_sumfile.createDimension('time', None)

        nc_sumfile.createDimension('ens_size', esize)
        nc_sumfile.createDimension('nbin', opts_dict['nbin'])
        nc_sumfile.createDimension('nvars', nvars)
        nc_sumfile.createDimension('nvars_lhh', n_var_lhh)
        nc_sumfile.createDimension('nvars_ihh', n_var_ihh)
        nc_sumfile.createDimension('nvars_lqh', n_var_lqh)
        nc_sumfile.createDimension('nvars_lhq', n_var_lhq)

        nc_sumfile.createDimension('str_size', str_size)

        # Set global attributes
        now = time.strftime('%c')
        if verbose:
            print('VERBOSE: Setting global attributes .....')
        nc_sumfile.creation_date = now
        nc_sumfile.title = 'MOM6 verification ensemble summary file'
        nc_sumfile.tag = opts_dict['tag']
        nc_sumfile.model = opts_dict['model']
        nc_sumfile.resolution = opts_dict['res']
        nc_sumfile.machine = opts_dict['mach']

        # Create variables
        if verbose:
            print('VERBOSE: Creating variables .....')
        v_vars = nc_sumfile.createVariable('vars', 'S1', ('nvars', 'str_size'))
        v_var_lhh = nc_sumfile.createVariable('var_lhh', 'S1', ('nvars_lhh', 'str_size'))
        v_var_ihh = nc_sumfile.createVariable('var_ihh', 'S1', ('nvars_ihh', 'str_size'))
        v_var_lhq = nc_sumfile.createVariable('var_lhq', 'S1', ('nvars_lhq', 'str_size'))
        v_var_lqh = nc_sumfile.createVariable('var_lqh', 'S1', ('nvars_lqh', 'str_size'))
        v_time = nc_sumfile.createVariable('time', 'd', ('time',))

        v_ens_avg_lhh = nc_sumfile.createVariable(
            'ens_avg_lhh', 'f', ('time', 'nvars_lhh', 'z_l', 'yh', 'xh')
        )
        v_ens_stddev_lhh = nc_sumfile.createVariable(
            'ens_stddev_lhh', 'f', ('time', 'nvars_lhh', 'z_l', 'yh', 'xh')
        )

        v_ens_avg_ihh = nc_sumfile.createVariable(
            'ens_avg_ihh', 'f', ('time', 'nvars_ihh', 'z_i', 'yh', 'xh')
        )
        v_ens_stddev_ihh = nc_sumfile.createVariable(
            'ens_stddev_ihh', 'f', ('time', 'nvars_ihh', 'z_i', 'yh', 'xh')
        )
        v_ens_avg_lhq = nc_sumfile.createVariable(
            'ens_avg_lhq', 'f', ('time', 'nvars_lhq', 'z_l', 'yh', 'xq')
        )
        v_ens_stddev_lhq = nc_sumfile.createVariable(
            'ens_stddev_lhq', 'f', ('time', 'nvars_lhq', 'z_l', 'yh', 'xq')
        )
        v_ens_avg_lqh = nc_sumfile.createVariable(
            'ens_avg_lqh', 'f', ('time', 'nvars_lqh', 'z_l', 'yq', 'xh')
        )
        v_ens_stddev_lqh = nc_sumfile.createVariable(
            'ens_stddev_lqh', 'f', ('time', 'nvars_lqh', 'z_l', 'yq', 'xh')
        )
        v_RMSZ = nc_sumfile.createVariable('RMSZ', 'f', ('time', 'nvars', 'ens_size', 'nbin'))

        # Assign var names
        # strings need to be the same length for netcdf (is this still true?)
        if verbose:
            print('VERBOSE: Assigning var name arrays .....')

        eq_all_var_names = []
        eq_lhh_var_names = []
        eq_ihh_var_names = []
        eq_lhq_var_names = []
        eq_lqh_var_names = []

        all_var_names = list(Var_lhh)
        all_var_names += Var_ihh
        all_var_names += Var_lhq
        all_var_names += Var_lqh

        # all vars
        l_eq = len(all_var_names)
        for i in range(l_eq):
            tt = list(all_var_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_all_var_names.append(tt)

        # lhh
        l_eq = len(Var_lhh)
        for i in range(l_eq):
            tt = list(Var_lhh[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_lhh_var_names.append(tt)

        # ihh
        l_eq = len(Var_ihh)
        for i in range(l_eq):
            tt = list(Var_ihh[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_ihh_var_names.append(tt)

        # lhq
        l_eq = len(Var_lhq)
        for i in range(l_eq):
            tt = list(Var_lhq[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_lhq_var_names.append(tt)
        # lqh
        l_eq = len(Var_lqh)
        for i in range(l_eq):
            tt = list(Var_lqh[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_lqh_var_names.append(tt)

        v_vars[:] = eq_all_var_names[:]
        v_var_lhh[:] = eq_lhh_var_names[:]
        v_var_ihh[:] = eq_ihh_var_names[:]
        v_var_lhq[:] = eq_lhq_var_names[:]
        v_var_lqh[:] = eq_lqh_var_names[:]

        # Time-invarient level depths
        if verbose:
            print('VERBOSE: Assigning time invariant metadata .....')

        # TO DO - uncomment later
        # vars_dict = first_file.variables
        # zl_lev_data = vars_dict['z_l']
        # zi_lev_data = vars_dict['z_i']
        # v_zi_lev[:] = zi_lev_data[:]
        # v_zl_lev_data[:] = zl_lev_data[i]

    # end of rank 0

    # All:
    # Time-varient metadata
    if verbose:
        if me.get_rank() == 0:
            print('VERBOSE: Assigning time variant metadata .....')
    # one time value per timeslice (taken from the first file of each slice)
    time_array = np.zeros(len(slice_files), dtype=np.float64)
    for i, full_in_files in enumerate(slice_files):
        with nc.Dataset(full_in_files[0], 'r') as f:
            time_array[i] = f.variables['time'][0]

    # gather time array to root (0)
    if opts_dict['mpi_enable']:
        time_array = pyEnsLib.gather_npArray_pop(time_array, me, (me.get_size(),))
    if me.get_rank() == 0:
        v_time[:] = time_array[:]

    # Assign zero values to first time slice of RMSZ and avg and stddev for 2d & 3d
    # in case of a calculation problem before finishing
    b_size = opts_dict['nbin']

    z_ens_avg_lhh = np.zeros((n_var_lhh, z_l, yh, xh), dtype=np.float32)
    z_ens_stddev_lhh = np.zeros((n_var_lhh, z_l, yh, xh), dtype=np.float32)

    z_ens_avg_ihh = np.zeros((n_var_ihh, z_i, yh, xh), dtype=np.float32)
    z_ens_stddev_ihh = np.zeros((n_var_ihh, z_i, yh, xh), dtype=np.float32)

    z_ens_avg_lhq = np.zeros((n_var_lhq, z_l, yh, xq), dtype=np.float32)
    z_ens_stddev_lhq = np.zeros((n_var_lhq, z_l, yh, xq), dtype=np.float32)

    z_ens_avg_lqh = np.zeros((n_var_lqh, z_l, yq, xh), dtype=np.float32)
    z_ens_stddev_lqh = np.zeros((n_var_lqh, z_l, yq, xh), dtype=np.float32)

    z_RMSZ = np.zeros((nvars, esize, b_size), dtype=np.float32)

    # rank 0 (put zero values in summary file)
    if me.get_rank() == 0:
        v_RMSZ[0, :, :, :] = z_RMSZ[:, :, :]

        v_ens_avg_lhh[0, :, :, :, :] = z_ens_avg_lhh[:, :, :, :]
        v_ens_stddev_lhh[0, :, :, :, :] = z_ens_stddev_lhh[:, :, :, :]

        v_ens_avg_ihh[0, :, :, :, :] = z_ens_avg_ihh[:, :, :, :]
        v_ens_stddev_ihh[0, :, :, :, :] = z_ens_stddev_ihh[:, :, :, :]

        v_ens_avg_lhq[0, :, :, :, :] = z_ens_avg_lhq[:, :, :, :]
        v_ens_stddev_lhq[0, :, :, :, :] = z_ens_stddev_lhq[:, :, :, :]

        v_ens_avg_lqh[0, :, :, :, :] = z_ens_avg_lqh[:, :, :, :]
        v_ens_stddev_lqh[0, :, :, :, :] = z_ens_stddev_lqh[:, :, :, :]

    # close file[0]
    first_file.close()

    # Calculate RMSZ scores
    if verbose and me.get_rank() == 0:
        print('VERBOSE: Calculating RMSZ scores .....')

    # compute each of this rank's timeslices, then stack the results so that
    # every array gets a leading time dimension
    results = []
    for full_in_files in slice_files:
        results.append(
            pyEnsLib.mom6_calc_rmsz(full_in_files, Var_lhh, Var_ihh, Var_lhq, Var_lqh, opts_dict)
        )
    (
        zscore_lhh,
        zscore_ihh,
        zscore_lhq,
        zscore_lqh,
        ens_avg_lhh,
        ens_stddev_lhh,
        ens_avg_ihh,
        ens_stddev_ihh,
        ens_avg_lqh,
        ens_stddev_lqh,
        ens_avg_lhq,
        ens_stddev_lhq,
    ) = [np.stack(r, axis=0) for r in zip(*results)]

    if verbose and me.get_rank() == 0:
        print('VERBOSE: Finished with RMSZ scores .....')

    zmall = np.concatenate((zscore_lhh, zscore_ihh, zscore_lhq, zscore_lhq), axis=1)

    # Collect from all processors (serial already has all timeslices)
    if opts_dict['mpi_enable']:
        # Gather the variable results from all processors to the master processor
        # (each rank has exactly one timeslice, so pass in index 0)
        zmall = pyEnsLib.gather_npArray_pop(
            zmall[0],
            me,
            (
                me.get_size(),
                n_var_lhh + n_var_ihh + n_var_lqh + n_var_lhq,
                esize,
                nbin,
            ),
        )
        ens_avg_lhh = pyEnsLib.gather_npArray_pop(
            ens_avg_lhh[0], me, (me.get_size(), n_var_lhh, z_l, yh, xh)
        )
        ens_avg_ihh = pyEnsLib.gather_npArray_pop(
            ens_avg_ihh[0], me, (me.get_size(), n_var_ihh, z_i, yh, xh)
        )
        ens_avg_lhq = pyEnsLib.gather_npArray_pop(
            ens_avg_lhq[0], me, (me.get_size(), n_var_lhq, z_l, yh, xq)
        )
        ens_avg_lqh = pyEnsLib.gather_npArray_pop(
            ens_avg_lqh[0], me, (me.get_size(), n_var_lqh, z_l, yq, xh)
        )

        ens_stddev_lhh = pyEnsLib.gather_npArray_pop(
            ens_stddev_lhh[0], me, (me.get_size(), n_var_lhh, z_l, yh, xh)
        )
        ens_stddev_ihh = pyEnsLib.gather_npArray_pop(
            ens_stddev_ihh[0], me, (me.get_size(), n_var_ihh, z_i, yh, xh)
        )
        ens_stddev_lhq = pyEnsLib.gather_npArray_pop(
            ens_stddev_lhq[0], me, (me.get_size(), n_var_lhq, z_l, yh, xq)
        )
        ens_stddev_lqh = pyEnsLib.gather_npArray_pop(
            ens_stddev_lqh[0], me, (me.get_size(), n_var_lqh, z_l, yq, xh)
        )


    # Assign to summary file:
    if me.get_rank() == 0:
        #print("RMSZ = ", v_RMSZ.shape)
        v_RMSZ[:, :, :, :] = zmall[:, :, :, :]
        v_ens_avg_lhh[:, :, :, :, :] = ens_avg_lhh[:, :, :, :, :]
        v_ens_stddev_lhh[:, :, :, :, :] = ens_stddev_lhh[:, :, :, :, :]
        v_ens_avg_ihh[:, :, :, :, :] = ens_avg_ihh[:, :, :, :, :]
        v_ens_stddev_ihh[:, :, :, :, :] = ens_stddev_ihh[:, :, :, :, :]
        v_ens_avg_lhq[:, :, :, :, :] = ens_avg_lhq[:, :, :, :, :]
        v_ens_stddev_lhq[:, :, :, :, :] = ens_stddev_lhq[:, :, :, :, :]
        v_ens_avg_lqh[:, :, :, :, :] = ens_avg_lqh[:, :, :, :, :]
        v_ens_stddev_lqh[:, :, :, :, :] = ens_stddev_lqh[:, :, :, :, :]

        print('STATUS: PyEnsSumMom6 has completed.')
        nc_sumfile.close()


if __name__ == '__main__':
    main(sys.argv[1:])
