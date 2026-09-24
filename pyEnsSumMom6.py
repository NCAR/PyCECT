#!/usr/bin/env python
import configparser
import getopt
import os
import re
import sys
import time

import netCDF4 as nc
import numpy as np

import pyEnsLib
import pyTools
from pyTools import Duplicate, EqualStride


def main(argv):
    # Get command line stuff and store in a dictionary
    s = 'tag= mach= nyear= nmonth= esize= nbin= min_range= maxrange= res= sumfile= indir= jsonfile= verbose mpi_enable mpi_disable nrand= rand seq= jsondir='
    optkeys = s.split()

    try:
        opts, args = getopt.getopt(argv, 'h', optkeys)
    except getopt.GetoptError:
        pyEnsLib.EnsSumPop_usage()
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
    opts_dict['verbose'] = True
    opts_dict['mpi_enable'] = True
    opts_dict['mpi_disable'] = False
    opts_dict['seq'] = 0
    opts_dict['jsondir'] = './'

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
        Var_lhh, Var_ihh, Var_lhq, Var_lqh = pyEnsLib.read_jsonlist(opts_dict['jsonfile'], 'ES_MOM')
        # check for error opening file
        if len(Var_lhh) > 0:
            if Var_lhh[0] == 'JSONERROR':
                me.abort()
        # get max size of var names
        str_size = 0
        for str in Var_lhh:
            if str_size < len(str):
                str_size = len(str)
        for str in Var_ihh:
            if str_size < len(str):
                str_size = len(str)
        for str in Var_lhq:
            if str_size < len(str):
                str_size = len(str)
        for str in Var_lqh:
            if str_size < len(str):
                str_size = len(str)

    # get number of each variable type
    n_var_lhh = len(Var_lhh)
    n_var_ihh = len(Var_ihh)
    n_var_lqh = len(Var_lqh)
    n_var_lhq = len(Var_lhq)

    if me.get_rank() == 0:
        print('STATUS: Running pyEnsSumMom6!')

        if verbose:
            print('VERBOSE: opts_dict = ')
            print(opts_dict)

    in_files = []
    if os.path.exists(input_dir):
        # Get the list of files
        in_files_temp = os.listdir(input_dir)
        in_files = sorted(in_files_temp)
        num_files = len(in_files)
    else:
        if me.get_rank() == 0:
            print('ERROR: Input directory: ', input_dir, ' not found => EXITING....')
        sys.exit(2)

    # make sure we have enough files
    files_needed = opts_dict['nmonth'] * esize * opts_dict['nyear']
    if num_files < files_needed:
        if me.get_rank() == 0:
            print(
                'ERROR: Input directory does not contain enough files (must be esize*nyear*nmonth = ',
                files_needed,
                ' ) and it has only ',
                num_files,
                ' files).',
            )
        sys.exit(2)

    # Don't want more processors than months
    if me.get_size() > opts_dict['nmonth']:
        if me.get_rank() == 0:
            print(
                'ERROR: more processors requested than the number of months. Recommendation is one processor per month (or fewer).'
            )
        sys.exit(2)

    # TO DO (verify/simplify)
    # Partition the input file list (ideally we have one processor per month)
    in_file_list = me.partition(in_files, func=EqualStride(), involved=True)

    # Check the files in the input directory
    full_in_files = []
    if me.get_rank() == 0 and opts_dict['verbose']:
        print('VERBOSE: Input files are:')

    for onefile in in_file_list:
        fname = input_dir + '/' + onefile
        # if opts_dict['verbose']:
        #    print( "my_rank = ", me.get_rank(), "  ", fname)
        if os.path.isfile(fname):
            full_in_files.append(fname)
        else:
            print('ERROR: Could not locate file: ' + fname + ' => EXITING....')
            sys.exit()

    # open just the first file (all procs) to get metadata
    first_file = nc.Dataset(full_in_files[0], 'r')

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
        if key == 'z_i':
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
        sys.exit()

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
        nc_sumfile.createDimension('nvars', n_var_lhh + n_var_ihh + n_var_lhq + n_var_lqh)
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
        v_var_lhq = nc_sumfile.createVariable('var_ihq', 'S1', ('nvars_lhq', 'str_size'))
        v_var_lqh = nc_sumfile.createVariable('var_iqh', 'S1', ('nvars_lqh', 'str_size'))
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
    vars_dict = first_file.variables
    time_value = vars_dict['time']
    time_array = np.array([time_value])

    # gather time array to root (0)
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

    z_RMSZ = np.zeros(((len(all_var_names)), esize, b_size), dtype=np.float32)

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
    ) = pyEnsLib.mom6_calc_rmsz(full_in_files, Var_lhh, Var_ihh, Var_lhq, Var_lqh, opts_dict)

    if verbose and me.get_rank() == 0:
        print('VERBOSE: Finished with RMSZ scores .....')

    # Collect from all processors
    if opts_dict['mpi_enable']:
        # Gather the variable results from all processors to the master processor

        zmall = np.concatenate((zscore_lhh, zscore_ihh, zscore_lqh, zscore_lhq), axis=0)
        zmall = pyEnsLib.gather_npArray_pop(
            zmall,
            me,
            (
                me.get_size(),
                n_var_lhh + n_var_ihh + n_var_lqh + n_var_lhq,
                len(full_in_files),
                nbin,
            ),
        )

        ens_avg_lhh = pyEnsLib.gather_npArray_pop(
            ens_avg_lhh, me, (me.get_size(), n_var_lhh, z_l, yh, xh)
        )
        ens_avg_ihh = pyEnsLib.gather_npArray_pop(
            ens_avg_ihh, me, (me.get_size(), n_var_ihh, z_i, yh, xh)
        )
        ens_avg_lhq = pyEnsLib.gather_npArray_pop(
            ens_avg_lhq, me, (me.get_size(), n_var_lhq, z_l, yh, xq)
        )
        ens_avg_lqh = pyEnsLib.gather_npArray_pop(
            ens_avg_lqh, me, (me.get_size(), n_var_lqh, z_l, yq, xh)
        )

        ens_stddev_lhh = pyEnsLib.gather_npArray_pop(
            ens_stddev_lhh, me, (me.get_size(), n_var_lhh, z_l, yh, xh)
        )
        ens_stddev_ihh = pyEnsLib.gather_npArray_pop(
            ens_stddev_ihh, me, (me.get_size(), n_var_ihh, z_i, yh, xh)
        )
        ens_stddev_lhq = pyEnsLib.gather_npArray_pop(
            ens_stddev_lhq, me, (me.get_size(), n_var_lhq, z_l, yh, xq)
        )
        ens_stddev_lqh = pyEnsLib.gather_npArray_pop(
            ens_stddev_lqh, me, (me.get_size(), n_var_lqh, z_l, yq, xh)
        )

        # Assign to summary file:
        if me.get_rank() == 0:
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
