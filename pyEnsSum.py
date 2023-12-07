#!/usr/bin/env python
import configparser
import getopt
import json
import os
import re
import sys
import time

import netCDF4 as nc
import numpy as np

import pyEnsLib
import pyTools
from pyTools import Duplicate, EqualLength, EqualStride

# This routine creates a summary file from an ensemble of CAM
# output files


def main(argv):
    # Get command line stuff and store in a dictionary
    s = 'tag= compset= esize= tslice= res= sumfile= indir= sumfiledir= mach= verbose jsonfile= mpi_enable maxnorm  popens regx= startMon= endMon= fIndex= mpi_disable'
    optkeys = s.split()
    try:
        opts, args = getopt.getopt(argv, 'h', optkeys)
    except getopt.GetoptError:
        pyEnsLib.EnsSum_usage()
        sys.exit(2)

    # Put command line options in a dictionary - also set defaults
    opts_dict = {}

    # Defaults
    opts_dict['tag'] = 'cesm2_0'
    opts_dict['compset'] = 'F2000climo'
    opts_dict['mach'] = 'cheyenne'
    opts_dict['esize'] = 350
    opts_dict['tslice'] = 1
    opts_dict['res'] = 'f19_f19'
    opts_dict['sumfile'] = 'ens.summary.nc'
    opts_dict['indir'] = './'
    opts_dict['sumfiledir'] = './'
    opts_dict['jsonfile'] = 'empty_excluded.json'
    opts_dict['verbose'] = False
    opts_dict['mpi_enable'] = True
    opts_dict['mpi_disable'] = False
    opts_dict['maxnorm'] = False
    opts_dict['popens'] = False
    opts_dict['regx'] = 'test'
    opts_dict['startMon'] = 1
    opts_dict['endMon'] = 1
    opts_dict['fIndex'] = 151

    # This creates the dictionary of input arguments
    opts_dict = pyEnsLib.getopt_parseconfig(opts, optkeys, 'ES', opts_dict)

    verbose = opts_dict['verbose']

    st = opts_dict['esize']
    esize = int(st)

    if opts_dict['popens']:
        print('ERROR: Please use pyEnsSumPop.py for a POP ensemble (not --popens)  => EXITING....')
        sys.exit()

    if not (opts_dict['tag'] and opts_dict['compset'] and opts_dict['mach'] or opts_dict['res']):
        print('ERROR: Please specify --tag, --compset, --mach and --res options  => EXITING....')
        sys.exit()

    if opts_dict['mpi_disable']:
        opts_dict['mpi_enable'] = False

    # Now find file names in indir
    input_dir = opts_dict['indir']
    # The var list that will be excluded
    ex_varlist = []

    # Create a mpi simplecomm object
    if opts_dict['mpi_enable']:
        me = pyTools.create_comm()
    else:
        me = pyTools.create_comm(not opts_dict['mpi_enable'])

    if me.get_rank() == 0:
        print('STATUS: Running pyEnsSum.py')

    if me.get_rank() == 0 and verbose:
        print(opts_dict)
        print('STATUS: Ensemble size for summary = ', esize)

    if me.get_rank() == 0:
        if opts_dict['jsonfile']:
            # Read in the excluded var list
            ex_varlist, exclude = pyEnsLib.read_jsonlist(opts_dict['jsonfile'], 'ES')
            if len(ex_varlist) > 0:
                if ex_varlist[0] == 'JSONERROR':
                    me.abort()

    # Broadcast the excluded var list to each processor
    if opts_dict['mpi_enable']:
        ex_varlist = me.partition(ex_varlist, func=Duplicate(), involved=True)

    in_files = []
    if os.path.exists(input_dir):
        # Get the list of files
        in_files_temp = os.listdir(input_dir)
        in_files = sorted(in_files_temp)

        # Make sure we have enough
        num_files = len(in_files)
        if me.get_rank() == 0 and verbose:
            print('VERBOSE: Number of files in input directory = ', num_files)
        if num_files < esize:
            if me.get_rank() == 0 and verbose:
                print(
                    'VERBOSE: Number of files in input directory (',
                    num_files,
                    ') is less than specified ensemble size of ',
                    esize,
                )
            sys.exit(2)
        if num_files > esize:
            if me.get_rank() == 0 and verbose:
                print(
                    'VERBOSE: Note that the number of files in ',
                    input_dir,
                    'is greater than specified ensemble size of ',
                    esize,
                    '\nwill just use the first ',
                    esize,
                    'files',
                )
    else:
        if me.get_rank() == 0:
            print('ERROR: Input directory: ', input_dir, ' not found')
        sys.exit(2)

    # Check full file names in input directory (don't open yet)
    full_in_files = []
    if me.get_rank() == 0 and opts_dict['verbose']:
        print('VERBOSE: Input files are: ')

    for onefile in in_files[0:esize]:
        fname = input_dir + '/' + onefile
        if me.get_rank() == 0 and opts_dict['verbose']:
            print(fname)
        if os.path.isfile(fname):
            full_in_files.append(fname)
        else:
            if me.get_rank() == 0:
                print('ERROR: Could not locate file ', fname, ' => EXITING....')
            sys.exit()

    # open just the first file
    first_file = nc.Dataset(full_in_files[0], 'r')

    # Store dimensions of the input fields
    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Getting spatial dimensions')
    nlev = -1
    nilev = -1
    ncol = -1
    nlat = -1
    nlon = -1
    # Look at first file and get dims
    input_dims = first_file.dimensions
    # ndims = len(input_dims)

    for key in input_dims:
        if key == 'lev':
            nlev = len(input_dims['lev'])
        elif key == 'ilev':
            nilev = len(input_dims['ilev'])
        elif key == 'ncol':
            ncol = len(input_dims['ncol'])
        elif (key == 'nlon') or (key == 'lon'):
            nlon = len(input_dims[key])
        elif (key == 'nlat') or (key == 'lat'):
            nlat = len(input_dims[key])

    if nlev == -1:
        if me.get_rank() == 0:
            print('ERROR: could not locate a valid dimension (lev) => EXITING....')
        sys.exit()

    if (ncol == -1) and ((nlat == -1) or (nlon == -1)):
        if me.get_rank() == 0:
            print('ERROR: Need either lat/lon or ncol  => EXITING....')
        sys.exit()

    # Check if this is SE or FV data
    if ncol != -1:
        is_SE = True
    else:
        is_SE = False

    # output dimensions
    if me.get_rank() == 0 and verbose:
        print('lev = ', nlev)
        if is_SE:
            print('ncol = ', ncol)
        else:
            print('nlat = ', nlat)
            print('nlon = ', nlon)

    # invarient metadata  (will write to sum file later)
    lev_data = first_file.variables['lev']
    lev_data_copy = lev_data[:]  # doesn't go away when close first_file

    # Get 2d vars, 3d vars and all vars (For now include all variables)
    vars_dict_all = first_file.variables

    # Remove the excluded variables (specified in json file) from variable dictionary
    vars_dict = vars_dict_all.copy()
    for i in ex_varlist:
        if i in vars_dict:
            del vars_dict[i]

    # num_vars = len(vars_dict)

    str_size = 0
    d2_var_names = []
    d3_var_names = []
    num_2d = 0
    num_3d = 0

    # Which are 2d, which are 3d and max str_size
    for k, v in vars_dict.items():
        # var = k
        # vd = v.dimensions  # all the variable's dimensions (names)
        vr = len(v.dimensions)  # num dimension
        vs = v.shape  # dim values
        is_2d = False
        is_3d = False
        if is_SE:  # (time, lev, ncol) or (time, ncol)
            if (vr == 2) and (vs[1] == ncol):
                is_2d = True
                num_2d += 1
            elif (vr == 3) and (vs[2] == ncol and vs[1] == nlev):
                is_3d = True
                num_3d += 1
        else:  # (time, lev, nlon, nlon) or (time, nlat, nlon)
            if (vr == 3) and (vs[1] == nlat and vs[2] == nlon):
                is_2d = True
                num_2d += 1
            elif (vr == 4) and (
                vs[2] == nlat and vs[3] == nlon and (vs[1] == nlev or vs[1] == nilev)
            ):
                is_3d = True
                num_3d += 1

        if is_3d:
            str_size = max(str_size, len(k))
            d3_var_names.append(k)
        elif is_2d:
            str_size = max(str_size, len(k))
            d2_var_names.append(k)

    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Number of variables found:  ', num_3d + num_2d)
        print('VERBOSE: 3D variables: ' + str(num_3d) + ', 2D variables: ' + str(num_2d))

    # Now sort these and combine (this sorts caps first, then lower case -
    # which is what we want)
    d2_var_names.sort()
    d3_var_names.sort()

    if esize < num_2d + num_3d:
        if me.get_rank() == 0:
            print(
                '************************************************************************************************************************************'
            )
            print(
                '  ERROR: the total number of 3D and 2D variables '
                + str(num_2d + num_3d)
                + ' is larger than the number of ensemble files '
                + str(esize)
            )
            print(
                '  Cannot generate ensemble summary file, please remove more variables from your included variable list,'
            )
            print('  or add more variables in your excluded variable list  => EXITING....')
            print(
                '************************************************************************************************************************************'
            )
        sys.exit()
    # All vars is 3d vars first (sorted), the 2d vars
    all_var_names = list(d3_var_names)
    all_var_names += d2_var_names

    # Rank 0 - Create new summary ensemble file
    this_sumfile = opts_dict['sumfile']

    # check if directory is valid
    sum_dir = os.path.dirname(this_sumfile)
    if len(sum_dir) == 0:
        sum_dir = '.'
    if os.path.exists(sum_dir) is False:
        if me.get_rank() == 0:
            print('ERROR: Summary file directory: ', sum_dir, ' not found')
        sys.exit(2)

    this_sumfile = sum_dir + '/' + this_sumfile

    # All:
    var3_list_loc = me.partition(d3_var_names, func=EqualStride(), involved=True)
    var2_list_loc = me.partition(d2_var_names, func=EqualStride(), involved=True)

    # close first_file
    first_file.close()

    # Calculate global means #
    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Calculating global means .....')

    gm3d, gm2d = pyEnsLib.generate_global_mean_for_summary(
        full_in_files, var3_list_loc, var2_list_loc, is_SE, opts_dict
    )

    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Finished calculating global means .....')

    # gather to rank = 0
    if opts_dict['mpi_enable']:
        # Gather the 3d variable results from all processors to the master processor
        slice_index = get_stride_list(len(d3_var_names), me)

        # Gather global means 3d results
        gm3d = gather_npArray(gm3d, me, slice_index, (len(d3_var_names), len(full_in_files)))

        # Gather 2d variable results from all processors to the master processor
        slice_index = get_stride_list(len(d2_var_names), me)

        # Gather global means 2d results
        gm2d = gather_npArray(gm2d, me, slice_index, (len(d2_var_names), len(full_in_files)))

    # rank =0 : complete calculations for summary file
    if me.get_rank() == 0:
        gmall = np.concatenate((gm3d, gm2d), axis=0)

        # PCA prep and calculation
        (
            mu_gm,
            sigma_gm,
            standardized_global_mean,
            loadings_gm,
            scores_gm,
            new_ex_varlist,
            new_gmall,
            b_exit,
        ) = pyEnsLib.pre_PCA(gmall, all_var_names, ex_varlist, me)

        # if PCA calc encounters an error, then remove the summary file and exit
        if b_exit:
            print('STATUS: Summary could not be created.')
            sys.exit(2)

        # update json file?  update var 2d and 3d var lists?

        # print('ex_varlist len = ', len(ex_varlist))
        # print('new ex_varlist len = ', len(new_ex_varlist))
        # print(new_ex_varlist)

        if len(ex_varlist) < len(new_ex_varlist):
            print('STATUS: Creating an updated JSON file (with prefix "NEW.")')
            new_name = 'NEW.' + opts_dict['jsonfile']
            print(
                'STATUS: Adding ', len(new_ex_varlist) - len(ex_varlist), ' variables to ', new_name
            )
            jdict = {}
            jdict['ExcludedVar'] = new_ex_varlist
            with open(new_name, 'w') as outfile:
                json.dump(jdict, outfile)

            # update num_2d, num_3d => by removing vars from  d3_var_names and d2_var_names
            for i in new_ex_varlist:
                if i in all_var_names:
                    all_var_names.remove(i)
                if i in d3_var_names:
                    d3_var_names.remove(i)
                elif i in d2_var_names:
                    d2_var_names.remove(i)

            num_2d = len(d2_var_names)
            num_3d = len(d3_var_names)

            nvars = loadings_gm.shape[0]
            if nvars != (num_2d + num_3d):
                print('DIMENSION ERROR!')
                print('STATUS: Summary could not be created.')
                sys.exit(2)

        # create the summary file  (still rank 0)
        if verbose:
            print('VERBOSE: Creating ', this_sumfile, '  ...')

        if os.path.isfile(this_sumfile):
            os.unlink(this_sumfile)

        nc_sumfile = nc.Dataset(this_sumfile, 'w', format='NETCDF4_CLASSIC')

        # Set dimensions
        if verbose:
            print('VERBOSE: Setting dimensions .....')
        if is_SE:
            nc_sumfile.createDimension('ncol', ncol)
        else:
            nc_sumfile.createDimension('nlat', nlat)
            nc_sumfile.createDimension('nlon', nlon)

        nc_sumfile.createDimension('nlev', nlev)
        nc_sumfile.createDimension('ens_size', esize)
        nc_sumfile.createDimension('nvars', num_3d + num_2d)
        nc_sumfile.createDimension('nvars3d', num_3d)
        nc_sumfile.createDimension('nvars2d', num_2d)
        nc_sumfile.createDimension('str_size', str_size)

        # Set global attributes
        now = time.strftime('%c')
        if verbose:
            print('VERBOSE: Setting global attributes .....')
        nc_sumfile.creation_date = now
        nc_sumfile.title = 'CAM verification ensemble summary file'
        nc_sumfile.tag = opts_dict['tag']
        nc_sumfile.compset = opts_dict['compset']
        nc_sumfile.resolution = opts_dict['res']
        nc_sumfile.machine = opts_dict['mach']

        # Create variables
        if verbose:
            print('VERBOSE: Creating variables .....')
        v_lev = nc_sumfile.createVariable('lev', 'f8', ('nlev',))
        v_vars = nc_sumfile.createVariable('vars', 'S1', ('nvars', 'str_size'))
        v_var3d = nc_sumfile.createVariable('var3d', 'S1', ('nvars3d', 'str_size'))
        v_var2d = nc_sumfile.createVariable('var2d', 'S1', ('nvars2d', 'str_size'))

        v_gm = nc_sumfile.createVariable('global_mean', 'f8', ('nvars', 'ens_size'))
        v_standardized_gm = nc_sumfile.createVariable(
            'standardized_gm', 'f8', ('nvars', 'ens_size')
        )
        v_loadings_gm = nc_sumfile.createVariable('loadings_gm', 'f8', ('nvars', 'nvars'))
        v_mu_gm = nc_sumfile.createVariable('mu_gm', 'f8', ('nvars',))
        v_sigma_gm = nc_sumfile.createVariable('sigma_gm', 'f8', ('nvars',))
        v_sigma_scores_gm = nc_sumfile.createVariable('sigma_scores_gm', 'f8', ('nvars',))

        # Assign vars, var3d and var2d
        if verbose:
            print('VERBOSE: Assigning vars, var3d, and var2d .....')

        eq_all_var_names = []
        eq_d3_var_names = []
        eq_d2_var_names = []

        l_eq = len(all_var_names)
        for i in range(l_eq):
            tt = list(all_var_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_all_var_names.append(tt)

        l_eq = len(d3_var_names)
        for i in range(l_eq):
            tt = list(d3_var_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_d3_var_names.append(tt)

        l_eq = len(d2_var_names)
        for i in range(l_eq):
            tt = list(d2_var_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_d2_var_names.append(tt)

        v_vars[:] = eq_all_var_names[:]
        v_var3d[:] = eq_d3_var_names[:]
        v_var2d[:] = eq_d2_var_names[:]

        # populate variables
        v_lev[:] = lev_data_copy[:]
        v_gm[:, :] = new_gmall[:, :]
        v_standardized_gm[:, :] = standardized_global_mean[:, :]
        v_mu_gm[:] = mu_gm[:]
        v_sigma_gm[:] = sigma_gm[:]
        v_loadings_gm[:, :] = loadings_gm[:, :]
        v_sigma_scores_gm[:] = scores_gm[:]

        print('STATUS: Summary file is complete.')

        nc_sumfile.close()


def get_cumul_filelist(opts_dict, indir, regx):
    if not opts_dict['indir']:
        print('input dir is not specified')
        sys.exit(2)
    regx_list = ['mon', 'gnu', 'pgi']
    all_files = []
    for prefix in regx_list:
        for i in range(opts_dict['fIndex'], opts_dict['fIndex'] + opts_dict['esize'] / 3):
            for j in range(opts_dict['startMon'], opts_dict['endMon'] + 1):
                mon_str = str(j).zfill(2)
                regx = '(^' + prefix + '(.)*' + str(i) + '(.)*-(' + mon_str + '))'
                # print 'regx=',regx
                res = [f for f in os.listdir(indir) if re.search(regx, f)]
                in_files = sorted(res)
                all_files.extend(in_files)

    return all_files


#
# Get the shape of all variable list in tuple for all processor
#
def get_shape(shape_tuple, shape1, rank):
    lst = list(shape_tuple)
    lst[0] = shape1
    shape_tuple = tuple(lst)
    return shape_tuple


#
# Get the mpi partition list for each processor
#
def get_stride_list(len_of_list, me):
    slice_index = []
    for i in range(me.get_size()):
        index_arr = np.arange(len_of_list)
        slice_index.append(index_arr[i :: me.get_size()])
    return slice_index


def gather_list(var_list, me):
    whole_list = []
    if me.get_rank() == 0:
        whole_list.extend(var_list)
    for i in range(1, me.get_size()):
        if me.get_rank() == 0:
            rank_id, var_list = me.collect()
            whole_list.extend(var_list)
    if me.get_rank() != 0:
        me.collect(var_list)
    me.sync()
    return whole_list


#
# Gather arrays from each processor by the var_list to the master processor and make it an array
#
def gather_npArray(npArray, me, slice_index, array_shape):
    the_array = np.zeros(array_shape, dtype=np.float64)
    if me.get_rank() == 0:
        k = 0
        for j in slice_index[me.get_rank()]:
            the_array[j, :] = npArray[k, :]
            k = k + 1
    for i in range(1, me.get_size()):
        if me.get_rank() == 0:
            rank, npArray = me.collect()
            k = 0
            for j in slice_index[rank]:
                the_array[j, :] = npArray[k, :]
                k = k + 1
    if me.get_rank() != 0:
        # message = {'from_rank': me.get_rank(), 'shape': npArray.shape}
        me.collect(npArray)
    me.sync()
    return the_array


if __name__ == '__main__':
    main(sys.argv[1:])
