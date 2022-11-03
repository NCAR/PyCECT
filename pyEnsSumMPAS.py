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
from pyTools import Duplicate, EqualLength, EqualStride

# This routine creates a summary file from an ensemble of MPAS
# output files


def main(argv):

    # Get command line stuff and store in a dictionary
    s = 'tag= compset= esize= tslice= core= model= mesh= sumfile= indir= sumfiledir= mach= verbose jsonfile= mpi_enable   mpi_disable'
    optkeys = s.split()

    try:
        opts, args = getopt.getopt(argv, 'h', optkeys)
    except getopt.GetoptError:
        pyEnsLib.EnsSumMPAS_usage()
        sys.exit(2)

    # Put command line options in a dictionary - also set defaults
    opts_dict = {}

    # Defaults
    opts_dict['model'] = 'mpas'
    opts_dict['core'] = 'atmosphere'
    opts_dict['mesh'] = 'mesh'
    opts_dict['tag'] = 'v.7.1'
    opts_dict['mach'] = 'cheyenne'
    opts_dict['esize'] = 10
    opts_dict['tslice'] = 0
    opts_dict['sumfile'] = 'mpas.ens.summary.nc'
    opts_dict['indir'] = './'
    opts_dict['sumfiledir'] = './'
    opts_dict['jsonfile'] = 'empty_excluded.json'
    opts_dict['verbose'] = False
    opts_dict['mpi_enable'] = True
    opts_dict['mpi_disable'] = False

    # This creates the dictionary of input arguments
    opts_dict = pyEnsLib.getopt_parseconfig(opts, optkeys, 'ES_MPAS', opts_dict)

    verbose = opts_dict['verbose']

    st = opts_dict['esize']
    esize = int(st)

    if not (
        opts_dict['tag']
        and opts_dict['core']
        and opts_dict['mach']
        or opts_dict['mesh']
        or opts_dict['model']
    ):
        print(
            'ERROR: Please specify --tag, --core, --mach, --mesh and --model options  => EXITING....'
        )
        sys.exit()

    if opts_dict['mpi_disable']:
        opts_dict['mpi_enable'] = False

    # Now find file names in indir
    input_dir = opts_dict['indir']
    # The var list that will be excluded
    ex_varlist = []
    inc_varlist = []

    # Create a mpi simplecomm object
    if opts_dict['mpi_enable']:
        me = pyTools.create_comm()
    else:
        me = pyTools.create_comm(not opts_dict['mpi_enable'])

    if me.get_rank() == 0:
        print('STATUS: Running pyEnsSumMPAS.py')

    if me.get_rank() == 0 and verbose:
        print(opts_dict)
        print('STATUS: Ensemble size for summary = ', esize)

    exclude = False
    if me.get_rank() == 0:
        if opts_dict['jsonfile']:
            inc_varlist = []
            # Read in the excluded or included var list
            ex_varlist, exclude = pyEnsLib.read_jsonlist(opts_dict['jsonfile'], 'ES')
            if len(ex_varlist) > 0:
                if ex_varlist[0] == 'JSONERROR':
                    me.abort()
            if exclude is False:
                inc_varlist = ex_varlist
                ex_varlist = []

    # Broadcast the excluded var list to each processor
    if opts_dict['mpi_enable']:
        exclude = me.partition(exclude, func=Duplicate(), involved=True)
        if exclude:
            ex_varlist = me.partition(ex_varlist, func=Duplicate(), involved=True)
        else:
            inc_varlist = me.partition(inc_varlist, func=Duplicate(), involved=True)

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
    nlevp1 = -1
    nsoil = -1
    ncell = -1
    nedge = -1
    nvertex = -1
    # Look at first file and get dims
    input_dims = first_file.dimensions

    for key in input_dims:
        if key == 'nVertLevels':
            nlev = len(input_dims[key])
        elif key == 'nVertLevelsP1':
            nlevp1 = len(input_dims[key])
        elif key == 'nSoilLevels':
            nsoil = len(input_dims[key])
        elif key == 'nCells':
            ncell = len(input_dims[key])
        elif key == 'nEdges':
            nedge = len(input_dims[key])
        elif key == 'nVertices':
            nvertex = len(input_dims[key])

    if nlev == -1 or nlevp1 == -1 or nsoil == -1:
        if me.get_rank() == 0:
            print('ERROR: Need nVertLevels and nVertLevelsP1 and NSoilLevels => EXITING....')
        sys.exit()

    if (ncell == -1) or (nedge == -1) or (nvertex == -1):
        if me.get_rank() == 0:
            print('ERROR: Need nCells and nVertices and nEdges  => EXITING....')
        sys.exit()

    # output dimensions
    if me.get_rank() == 0 and verbose:
        print('nVertLevels = ', nlev)
        print('nVertLevelsP1 = ', nlevp1)
        print('nSoilLevels = ', nsoil)
        print('nCells = ', ncell)
        print('nEdges = ', nedge)
        print('nVertices = ', nvertex)

    # Get all vars (For now include all variables)
    vars_dict_all = first_file.variables

    # Remove the excluded variables (specified in json file) from variable dictionary
    if exclude:
        vars_dict = vars_dict_all.copy()
        for i in ex_varlist:
            if i in vars_dict:
                del vars_dict[i]
    # Given an included var list, remove all the variables that are not on the list
    else:
        vars_dict = vars_dict_all.copy()
        for k, v in vars_dict_all.items():
            if (k not in inc_varlist) and (vars_dict_all[k].typecode() == 'f'):
                del vars_dict[k]

    # We have cell vars and edge vars and vertex vars (and only want time-dependent vars)
    str_size = 0  # longest var name
    cell_names = []
    edge_names = []
    vertex_names = []
    t_dim = 'Time'
    c_dim = 'nCells'
    e_dim = 'nEdges'
    v_dim = 'nVertices'

    # CHECK FOR edge variable u (horizontal wind velocity vector)

    # sort to cell, edge, and vertex (and grab max str_size)
    for k, v in vars_dict.items():
        # var = k
        vd = v.dimensions  # all the variable's dimensions (names)
        if t_dim in vd:
            if c_dim in vd:
                cell_names.append(k)
            elif e_dim in vd:
                edge_names.append(k)
                if k == 'u':
                    print(
                        'Note: We suggest that variable u (Horizontal normal velocity at edges) be excluded from the summary file in favor of uReconstructZonal and uReconstructMeridional (cell variables).'
                    )
            elif v_dim in vd:
                vertex_names.append(k)
            else:
                print(
                    'Note: variable ',
                    k,
                    ' contains time but not cells, edges, or vertices (and will be excluded).',
                )
                continue
            str_size = max(str_size, len(k))

    num_cell = len(cell_names)
    num_edge = len(edge_names)
    num_vertex = len(vertex_names)
    total = num_cell + num_edge + num_vertex

    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Number of variables found:  ', total)
        print(
            'VERBOSE: Cell variables: ',
            num_cell,
            ', Edge variables: ',
            num_edge,
            ' Vertex variables: ',
            num_vertex,
        )

    # Now sort these and combine (this sorts caps first, then lower case -
    # which is what we want)
    cell_names.sort()
    edge_names.sort()
    vertex_names.sort()

    if esize < total:
        if me.get_rank() == 0:
            print(
                '**************************************************************************************************'
            )
            print(
                '  ERROR: the total number of variables '
                + str(total)
                + ' is larger than the number of ensemble files '
                + str(esize)
            )
            print(
                '  Cannot generate ensemble summary file, please remove more variables from your included variable list,'
            )
            print('  or add more variables in your excluded variable list  => EXITING....')
            print(
                '**************************************************************************************************'
            )
        sys.exit()

    # All vars is cell vars, the edge vars, the vertex
    all_var_names = list(cell_names)
    all_var_names += edge_names
    all_var_names += vertex_names

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

    if me.get_rank() == 0:

        if verbose:
            print('VERBOSE: Creating ', this_sumfile, '  ...')

        if os.path.isfile(this_sumfile):
            os.unlink(this_sumfile)

        nc_sumfile = nc.Dataset(this_sumfile, 'w', format='NETCDF4_CLASSIC')

        # Set dimensions
        if verbose:
            print('VERBOSE: Setting dimensions .....')

        nc_sumfile.createDimension('nCells', ncell)
        nc_sumfile.createDimension('nEdges', nedge)
        nc_sumfile.createDimension('nVertices', nvertex)
        nc_sumfile.createDimension('nVertLevels', nlev)
        nc_sumfile.createDimension('nVertLevelsP1', nlevp1)
        nc_sumfile.createDimension('nSoilLevels', nsoil)

        nc_sumfile.createDimension('ens_size', esize)
        nc_sumfile.createDimension('nvars', total)
        nc_sumfile.createDimension('nvarsCell', num_cell)
        nc_sumfile.createDimension('nvarsEdge', num_edge)
        nc_sumfile.createDimension('nvarsVertex', num_vertex)
        nc_sumfile.createDimension('str_size', str_size)

        # Set global attributes
        now = time.strftime('%c')
        if verbose:
            print('VERBOSE: Setting global attributes .....')
        nc_sumfile.creation_date = now
        nc_sumfile.title = 'MPAS verification ensemble summary file'
        nc_sumfile.tag = opts_dict['tag']
        nc_sumfile.model = opts_dict['model']
        nc_sumfile.core = opts_dict['core']
        nc_sumfile.mesh = opts_dict['mesh']
        nc_sumfile.machine = opts_dict['mach']

        # Create variables
        if verbose:
            print('VERBOSE: Creating variables .....')
        # TO DO - any time invarient data that we want?
        # v_lev = nc_sumfile.createVariable('lev', 'f8', ('nlev',))

        v_vars = nc_sumfile.createVariable('vars', 'S1', ('nvars', 'str_size'))
        v_varCell = nc_sumfile.createVariable('varCell', 'S1', ('nvarsCell', 'str_size'))
        v_varEdge = nc_sumfile.createVariable('varEdge', 'S1', ('nvarsEdge', 'str_size'))
        v_varVertex = nc_sumfile.createVariable('varVertex', 'S1', ('nvarsVertex', 'str_size'))

        v_gm = nc_sumfile.createVariable('global_mean', 'f8', ('nvars', 'ens_size'))
        v_standardized_gm = nc_sumfile.createVariable(
            'standardized_gm', 'f8', ('nvars', 'ens_size')
        )
        v_loadings_gm = nc_sumfile.createVariable('loadings_gm', 'f8', ('nvars', 'nvars'))
        v_mu_gm = nc_sumfile.createVariable('mu_gm', 'f8', ('nvars',))
        v_sigma_gm = nc_sumfile.createVariable('sigma_gm', 'f8', ('nvars',))
        v_sigma_scores_gm = nc_sumfile.createVariable('sigma_scores_gm', 'f8', ('nvars',))

        # Assign vars, var3d and var2d
        # strings need to be the same length...
        if verbose:
            print('VERBOSE: Assigning vars ...')

        eq_all_var_names = []
        eq_cell_names = []
        eq_edge_names = []
        eq_vertex_names = []

        l_eq = len(all_var_names)
        for i in range(l_eq):
            tt = list(all_var_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_all_var_names.append(tt)

        l_eq = len(cell_names)
        for i in range(l_eq):
            tt = list(cell_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_cell_names.append(tt)

        l_eq = len(edge_names)
        for i in range(l_eq):
            tt = list(edge_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_edge_names.append(tt)

        l_eq = len(vertex_names)
        for i in range(l_eq):
            tt = list(vertex_names[i])
            l_tt = len(tt)
            if l_tt < str_size:
                extra = list(' ') * (str_size - l_tt)
                tt.extend(extra)
            eq_vertex_names.append(tt)

        v_vars[:] = eq_all_var_names[:]
        v_varCell[:] = eq_cell_names[:]
        v_varEdge[:] = eq_edge_names[:]
        v_varVertex[:] = eq_vertex_names[:]

        # Time-invarient metadata
        # TO DO
        if verbose:
            print('VERBOSE: Assigning time invariant metadata .....')
        # lev_data = first_file.variables['lev']
        # v_lev[:] = lev_data[:]

    # end of rank=0 work

    # All:

    varCell_list_loc = me.partition(cell_names, func=EqualStride(), involved=True)
    varEdge_list_loc = me.partition(edge_names, func=EqualStride(), involved=True)
    varVertex_list_loc = me.partition(vertex_names, func=EqualStride(), involved=True)

    # close first_file
    first_file.close()

    # Calculate global means #
    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Calculating global means .....')

    gmCell, gmEdge, gmVertex = pyEnsLib.generate_global_mean_for_summary_MPAS(
        full_in_files, varCell_list_loc, varEdge_list_loc, varVertex_list_loc, opts_dict
    )

    if me.get_rank() == 0 and verbose:
        print('VERBOSE: Finished calculating global means .....')

    # empty - not excluding any vars due to global means at this time
    var_list = []

    # gather to rank = 0
    if opts_dict['mpi_enable']:

        # Gather the cell variable results from all processors to the master processor
        slice_index = get_stride_list(len(cell_names), me)
        # Gather global means cell results

        # print("MYRANK = ", me.get_rank(), slice_index)
        gmCell = gather_npArray(gmCell, me, slice_index, (len(cell_names), len(full_in_files)))
        # print(gmCell)

        # Gather the edge variable results from all processors to the master processor
        slice_index = get_stride_list(len(edge_names), me)
        # Gather global means edge results
        gmEdge = gather_npArray(gmEdge, me, slice_index, (len(edge_names), len(full_in_files)))

        # Gather the vertex variable results from all processors to the master processor
        slice_index = get_stride_list(len(vertex_names), me)
        # Gather global means vertex results
        gmVertex = gather_npArray(
            gmVertex, me, slice_index, (len(vertex_names), len(full_in_files))
        )

        #################
        # gather variables to exclude (for pre_pca - currently will be empty)
        var_list = gather_list(var_list, me)

    # rank =0 : complete calculations for summary file
    if me.get_rank() == 0:
        gmall = np.concatenate((gmCell, gmEdge, gmVertex), axis=0)
        # save to summary file
        v_gm[:, :] = gmall[:, :]

        # print("gmall = ", gmall)

        # PCA prep and calculation
        (
            mu_gm,
            sigma_gm,
            standardized_global_mean,
            loadings_gm,
            scores_gm,
            b_exit,
        ) = pyEnsLib.pre_PCA(gmall, all_var_names, var_list, me)

        # if PCA calc encounters an error, then remove the summary file and exit
        if b_exit:
            nc_sumfile.close()

            os.unlink(this_sumfile)
            print('STATUS: Summary could not be created.')
            sys.exit(2)

        v_standardized_gm[:, :] = standardized_global_mean[:, :]
        v_mu_gm[:] = mu_gm[:]
        v_sigma_gm[:] = sigma_gm[:]
        v_loadings_gm[:, :] = loadings_gm[:, :]
        v_sigma_scores_gm[:] = scores_gm[:]

        print('STATUS: Summary file is complete.')

        nc_sumfile.close()


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
