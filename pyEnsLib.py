#!/usr/bin/env python
import configparser
import fnmatch
import getopt
import glob
import itertools
import json
import os
import random
import re
import sys
import time
from itertools import islice

import netCDF4 as nc
import numpy as np
from scipy import linalg as sla

from EET import exhaustive_test

#
# Parse header file of a netcdf to get the variable 3d/2d/1d list
#


def parse_header_file(filename):
    command = 'ncdump -h ' + filename
    print(command)

    retvalue = os.popen(command).readline()
    print(retvalue)


#
# Create RMSZ zscores for ensemble file sets
# o_files are not open
# this is used for POP


def calc_rmsz(o_files, var_name3d, var_name2d, opts_dict):
    threshold = 1e-12
    popens = opts_dict['popens']
    tslice = opts_dict['tslice']
    nbin = opts_dict['nbin']
    minrange = opts_dict['minrange']
    maxrange = opts_dict['maxrange']

    if not popens:
        print('ERROR: should not be calculating rmsz for CAM => EXITING')
        sys.exit(2)

    first_file = nc.Dataset(o_files[0], 'r')
    input_dims = first_file.dimensions

    # Create array variables
    nlev = len(input_dims['z_t'])
    if 'nlon' in input_dims:
        nlon = len(input_dims['nlon'])
        nlat = len(input_dims['nlat'])
    elif 'lon' in input_dims:
        nlon = len(input_dims['lon'])
        nlat = len(input_dims['lat'])

    output3d = np.zeros((len(o_files), nlev, nlat, nlon), dtype=np.float32)
    output2d = np.zeros((len(o_files), nlat, nlon), dtype=np.float32)

    ens_avg3d = np.zeros((len(var_name3d), nlev, nlat, nlon), dtype=np.float32)
    ens_stddev3d = np.zeros((len(var_name3d), nlev, nlat, nlon), dtype=np.float32)
    ens_avg2d = np.zeros((len(var_name2d), nlat, nlon), dtype=np.float32)
    ens_stddev2d = np.zeros((len(var_name2d), nlat, nlon), dtype=np.float32)

    Zscore3d = np.zeros((len(var_name3d), len(o_files), (nbin)), dtype=np.float32)
    Zscore2d = np.zeros((len(var_name2d), len(o_files), (nbin)), dtype=np.float32)

    first_file.close()

    # open all of the files at once
    # (not too many for pop - and no longer doing this for cam)
    handle_o_files = []
    for fname in o_files:
        handle_o_files.append(nc.Dataset(fname, 'r'))

    # Now lOOP THROUGH 3D
    for vcount, vname in enumerate(var_name3d):
        # Read in vname's data from all ens. files
        for fcount, this_file in enumerate(handle_o_files):
            data = this_file.variables[vname]
            output3d[fcount, :, :, :] = data[tslice, :, :, :]

        # for this variable, Generate ens_avg and ens_stddev to store in the ensemble summary file
        moutput3d = np.ma.masked_values(output3d, data._FillValue)
        ens_avg3d[vcount] = np.ma.average(moutput3d, axis=0)
        ens_stddev3d[vcount] = np.ma.std(moutput3d, axis=0, dtype=np.float32)

        # Generate avg, stddev and zscore for this 3d variable
        for fcount, this_file in enumerate(handle_o_files):
            data = this_file.variables[vname]
            # rmask contains a number for each grid point indicating it's region
            rmask = this_file.variables['REGION_MASK']
            Zscore = pop_zpdf(
                output3d[fcount],
                nbin,
                (minrange, maxrange),
                ens_avg3d[vcount],
                ens_stddev3d[vcount],
                data._FillValue,
                threshold,
                rmask,
                opts_dict,
            )
            Zscore3d[vcount, fcount, :] = Zscore[:]

    # LOOP THROUGH 2D
    for vcount, vname in enumerate(var_name2d):
        # Read in vname's data of all files
        for fcount, this_file in enumerate(handle_o_files):
            data = this_file.variables[vname]
            output2d[fcount, :, :] = data[tslice, :, :]

        # Generate ens_avg and esn_stddev to store in the ensemble summary file
        moutput2d = np.ma.masked_values(output2d, data._FillValue)
        ens_avg2d[vcount] = np.ma.average(moutput2d, axis=0)
        ens_stddev2d[vcount] = np.ma.std(moutput2d, axis=0, dtype=np.float32)

        # Generate avg, stddev and zscore for 3d variable
        for fcount, this_file in enumerate(handle_o_files):
            data = this_file.variables[vname]

            rmask = this_file.variables['REGION_MASK']
            Zscore = pop_zpdf(
                output2d[fcount],
                nbin,
                (minrange, maxrange),
                ens_avg2d[vcount],
                ens_stddev2d[vcount],
                data._FillValue,
                threshold,
                rmask,
                opts_dict,
            )
            Zscore2d[vcount, fcount, :] = Zscore[:]

    # close files
    for this_file in handle_o_files:
        this_file.close()

    return Zscore3d, Zscore2d, ens_avg3d, ens_stddev3d, ens_avg2d, ens_stddev2d


#
# Calculate pop zscore pass rate (ZPR) or pop zpdf values
#
def pop_zpdf(
    input_array, nbin, zrange, ens_avg, ens_stddev, FillValue, threshold, rmask, opts_dict
):
    if 'test_failure' in opts_dict:
        test_failure = opts_dict['test_failure']
    else:
        test_failure = False

    #   print("input_array.ndim = ", input_array.ndim)

    # Masked out the missing values (land)
    moutput = np.ma.masked_values(input_array, FillValue)
    if input_array.ndim == 3:
        rmask3d = np.zeros(input_array.shape, dtype=np.int32)
        for i in rmask3d:
            i[:, :] = rmask[:, :]
        rmask_array = rmask3d
    elif input_array.ndim == 2:
        rmask_array = np.zeros(input_array.shape, dtype=np.int32)
        rmask_array[:, :] = rmask[:, :]

    # Now we just want the open oceans (not marginal seas)
    # - so for g1xv7, those are 1,2,3,4,6
    # in the region mask - so we don't want rmask<1 or rmask>6
    moutput2 = np.ma.masked_where((rmask_array < 1) | (rmask_array > 6), moutput)

    # Use the masked array moutput2 to calculate Zscore_temp=(data-avg)/stddev
    Zscore_temp = np.fabs(
        (moutput2.astype(np.float64) - ens_avg)
        / np.where(ens_stddev <= threshold, FillValue, ens_stddev)
    )

    # To retrieve only the valid entries of Zscore_temp
    Zscore_nomask = Zscore_temp[~Zscore_temp.mask]

    # If just test failure, calculate ZPR only (DEFAULT - not chnagable via cmd line
    if test_failure:
        # Zpr=the count of Zscore_nomask is less than pop_tol (3.0)/ the total count of Zscore_nomask
        Zpr = np.where(Zscore_nomask <= opts_dict['pop_tol'])[0].size / float(Zscore_temp.count())
        return Zpr

    # Else calculate zpdf and return as zscore
    # Count the unmasked value
    count = Zscore_temp.count()

    Zscore, bins = np.histogram(Zscore_temp.compressed(), bins=nbin, range=zrange)

    # Normalize the number by dividing the count
    if count != 0:
        Zscore = Zscore.astype(np.float32) / count
    else:
        print(('count=0,sum=', np.sum(Zscore)))
    return Zscore


#
# Calculate rmsz score by compare the run file with the ensemble summary file
#
def calculate_raw_score(
    k, v, npts3d, npts2d, ens_avg, ens_stddev, is_SE, opts_dict, FillValue, timeslice, rmask
):
    count = 0
    Zscore = 0
    threshold = 1.0e-12
    has_zscore = True
    popens = opts_dict['popens']
    if popens:  # POP
        minrange = opts_dict['minrange']
        maxrange = opts_dict['maxrange']
        Zscore = pop_zpdf(
            v,
            opts_dict['nbin'],
            (minrange, maxrange),
            ens_avg,
            ens_stddev,
            FillValue,
            threshold,
            rmask,
            opts_dict,
        )
    else:  # CAM
        if k in ens_avg:
            if is_SE:
                if ens_avg[k].ndim == 1:
                    npts = npts2d
                else:
                    npts = npts3d
            else:
                if ens_avg[k].ndim == 2:
                    npts = npts2d
                else:
                    npts = npts3d

            count, return_val = calc_Z(
                v, ens_avg[k].astype(np.float64), ens_stddev[k].astype(np.float64), count, False
            )
            Zscore = np.sum(np.square(return_val.astype(np.float64)))
            if npts == count:
                Zscore = 0
            else:
                Zscore = np.sqrt(Zscore / (npts - count))
        else:
            has_zscore = False

    return Zscore, has_zscore


#
# Find the corresponding ensemble summary file from directory
# /glade/p/cesmdata/cseg/inputdata/validation/ when three
# validation files are input from the web server
#
# ifiles are not open
def search_sumfile(opts_dict, ifiles):
    sumfile_dir = opts_dict['sumfile']
    first_file = nc.Dataset(ifiles[0], 'r')
    machineid = ''
    compiler = ''

    global_att = first_file.ncattrs()
    for attr_name in global_att:
        val = getattr(first_file, attr_name)
        if attr_name == 'model_version':
            if val.find('-') != -1:
                model_version = val[0 : val.find('-')]
            else:
                model_version = val
        elif attr_name == 'compset':
            compset = val
        elif attr_name == 'testtype':
            testtype = val
            if val == 'UF-ECT':
                testtype = 'uf_ensembles'
                opts_dict['eet'] = len(ifiles)
            elif val == 'ECT':
                testtype = 'ensembles'
            elif val == 'POP':
                testtype = val + '_ensembles'
        elif attr_name == 'machineid':
            machineid = val
        elif attr_name == 'compiler':
            compiler = val
        elif attr_name == 'grid':
            grid = val

    if 'testtype' in global_att:
        sumfile_dir = sumfile_dir + '/' + testtype + '/'
    else:
        print('ERROR: No global attribute testtype in your validation file => EXITING....')
        sys.exit(2)

    if 'model_version' in global_att:
        sumfile_dir = sumfile_dir + '/' + model_version + '/'
    else:
        print('ERROR: No global attribute model_version in your validation file => EXITING....')
        sys.exit(2)

    first_file.close()

    if os.path.exists(sumfile_dir):
        thefile_id = 0
        for i in os.listdir(sumfile_dir):
            if os.path.isfile(sumfile_dir + i):
                sumfile_id = nc.Dataset(sumfile_dir + i, 'r')
                sumfile_gatt = sumfile_id.ncattrs()
                if 'grid' not in sumfile_gatt and 'resolution' not in sumfile_gatt:
                    print(
                        'ERROR: No global attribute grid or resolution in the summary file => EXITING....'
                    )
                    sys.exit(2)
                if 'compset' not in sumfile_gatt:
                    print('ERROR: No global attribute compset in the summary file')
                    sys.exit(2)
                if (
                    getattr(sumfile_id, 'resolution') == grid
                    and getattr(sumfile_id, 'compset') == compset
                ):
                    thefile_id = sumfile_id
                sumfile_id.close()
        if thefile_id == 0:
            print(
                "ERROR: The verification files don't have a matching ensemble summary file to compare => EXITING...."
            )
            sys.exit(2)
    else:
        print(('ERROR: Could not locate directory ' + sumfile_dir + ' => EXITING....'))
        sys.exit(2)

    return sumfile_dir + i, machineid, compiler


#
# Create some variables and call a function to calculate PCA
# now gm comes in at 64 bits...


# pas in exclude list in case we have to add to id
def pre_PCA(gm_orig, all_var_names, ex_list, me):
    # initialize
    b_exit = False
    gm_len = gm_orig.shape
    nvar = gm_len[0]
    nfile = gm_len[1]
    if gm_orig.dtype == np.float32:
        gm = gm_orig.astype(np.float64)
    else:
        gm = gm_orig[:]

    sigma_gm = np.std(gm, axis=1, ddof=1)

    # keep track of orig vars in exclude file
    new_ex_list = ex_list.copy()
    orig_len = len(ex_list)

    ##### check for constants across ensemble
    print('STATUS: checking for constant values across ensemble')
    for var in range(nvar):
        for file in range(nfile):
            if np.any(sigma_gm[var] == 0.0) and all_var_names[var] not in set(new_ex_list):
                # keep track of zeros standard deviations and append
                new_ex_list.append(all_var_names[var])

    # did we add vars to exclude?
    new_len = len(new_ex_list)
    if new_len > orig_len:
        sub_list = new_ex_list[orig_len:]
        if me.get_rank() == 0:
            print('\n')
            print(
                '*************************************************************************************'
            )
            print(
                'Warning: these ',
                new_len - orig_len,
                ' variables are constant across ensemble members, and will be excluded and added to a copy of the json file (--jsonfile): ',
            )
            print('\n')
            print((','.join(['"{0}"'.format(item) for item in sub_list])))
            print(
                '*************************************************************************************'
            )
            print('\n')

    #### now check for any variables that have less than 3% (of the ensemble size) unique values
    print('STATUS: checking for unique values across ensemble')
    cts = np.count_nonzero(np.diff(np.sort(gm)), axis=1) + 1
    thresh = 0.03 * gm.shape[1]
    result = np.where(cts < thresh)
    indices = result[0]
    if len(indices) > 0:
        nu_list = []
        for i in indices:
            # only add if not in ex_list already
            if all_var_names[i] not in set(new_ex_list):
                nu_list.append(all_var_names[i])

        if len(nu_list) > 0:
            print('\n')
            print(
                '********************************************************************************************'
            )
            print(
                'Warning: these ',
                len(nu_list),
                ' variables contain fewer than 3% unique values across the ensemble, and will be excluded and added to a copy of the json file (--jsonfile): ',
            )
            print('\n')
            print((','.join(['"{0}"'.format(item) for item in nu_list])))
            print(
                '********************************************************************************************'
            )
            print('\n')

            new_ex_list.extend(nu_list)

    ### REMOVE newly excluded stuff before the check for linear dependence
    # remove var from nvar, all_var_names, gm, and recalculate: mu_gm, sigma_gm
    new_len = len(new_ex_list)
    indx = []
    if new_len > orig_len:
        print('Updating ...')
        sub_list = new_ex_list[orig_len:]
        for i in sub_list:
            indx.append(all_var_names.index(i))
        # now delete the rows from gm and names from list
        gm_del = np.delete(gm, indx, axis=0)
        all_var_names_del = np.delete(all_var_names, indx).tolist()

        gm = gm_del
        all_var_names = all_var_names_del
        nvar = gm.shape[0]

    mu_gm = np.average(gm, axis=1)
    sigma_gm = np.std(gm, axis=1, ddof=1)
    standardized_global_mean = np.zeros(gm.shape, dtype=np.float64)

    ####### check for linear dependent vars
    print('STATUS: checking for linear dependence across ensemble')
    for var in range(nvar):
        for file in range(nfile):
            standardized_global_mean[var, file] = (gm[var, file] - mu_gm[var]) / sigma_gm[var]

    eps = np.finfo(np.float32).eps
    norm = np.linalg.norm(standardized_global_mean, ord=2)
    sh = max(standardized_global_mean.shape)
    mytol = sh * norm * eps

    # standardized_rank = np.linalg.matrix_rank(standardized_global_mean, mytol)
    print('STATUS: using QR...')
    print('sh, norm, eps ', sh, norm, eps)

    dep_var_list = get_dependent_vars_index(standardized_global_mean, mytol)
    num_dep = len(dep_var_list)
    new_len = len(new_ex_list)

    for i in dep_var_list:
        new_ex_list.append(all_var_names[i])

    if num_dep > 0:
        sub_list = new_ex_list[new_len:]

        print('\n')
        print(
            '********************************************************************************************'
        )
        print(
            'Warning: these ',
            num_dep,
            ' variables are linearly dependent, and will be excluded and added to a copy of the json file (--jsonfile): ',
        )
        print('\n')
        print((','.join(['"{0}"'.format(item) for item in sub_list])))
        print(
            '********************************************************************************************'
        )
        print('\n')

        # REMOVE FROM gm, standardized gm and names
        indx = []
        for i in sub_list:
            indx.append(all_var_names.index(i))
        # now delete the rows in index from gm, std gm, and names from list
        gm_del = np.delete(gm, indx, axis=0)
        sgm_del = np.delete(standardized_global_mean, indx, axis=0)
        all_var_names_del = np.delete(all_var_names, indx).tolist()

        gm = gm_del
        standardized_global_mean = sgm_del
        all_var_names = all_var_names_del
        nvar = gm.shape[0]

        mu_gm = np.average(gm, axis=1)
        sigma_gm = np.std(gm, axis=1, ddof=1)

    # COMPUTE PCA
    scores_gm = np.zeros(gm.shape, dtype=np.float64)
    # find principal components
    loadings_gm = princomp(standardized_global_mean)
    # now do coord transformation on the standardized means to get the scores
    scores_gm = np.dot(loadings_gm.T, standardized_global_mean)
    sigma_scores_gm = np.std(scores_gm, axis=1, ddof=1)

    return (
        mu_gm,
        sigma_gm,
        standardized_global_mean,
        loadings_gm,
        sigma_scores_gm,
        new_ex_list,
        gm,
        b_exit,
    )


#
# Performs principal components analysis  (PCA) on the p-by-n data matrix A
# rows of A correspond to (p) variables AND cols of A correspond to the (n) tests
# assume already standardized
#
# Returns the loadings: p-by-p matrix, each column containing coefficients
# for one principal component.
#
def princomp(standardized_global_mean):
    # find covariance matrix (will be pxp)
    co_mat = np.cov(standardized_global_mean)
    # Calculate evals and evecs of covariance matrix (evecs are also pxp)
    [evals, evecs] = np.linalg.eig(co_mat)
    # Above may not be sorted - sort largest first
    new_index = np.argsort(evals)[::-1]
    evecs = evecs[:, new_index]
    evals = evals[new_index]

    return evecs


#
# Calculate (val-avg)/stddev and exclude zero value
#
def calc_Z(val, avg, stddev, count, flag):
    return_val = np.empty(val.shape, dtype=np.float32, order='C')
    tol = 1e-12
    if stddev[(stddev > tol)].size == 0:
        if flag:
            print('WARNING: ALL standard dev are < 1e-12')
            flag = False
        count = count + stddev[(stddev <= tol)].size
        return_val = np.zeros(val.shape, dtype=np.float32, order='C')
    else:
        if stddev[(stddev <= tol)].size > 0:
            if flag:
                print('WARNING: some standard dev are < 1e-12')
                flag = False
            count = count + stddev[(stddev <= tol)].size
            return_val[np.where(stddev <= tol)] = 0.0
            return_val[np.where(stddev > tol)] = (
                val[np.where(stddev > tol)] - avg[np.where(stddev > tol)]
            ) / stddev[np.where(stddev > tol)]
        else:
            return_val = (val - avg) / stddev
    return count, return_val


#
# Read a json file for the excluded list of variables
# (no longer allowing include files)
def read_jsonlist(metajson, method_name):
    # method_name = ES for ensemble summary (CAM, MPAS)
    #            = ESP for POP ensemble summary

    exclude = True
    if not os.path.exists(metajson):
        print('\n')
        print(
            '*************************************************************************************'
        )
        print('Warning: Specified json file does not exist: ', metajson)
        print(
            '*************************************************************************************'
        )
        print('\n')
        varList = []
        return varList, exclude
    else:
        fd = open(metajson)
        try:
            metainfo = json.load(fd)
        except json.JSONDecodeError:
            print(f'ERROR: JSONDecode Error in file{metajson}')
            varList = ['JSONERROR']
            exclude = []
            return varList, exclude
        if method_name == 'ES':  # CAM or MPAS
            exclude = True
            if 'ExcludedVar' in metainfo:
                varList = metainfo['ExcludedVar']
            return varList, exclude
        elif method_name == 'ESP':  # POP
            var2d = metainfo['Var2d']
            var3d = metainfo['Var3d']
            return var2d, var3d


#
# Calculate Normalized RMSE metric
#
def calc_nrmse(orig_array, comp_array):
    orig_size = orig_array.size
    sumsqr = np.sum(np.square(orig_array.astype(np.float64) - comp_array.astype(np.float64)))
    rng = np.max(orig_array) - np.min(orig_array)
    if abs(rng) < 1e-18:
        rmse = 0.0
    else:
        rmse = np.sqrt(sumsqr / orig_size) / rng

    return rmse


#
# Calculate weighted global mean for one level of CAM output
# works in dp
def area_avg(data_orig, weight, is_SE):
    # TO DO: take into account missing values
    if data_orig.dtype == np.float32:
        data = data_orig.astype(np.float64)
    else:
        data = data_orig[:]

    if is_SE:
        a = np.average(data, weights=weight)
    else:  # FV
        # weights are for lat
        a_lat = np.average(data, axis=0, weights=weight)
        a = np.average(a_lat)
    return a


#
# Calculate weighted global mean for one level of OCN output
#
def pop_area_avg(data_orig, weight):
    # Take into account missing values
    # weights are for lat
    if data_orig.dtype == np.float32:
        data = data_orig.astype(np.float64)
    else:
        data = data_orig[:]

    a = np.ma.average(data, weights=weight)
    return a


#
def get_nlev(o_files, popens):
    first_file = nc.Dataset(o_files[0], 'r')
    input_dims = first_file.dimensions

    if not popens:
        nlev = len(input_dims['lev'])
    else:
        nlev = len(input_dims['z_t'])

    first_file.close()

    return nlev


#
# Calculate area_wgt when processes cam se/cam fv/pop files
#
def get_area_wgt(o_files, is_SE, nlev, popens):
    z_wgt = {}
    first_file = nc.Dataset(o_files[0], 'r')
    input_dims = first_file.dimensions

    if is_SE:
        ncol = len(input_dims['ncol'])
        output3d = np.zeros((nlev, ncol), dtype=np.float64)
        output2d = np.zeros(ncol, dtype=np.float64)
        area_wgt = np.zeros(ncol, dtype=np.float64)
        area = first_file.variables['area']
        area_wgt[:] = area[:]
        total = np.sum(area_wgt)
        area_wgt[:] /= total
    else:
        if not popens:
            nlon = len(input_dims['lon'])
            nlat = len(input_dims['lat'])
            gw = first_file.variables['gw']
        else:
            if 'nlon' in input_dims:
                nlon = len(input_dims['nlon'])
                nlat = len(input_dims['nlat'])
            elif 'lon' in input_dims:
                nlon = len(input_dims['lon'])
                nlat = len(input_dims['lat'])
            gw = first_file.variables['TAREA']
            z_wgt = first_file.variables['dz']
        output3d = np.zeros((nlev, nlat, nlon), dtype=np.float64)
        output2d = np.zeros((nlat, nlon), dtype=np.float64)
        area_wgt = np.zeros(nlat, dtype=np.float64)  # note gauss weights are length nlat
        area_wgt[:] = gw[:]

        first_file.close()

    return output3d, output2d, area_wgt, z_wgt


# ofiles are not open
def generate_global_mean_for_summary_MPAS(o_files, var_cell, var_edge, var_vertex, opts_dict):
    tslice = opts_dict['tslice']

    nCell = len(var_cell)
    nEdge = len(var_edge)
    nVertex = len(var_vertex)

    gmCell = np.zeros((nCell, len(o_files)), dtype=np.float64)
    gmEdge = np.zeros((nEdge, len(o_files)), dtype=np.float64)
    gmVertex = np.zeros((nVertex, len(o_files)), dtype=np.float64)

    # get weights for area
    first_file = nc.Dataset(o_files[0], 'r')
    input_dims = first_file.dimensions

    # cells weighted by areaCell
    nCellD = len(input_dims['nCells'])
    cell_wgt = np.zeros(nCellD, dtype=np.float64)
    cell_area = first_file.variables['areaCell']
    cell_wgt[:] = cell_area[:]

    # edges weighted by dvEdge
    nEdgeD = len(input_dims['nEdges'])
    edge_wgt = np.zeros(nEdgeD, dtype=np.float64)
    edge_area = first_file.variables['dvEdge']
    edge_wgt[:] = edge_area[:]

    # vertices weighted by areaTriangle
    nVertexD = len(input_dims['nVertices'])
    vertex_wgt = np.zeros(nVertexD, dtype=np.float64)
    vertex_area = first_file.variables['areaTriangle']
    vertex_wgt[:] = vertex_area[:]

    weights = {}
    weights['cell'] = cell_wgt
    weights['edge'] = edge_wgt
    weights['vertex'] = vertex_wgt

    # loop through the input file list to calculate global means
    # print('Examining data from files ...')
    for fcount, in_file in enumerate(o_files):
        fname = nc.Dataset(in_file, 'r')
        (
            gmCell[:, fcount],
            gmEdge[:, fcount],
            gmVertex[:, fcount],
        ) = calc_global_mean_for_onefile_MPAS(
            fname,
            weights,
            var_cell,
            var_edge,
            var_vertex,
            tslice,
        )

        fname.close()

    return gmCell, gmEdge, gmVertex


# fname is open
def calc_global_mean_for_onefile_MPAS(fname, weight_dict, var_cell, var_edge, var_vertex, tslice):
    nan_flag = False

    # how many of each variable to work on
    nCellVars = len(var_cell)
    nEdgeVars = len(var_edge)
    nVertexVars = len(var_vertex)

    gmCell = np.zeros((nCellVars), dtype=np.float64)
    gmEdge = np.zeros((nEdgeVars), dtype=np.float64)
    gmVertex = np.zeros((nVertexVars), dtype=np.float64)

    cell_wgt = weight_dict['cell']
    edge_wgt = weight_dict['edge']
    vertex_wgt = weight_dict['vertex']

    # calculate global mean for each Cell var
    # note: some vars are 2d and some 3d
    for count, vname in enumerate(var_cell):
        if isinstance(vname, str):
            vname_d = vname
        else:
            vname_d = vname.decode('utf-8')
        if vname_d not in fname.variables:
            print(
                'WARNING 1: the test file does not have the variable ',
                vname_d,
                ' that is in the ensemble summary file ...',
            )
            continue
        data = fname.variables[vname_d]
        if not data[tslice].size:
            print('ERROR: ', vname_d, ' data is empty => EXITING....')
            sys.exit(2)
        if np.any(np.isnan(data)):
            print('ERROR: ', vname_d, ' data contains NaNs - please check input => EXITING')
            nan_flag = True
            continue

        data_slice = data[tslice]
        a = np.average(data_slice, axis=0, weights=cell_wgt)
        # print("weightd = ", cell_wgt)
        # if 3d, have to average over levels (unweighted)
        if len(a.shape) > 0:
            a = np.average(a)
        gmCell[count] = a
        # print("a = ", a)

    # calculate global mean for each Edge var
    for count, vname in enumerate(var_edge):
        if isinstance(vname, str):
            vname_d = vname
        else:
            vname_d = vname.decode('utf-8')
        if vname_d not in fname.variables:
            print(
                'WARNING 1: the test file does not have the variable ',
                vname_d,
                ' that is in the ensemble summary file ...',
            )
            continue
        data = fname.variables[vname_d]
        if not data[tslice].size:
            print('ERROR: ', vname_d, ' data is empty => EXITING....')
            sys.exit(2)
        if np.any(np.isnan(data)):
            print('ERROR: ', vname_d, ' data contains NaNs - please check input => EXITING')
            nan_flag = True
            continue

        data_slice = data[tslice]
        a = np.average(data_slice, axis=0, weights=edge_wgt)
        # if 3d, have to average over levels (unweighted)
        if len(a.shape) > 0:
            a = np.average(a)
        gmEdge[count] = a

    # calculate global mean for each Vertex var
    for count, vname in enumerate(var_vertex):
        if isinstance(vname, str):
            vname_d = vname
        else:
            vname_d = vname.decode('utf-8')
        if vname_d not in fname.variables:
            print(
                'WARNING 1: the test file does not have the variable ',
                vname_d,
                ' that is in the ensemble summary file ...',
            )
            continue
        data = fname.variables[vname_d]
        if not data[tslice].size:
            print('ERROR: ', vname_d, ' data is empty => EXITING....')
            sys.exit(2)
        if np.any(np.isnan(data)):
            print('ERROR: ', vname_d, ' data contains NaNs - please check input => EXITING')
            nan_flag = True
            continue

        data_slice = data[tslice]
        a = np.average(data_slice, axis=0, weights=vertex_wgt)
        # if 3d, have to average over levels (unweighted)
        if len(a.shape) > 0:
            a = np.average(a)
        gmVertex[count] = a

    if nan_flag:
        print('ERROR: Nans in input data => EXITING....')
        sys.exit()

    return gmCell, gmEdge, gmVertex


#
# compute area_wgts, and then loop through all files to call calc_global_means_for_onefile
# o_files are not open for CAM
# 12/19 - summary file will now be double precision
def generate_global_mean_for_summary(o_files, var_name3d, var_name2d, is_SE, opts_dict):
    tslice = opts_dict['tslice']
    popens = opts_dict['popens']

    n3d = len(var_name3d)
    n2d = len(var_name2d)

    gm3d = np.zeros((n3d, len(o_files)), dtype=np.float64)
    gm2d = np.zeros((n2d, len(o_files)), dtype=np.float64)

    nlev = get_nlev(o_files, popens)

    output3d, output2d, area_wgt, z_wgt = get_area_wgt(o_files, is_SE, nlev, popens)

    # loop through the input file list to calculate global means
    for fcount, in_file in enumerate(o_files):
        fname = nc.Dataset(in_file, 'r')

        if popens:
            gm3d[:, fcount], gm2d[:, fcount] = calc_global_mean_for_onefile_pop(
                fname,
                area_wgt,
                z_wgt,
                var_name3d,
                var_name2d,
                output3d,
                output2d,
                tslice,
                is_SE,
                nlev,
                opts_dict,
            )

        else:  # CAM
            gm3d[:, fcount], gm2d[:, fcount] = calc_global_mean_for_onefile(
                fname,
                area_wgt,
                var_name3d,
                var_name2d,
                output3d,
                output2d,
                tslice,
                is_SE,
                nlev,
                opts_dict,
            )

        fname.close()

    return gm3d, gm2d


# Calculate global means for one OCN input file
# (fname is open) NO LONGER USING GLOBAL MEANS for POP
def calc_global_mean_for_onefile_pop(
    fname,
    area_wgt,
    z_wgt,
    var_name3d,
    var_name2d,
    output3d,
    output2d,
    tslice,
    is_SE,
    nlev,
    opts_dict,
):
    nan_flag = False

    n3d = len(var_name3d)
    n2d = len(var_name2d)

    gm3d = np.zeros((n3d), dtype=np.float64)
    gm2d = np.zeros((n2d), dtype=np.float64)

    # calculate global mean for each 3D variable
    for count, vname in enumerate(var_name3d):
        gm_lev = np.zeros(nlev, dtype=np.float64)
        data = fname.variables[vname]
        if np.any(np.isnan(data)):
            print('ERROR: ', vname, ' data contains NaNs - please check input.')
            nan_flag = True
        output3d[:, :, :] = data[tslice, :, :, :]
        dbl_output3d = output3d.astype(dtype=np.float64)
        for k in range(nlev):
            moutput3d = np.ma.masked_values(dbl_output3d[k, :, :], data._FillValue)
            gm_lev[k] = pop_area_avg(moutput3d, area_wgt)
        # note: averaging over levels - in future, consider pressure-weighted (?)
        gm3d[count] = np.average(gm_lev, weights=z_wgt)

    # calculate global mean for each 2D variable
    for count, vname in enumerate(var_name2d):
        data = fname.variables[vname]
        if np.any(np.isnan(data)):
            print('ERROR: ', vname, ' data contains NaNs - please check input.')
            nan_flag = True
        output2d[:, :] = data[tslice, :, :]
        dbl_output2d = output2d.astype(dtype=np.float64)
        moutput2d = np.ma.masked_values(dbl_output2d[:, :], data._FillValue)
        gm2d_mean = pop_area_avg(moutput2d, area_wgt)
        gm2d[count] = gm2d_mean

    if nan_flag:
        print('ERROR: Nans in input data => EXITING....')
        sys.exit()

    return gm3d, gm2d


#
# Calculate global means for one CAM input file
# fname is open
def calc_global_mean_for_onefile(
    fname, area_wgt, var_name3d, var_name2d, output3d, output2d, tslice, is_SE, nlev, opts_dict
):
    nan_flag = False

    if 'cumul' in opts_dict:
        cumul = opts_dict['cumul']
    else:
        cumul = False
    n3d = len(var_name3d)
    n2d = len(var_name2d)

    gm3d = np.zeros((n3d), dtype=np.float64)
    gm2d = np.zeros((n2d), dtype=np.float64)

    # calculate global mean for each 3D variable (note: area_avg casts into dp before computation)
    for count, vname in enumerate(var_name3d):
        if isinstance(vname, str):
            vname_d = vname
        else:
            vname_d = vname.decode('utf-8')

        if vname_d not in fname.variables:
            print(
                'WARNING 1: the test file does not have the variable ',
                vname_d,
                ' that is in the ensemble summary file ...',
            )
            continue
        data = fname.variables[vname_d]
        if not data[tslice].size:
            print('ERROR: ', vname_d, ' data is empty => EXITING....')
            sys.exit(2)
        if np.any(np.isnan(data)):
            print('ERROR: ', vname_d, ' data contains NaNs - please check input => EXITING')
            nan_flag = True
            continue
        if is_SE:
            if not cumul:
                temp = data[tslice].shape[0]
                gm_lev = np.zeros(temp, dtype=np.float64)
                for k in range(temp):
                    gm_lev[k] = area_avg(data[tslice, k, :], area_wgt, is_SE)
            else:
                gm_lev = np.zeros(nlev, dtype=np.float64)
                for k in range(nlev):
                    gm_lev[k] = area_avg(output3d[k, :], area_wgt, is_SE)
        else:
            if not cumul:
                temp = data[tslice].shape[0]
                gm_lev = np.zeros(temp, dtype=np.float64)
                for k in range(temp):
                    gm_lev[k] = area_avg(data[tslice, k, :, :], area_wgt, is_SE)
            else:
                gm_lev = np.zeros(nlev)
                for k in range(nlev):
                    gm_lev[k] = area_avg(output3d[k, :, :], area_wgt, is_SE)
        # note: averaging over levels could be pressure-weighted (?)
        gm3d[count] = np.mean(gm_lev)

    # calculate global mean for each 2D variable
    for count, vname in enumerate(var_name2d):
        if isinstance(vname, str):
            vname_d = vname
        else:
            vname_d = vname.decode('utf-8')

        if vname_d not in fname.variables:
            print(
                'WARNING 2: the test file does not have the variable ',
                vname_d,
                ' that is in the ensemble summary file',
            )
            continue
        data = fname.variables[vname_d]
        if np.any(np.isnan(data)):
            print('ERROR: ', vname_d, ' data contains NaNs - please check input => EXITING....')
            nan_flag = True
            continue
        if is_SE:
            if not cumul:
                output2d[:] = data[tslice, :]
            gm2d_mean = area_avg(output2d[:], area_wgt, is_SE)
        else:
            if not cumul:
                output2d[:, :] = data[tslice, :, :]
            gm2d_mean = area_avg(output2d[:, :], area_wgt, is_SE)
        gm2d[count] = gm2d_mean

    if nan_flag:
        print('ERROR: Nans in input data => EXITING....')
        sys.exit()

    return gm3d, gm2d


# Read variable values from ensemble summary file
#
def read_ensemble_summary(ens_file):
    if os.path.isfile(ens_file):
        fens = nc.Dataset(ens_file, 'r')
    else:
        print('ERROR: file ens summary: ', ens_file, ' not found => EXITING....')
        sys.exit(2)

    is_SE = False
    dims = fens.dimensions
    if 'ncol' in dims:
        is_SE = True

    esize = len(dims['ens_size'])
    str_size = len(dims['str_size'])

    ens_avg = {}
    ens_stddev = {}
    ens_var_name = []
    ens_rmsz = {}
    ens_gm = {}
    std_gm = {}

    # Retrieve the variable list from ensemble file
    for k, v in fens.variables.items():
        if k == 'vars':
            for i in v[0 : len(v)]:
                lcount = 0
                for j in i:
                    if j:
                        lcount = lcount + 1
                tempn = i[0:lcount].tostring().strip()
                tempn = tempn.decode('UTF-8')
                ens_var_name.append(tempn)
        elif k == 'var3d':
            num_var3d = len(v)
        elif k == 'var2d':
            num_var2d = len(v)

    for k, v in fens.variables.items():
        # Retrieve the ens_avg3d or ens_avg2d array
        if k == 'ens_avg3d' or k == 'ens_avg2d':
            if k == 'ens_avg2d':
                m = num_var3d
            else:
                m = 0
            if v:
                for i in v[0 : len(v)]:
                    temp_name = ens_var_name[m]
                    ens_avg[temp_name] = i
                    m = m + 1

        # Retrieve the ens_stddev3d or ens_stddev2d  array
        elif k == 'ens_stddev3d' or k == 'ens_stddev2d':
            if k == 'ens_stddev2d':
                m = num_var3d
            else:
                m = 0
            if v:
                for i in v[0 : len(v)]:
                    temp_name = ens_var_name[m]
                    ens_stddev[temp_name] = i
                    m = m + 1
        # Retrieve the RMSZ score array
        elif k == 'RMSZ':
            m = 0
            for i in v[0 : len(v)]:
                temp_name = ens_var_name[m]
                ens_rmsz[temp_name] = i
                m = m + 1
        elif k == 'global_mean':
            m = 0
            for i in v[0 : len(v)]:
                temp_name = ens_var_name[m]
                ens_gm[temp_name] = i
                m = m + 1
        elif k == 'standardized_gm':
            m = 0
            for i in v[0 : len(v)]:
                temp_name = ens_var_name[m]
                std_gm[temp_name] = i
                m = m + 1
            # also get as array (not just dictionary)
            std_gm_array = np.zeros((num_var3d + num_var2d, esize), dtype=np.float64)
            std_gm_array[:] = v[:, :]
        elif k == 'mu_gm':
            mu_gm = np.zeros((num_var3d + num_var2d), dtype=np.float64)
            mu_gm[:] = v[:]
        elif k == 'sigma_gm':
            sigma_gm = np.zeros((num_var3d + num_var2d), dtype=np.float64)
            sigma_gm[:] = v[:]
        elif k == 'loadings_gm':
            loadings_gm = np.zeros((num_var3d + num_var2d, num_var3d + num_var2d), dtype=np.float64)
            loadings_gm[:, :] = v[:, :]
        elif k == 'sigma_scores_gm':
            sigma_scores_gm = np.zeros((num_var3d + num_var2d), dtype=np.float64)
            sigma_scores_gm[:] = v[:]

    fens.close()

    return (
        ens_var_name,
        ens_avg,
        ens_stddev,
        ens_rmsz,
        ens_gm,
        num_var3d,
        mu_gm,
        sigma_gm,
        loadings_gm,
        sigma_scores_gm,
        is_SE,
        std_gm,
        std_gm_array,
        str_size,
    )


# MPAS: Read variable values from ensemble summary file
#
def mpas_read_ensemble_summary(ens_file):
    if os.path.isfile(ens_file):
        fens = nc.Dataset(ens_file, 'r')
    else:
        print('ERROR: file mpas ens summary: ', ens_file, ' not found => EXITING....')
        sys.exit(2)

    dims = fens.dimensions

    esize = len(dims['ens_size'])
    str_size = len(dims['str_size'])

    ens_var_name = []
    std_gm = {}
    ens_gm = {}

    # Retrieve the variable lists from ensemble file
    for k, v in fens.variables.items():
        if k == 'vars':
            for i in v[0 : len(v)]:
                lcount = 0
                for j in i:
                    if j:
                        lcount = lcount + 1
                tempn = i[0:lcount].tostring().strip()
                tempn = tempn.decode('UTF-8')
                ens_var_name.append(tempn)
        elif k == 'varCell':
            num_varCell = len(v)
        elif k == 'varEdge':
            num_varEdge = len(v)
        elif k == 'varVertex':
            num_varVertex = len(v)

    num_var = num_varCell + num_varEdge + num_varVertex

    for k, v in fens.variables.items():
        if k == 'global_mean':
            m = 0
            for i in v[0 : len(v)]:
                temp_name = ens_var_name[m]
                ens_gm[temp_name] = i
                m = m + 1
        elif k == 'standardized_gm':
            m = 0
            for i in v[0 : len(v)]:
                temp_name = ens_var_name[m]
                std_gm[temp_name] = i
                m = m + 1
            # also get as array (not just dictionary)
            std_gm_array = np.zeros((num_var, esize), dtype=np.float64)
            std_gm_array[:] = v[:, :]
        elif k == 'mu_gm':
            mu_gm = np.zeros(num_var, dtype=np.float64)
            mu_gm[:] = v[:]
        elif k == 'sigma_gm':
            sigma_gm = np.zeros(num_var, dtype=np.float64)
            sigma_gm[:] = v[:]
        elif k == 'loadings_gm':
            loadings_gm = np.zeros((num_var, num_var), dtype=np.float64)
            loadings_gm[:, :] = v[:, :]
        elif k == 'sigma_scores_gm':
            sigma_scores_gm = np.zeros((num_var), dtype=np.float64)
            sigma_scores_gm[:] = v[:]

    fens.close()

    return (
        ens_var_name,
        num_varCell,
        num_varEdge,
        num_varVertex,
        mu_gm,
        sigma_gm,
        loadings_gm,
        sigma_scores_gm,
        std_gm,
        std_gm_array,
        str_size,
        ens_gm,
    )


#
# Get the ncol and nlev value from cam run file
# (frun is not open)
def get_ncol_nlev(frun):
    o_frun = nc.Dataset(frun, 'r')
    input_dims = o_frun.dimensions
    ncol = -1
    nlev = -1

    nlat = -1
    nlon = -1

    for k, v in input_dims.items():
        if k == 'lev':
            nlev = len(v)
        if k == 'ncol':
            ncol = len(v)
        if (k == 'lat') or (k == 'nlat'):
            nlat = len(v)
        if (k == 'lon') or (k == 'nlon'):
            nlon = len(v)

    if ncol == -1:
        one_spatial_dim = False
    else:
        one_spatial_dim = True

    if one_spatial_dim:
        npts3d = float(nlev * ncol)
        npts2d = float(ncol)
    else:
        npts3d = float(nlev * nlat * nlon)
        npts2d = float(nlat * nlon)

    o_frun.close()

    return npts3d, npts2d, one_spatial_dim


#
# Calculate max norm ensemble value for each variable base on all ensemble run files
# the inputdir should only have all ensemble run files
#
def calculate_maxnormens(opts_dict, var_list):
    ifiles = []
    Maxnormens = {}
    threshold = 1e-12
    # input file directory
    inputdir = opts_dict['indir']

    # the timeslice that we want to process
    tstart = opts_dict['tslice']

    # open all files
    for frun_file in os.listdir(inputdir):
        if os.path.isfile(inputdir + frun_file):
            ifiles.append(nc.Dataset(inputdir + frun_file, 'r'))
        else:
            print('ERROR: Could not locate file= ' + inputdir + frun_file + ' => EXITING....')
            sys.exit()
    comparision = {}
    # loop through each variable
    for k in var_list:
        output = []
        # read all data of variable k from all files
        for f in ifiles:
            v = f.variables
            output.append(v[k][tstart])
        max_val = 0
        # open an output file
        outmaxnormens = k + '_ens_maxnorm.txt'
        fout = open(outmaxnormens, 'w')
        Maxnormens[k] = []

        # calculate E(i=0:n)(maxnormens[i][x])=max(comparision[i]-E(x=0:n)(output[x]))
        for n in range(len(ifiles)):
            Maxnormens[k].append(0)
            comparision[k] = ifiles[n].variables[k][tstart]
            for m in range(len(ifiles)):
                max_val = np.max(np.abs(comparision[k] - output[m]))
                if Maxnormens[k][n] < max_val:
                    Maxnormens[k][n] = max_val
            range_max = np.max((comparision[k]))
            range_min = np.min((comparision[k]))
            if range_max - range_min < threshold:
                Maxnormens[k][n] = 0.0
            else:
                Maxnormens[k][n] = Maxnormens[k][n] / (range_max - range_min)
            fout.write(str(Maxnormens[k][n]) + '\n')
        strtmp = (
            k
            + ' : '
            + 'ensmax min max'
            + ' : '
            + '{0:9.2e}'.format(min(Maxnormens[k]))
            + ' '
            + '{0:9.2e}'.format(max(Maxnormens[k]))
        )
        print(strtmp)
        fout.close()


#
# Parse options from command line or from config file
#
def getopt_parseconfig(opts, optkeys, caller, opts_dict):
    # integer
    integer = '-[0-9]+'
    int_p = re.compile(integer)
    # scientific notation
    flt = r'-*[0-9]+\.[0-9]+'
    flt_p = re.compile(flt)

    for opt, arg in opts:
        if opt == '-h' and caller == 'CECT':
            CECT_usage()
            sys.exit()
        elif opt == '-h' and caller == 'ES':
            EnsSum_usage()
            sys.exit()
        elif opt == '-h' and caller == 'ESP':
            EnsSumPop_usage()
            sys.exit()
        elif opt == '-h' and caller == 'ES_MPAS':
            EnsSumMPAS_usage()
            sys.exit()
        elif opt == '-f':
            opts_dict['orig'] = arg
        elif opt == '-m':
            opts_dict['reqmeth'] = arg
        # parse config file
        elif opt in ('--config'):
            configfile = arg
            config = configparser.ConfigParser()
            config.read(configfile)
            for sec in config.sections():
                for name, value in config.items(sec):
                    if sec == 'bool_arg' or sec == 'metrics':
                        opts_dict[name] = config.getboolean(sec, name)
                    elif sec == 'int_arg':
                        opts_dict[name] = config.getint(sec, name)
                    elif sec == 'float_arg':
                        opts_dict[name] = config.getfloat(sec, name)
                    else:
                        opts_dict[name] = value

        # parse command line options which might replace the settings in the config file
        else:
            for k in optkeys:
                if k.find('=') != -1:
                    keyword = k[0 : k.find('=')]
                    if opt == '--' + keyword:
                        if arg.isdigit():
                            opts_dict[keyword] = int(arg)
                        else:
                            if flt_p.match(arg):
                                opts_dict[keyword] = float(arg)
                            elif int_p.match(arg):
                                opts_dict[keyword] = int(arg)
                            else:
                                opts_dict[keyword] = arg
                else:
                    if opt == '--' + k:
                        opts_dict[k] = True
    return opts_dict


#
# Figure out the scores of the 3 new runs, standardized global means, then multiple by the loadings_gm
#
def standardized(gm, mu_gm, sigma_gm, loadings_gm, all_var_names, opts_dict, me):
    nvar = gm.shape[0]
    nfile = gm.shape[1]
    sum_std_mean = np.zeros((nvar,), dtype=np.float64)
    standardized_mean = np.zeros(gm.shape, dtype=np.float64)
    for var in range(nvar):
        for file in range(nfile):
            standardized_mean[var, file] = (
                gm[var, file].astype(np.float64) - mu_gm[var].astype(np.float64)
            ) / sigma_gm[var].astype(np.float64)
            sum_std_mean[var] = sum_std_mean[var] + np.abs(standardized_mean[var, file])
    new_scores = np.dot(loadings_gm.T.astype(np.float64), standardized_mean)

    return new_scores, sum_std_mean, standardized_mean


#
# Insert rmsz scores, global mean of new run to the dictionary results
#
def addresults(results, key, value, var, thefile):
    if var in results:
        temp = results[var]
        if key in temp:
            temp2 = temp[key]
            if thefile in temp2:
                temp3 = results[var][key][thefile]
            else:
                temp3 = {}
        else:
            temp[key] = {}
            temp2 = {}
            temp3 = {}
        temp3 = value
        temp2[thefile] = temp3
        temp[key] = temp2
        results[var] = temp
    else:
        results[var] = {}
        results[var][key] = {}
        results[var][key][thefile] = value

    return results


#
# Print out rmsz score failure, global mean failure summary
#
def printsummary(results, key, name, namerange, thefilecount, variables, label):
    thefile = 'f' + str(thefilecount)
    for k, v in results.items():
        if 'status' in v:
            temp0 = v['status']
            # strname = k.decode('utf-8')
            strname = k
            if key in temp0:
                if thefile in temp0[key]:
                    temp = temp0[key][thefile]
                    if temp < 1:
                        print(' ')
                        print(
                            f' {strname}: {v[name][thefile]:2e} outside of [{variables[k][namerange][0]:2e}, {variables[k][namerange][1]:2e}]'
                        )


#
# Insert the range of  global mean of the ensemble summary file to the dictionary variables
#
def addvariables(variables, var, vrange, thearray):
    if var in variables:
        variables[var][vrange] = (np.min(thearray), np.max(thearray))
    else:
        variables[var] = {}
        variables[var][vrange] = (np.min(thearray), np.max(thearray))

    return variables


#
# Evaluate if the new run global mean in the range of global mean of the ensemble summary
#
def evaluatestatus(name, rangename, variables, key, results, thefile):
    # print("thefile = ", thefile)
    # print("vars = ", variables)
    totalcount = 0
    for k, v in results.items():
        if name in v and rangename in variables[k]:
            temp0 = results[k]
            xrange = variables[k][rangename]
            if v[name][thefile] > xrange[1] or v[name][thefile] < xrange[0]:
                val = 0
            else:
                val = 1
            if 'status' in temp0:
                temp = temp0['status']
                if key in temp:
                    temp2 = temp[key]
                else:
                    temp[key] = temp2 = {}

                if val == 0:
                    totalcount = totalcount + 1
                temp2[thefile] = val
                temp[key] = temp2
                results[k]['status'] == temp
            else:
                temp0['status'] = {}
                temp0['status'][key] = {}
                temp0['status'][key][thefile] = val
                if val == 0:
                    totalcount = totalcount + 1

    return totalcount


#
# Evaluate if the new run PCA scores pass or fail by comparing with the PCA scores of the ensemble summary
# ifiles are open
def comparePCAscores(ifiles, new_scores, sigma_scores_gm, opts_dict, me):
    comp_array = np.zeros(new_scores.shape, dtype=np.int32)
    sum = np.zeros(new_scores.shape[0], dtype=np.int32)
    eachruncount = np.zeros(new_scores.shape[1], dtype=np.int32)
    totalcount = 0
    sum_index = []

    if me.get_rank() == 0:
        print('')
        print('*********************************************** ')
        print('PCA Test Results')
        print('*********************************************** ')

    # Test to check if new_scores out of range of sigMul*sigma_scores_gm
    for i in range(opts_dict['nPC']):
        for j in range(new_scores.shape[1]):
            if abs(new_scores[i][j]) > opts_dict['sigMul'] * (sigma_scores_gm[i]):
                comp_array[i][j] = 1
                eachruncount[j] = eachruncount[j] + 1
            # Only check the first nPC number of scores, and sum comp_array together
            sum[i] = sum[i] + comp_array[i][j]

    if len(ifiles) >= opts_dict['minRunFail']:
        num_run_less = False
    else:
        num_run_less = True
    # Check to see if sum is larger than min_run_fail, if so save the index of the sum
    for i in range(opts_dict['nPC']):
        if sum[i] >= opts_dict['minRunFail']:
            totalcount = totalcount + 1
            sum_index.append(i + 1)

    # save comp_array if filepath is provided
    # if me.get_rank() == 0:
    #    if len(opts_dict['savePCAMat']) > 0:
    #        np.save(opts_dict['savePCAMat'], comp_array)

    # false_positive=check_falsepositive(opts_dict,sum_index)

    # If the length of sum_index is larger than min_PC_fail, the three runs failed.
    # This doesn't apply for UF-ECT.
    if opts_dict['numRunFile'] > opts_dict['eet']:
        if len(sum_index) >= opts_dict['minPCFail']:
            decision = 'FAILED'
        else:
            decision = 'PASSED'
        if (num_run_less is False) and (me.get_rank() == 0):
            print(' ')
            print(
                'Summary: '
                + str(totalcount)
                + ' PC scores failed at least '
                + str(opts_dict['minRunFail'])
                + ' runs: ',
                sum_index,
            )
            print(' ')
            print('These runs ****' + decision + '**** according to our testing criterion.')
            print(' ')

        elif me.get_rank() == 0:
            print(' ')
            print(
                'The number of run files is less than minRunFail (=2), so we cannot determin an overall pass or fail.'
            )
            print(' ')

    # Record the histogram of comp_array which value is one by the PCA scores
    for i in range(opts_dict['nPC']):
        index_list = []
        for j in range(comp_array.shape[1]):
            if comp_array[i][j] == 1:
                index_list.append(j + 1)
        if len(index_list) > 0 and me.get_rank() == 0:
            print('PC ' + str(i + 1) + ': failed ' + str(len(index_list)) + ' runs ', index_list)
    if me.get_rank() == 0:
        print(' ')

    # Record the index of comp_array which value is one
    run_index = []

    if opts_dict['eet'] >= opts_dict['numRunFile']:
        eet = exhaustive_test()
        faildict = {}

        for j in range(comp_array.shape[1]):
            index_list = []
            for i in range(opts_dict['nPC']):
                if comp_array[i][j] == 1:
                    index_list.append(i + 1)
            if me.get_rank() == 0:
                print(
                    'Run ' + str(j + 1) + ': ' + str(eachruncount[j]) + ' PC scores failed ',
                    index_list,
                )
            run_index.append((j + 1))
            faildict[str(j + 1)] = set(index_list)

        passes, failures = eet.test_combinations(
            faildict, runsPerTest=opts_dict['numRunFile'], nRunFails=opts_dict['minRunFail']
        )
        if me.get_rank() == 0:
            print(' ')
            print('%d tests failed out of %d possible tests.' % (failures, passes + failures))
            print(
                'This represents a failure percent of %.2f.'
                % (100.0 * failures / float(failures + passes))
            )
            print(' ')
            if float(failures) > 0.1 * float(passes + failures):
                decision = 'FAILED'
            else:
                decision = 'PASSED'

            # save eet if filepath is provided
            if len(opts_dict['saveEET']) > 0:
                np.save(opts_dict['saveEET'], np.array([passes, passes + failures]))

    else:
        for j in range(comp_array.shape[1]):
            index_list = []
            for i in range(opts_dict['nPC']):
                if comp_array[i][j] == 1:
                    index_list.append(i + 1)
            if me.get_rank() == 0:
                print(
                    'Run ' + str(j + 1) + ': ' + str(eachruncount[j]) + ' PC scores failed ',
                    index_list,
                )
            run_index.append((j + 1))

    return run_index, decision


#
# Command options for pyCECT.py
#
def CECT_usage():
    print('\n Compare test runs to an ensemble summary file. \n')
    print('  ----------------------------')
    print('   Args for pyCECT (all models):')
    print('  ----------------------------')
    print('   pyCECT.py')
    print('   -h                      : prints out this usage message')
    print('   --verbose               : prints out in verbose mode (off by default)')
    print('   --sumfile  <ifile>      : the ensemble summary file (generated by pyEnsSum.py)')
    print(
        '   --indir    <path>       : directory containing the input run files (at least 3 files)'
    )
    print('   --tslice   <num>        : which time slice to use from input run files (default = 1)')
    print('   NOTE: Runs for CAM by default (see below to specify POP or MPAS instead)')
    print('  ----------------------------')
    print('   Args relevant to CAM-CECT/UF-CAM-ECT and MPAS-ECT only:')
    print('  ----------------------------')
    print(
        '   --nPC <num>             : number of principal components (PCs) to check (can\'t be greater than the number of variables)'
    )
    print(
        '   --sigMul   <num>        : number of standard deviations away from the mean defining the "acceptance region"'
    )
    print(
        '   --minPCFail <num>       : minimum number of PCs that must fail the specified number of runs for a FAILURE (default = 3)'
    )
    print(
        '   --minRunFail <num>      : minimum number of runs that <minPCfail> PCs must fail for a FAILURE (default = 2)'
    )
    print('   --numRunFile <num>      : total number of runs to include in test (default = 3)')
    print(
        '   --printStdMean          : print out variables that fall outside of the global mean ensemble distribution (off by default for a pass)'
    )
    print(
        '   --saveResults           : save a netcdf file with scores and std global means from the test runs (savefile.nc). '
    )
    print(
        '   --eet <num>             : enable Ensemble Exhaustive Test (EET) to compute failure percent of <num> runs (greater than or equal to numRunFile)'
    )

    print('  ----------------------------')
    print('   Args relevant to MPAS-CECT only:')
    print('  ----------------------------')
    print('   --mpas                  : indicate MPAS-ECT (required!)')

    print('  ----------------------------')
    print('   Args relevant to POP-CECT only :')
    print('  ----------------------------')
    print('   --popens or --pop       : indicate POP-ECT (required!)')
    print(
        '   --jsonfile  <file>      : list the json file that specifies variables to test (required!), e.g. pop_ensemble.json'
    )
    print('   --pop_tol <num>         : set pop zscore tolerance (default is 3.0 - recommended)')
    print('   --pop_threshold <num>   : set pop threshold (default is 0.9)')
    print(
        '   --input_globs <search pattern> : set the search pattern (wildcard) for the file(s) to compare from '
    )
    print(
        '                           the input directory (indir), such as core48.pop.h.0003-12 or core48.pop.h.0003 (more info in README)'
    )

    print(
        '   --base_year <num>       :We assume the pop test files names start in year 0001. Use this option to specify a different start year.'
    )


#    print 'Version 3.0.8'


#
# Command options for pyEnsSum.py
#
def EnsSum_usage():
    print('\n Creates the summary file for an ensemble of CAM data. \n')
    print('  ------------------------')
    print('   Args for pyEnsSum : ')
    print('  ------------------------')
    print('   pyEnsSum.py')
    print('   -h                   : prints out this usage message')
    print('   --verbose            : prints out in verbose mode (off by default)')
    print('   --sumfile <ofile>    : the output summary data file (default = ens.summary.nc)')
    print('   --indir <path>       : directory containing all of the ensemble runs (default = ./)')
    print('   --esize  <num>       : Number of ensemble members (default = 1800)')
    print('   --tag <name>         : Tag name used in metadata (default = cesm_version)')
    print('   --compset <name>     : Compset used in metadata (default = compset)')
    print('   --res <name>         : Resolution used in metadata (default = res)')
    print('   --mach <name>        : Machine name used in the metadata (default = derecho)')
    print('   --tslice <num>       : the index into the time dimension (default = 0)')
    print('   --jsonfile <fname>   : Jsonfile to provide that a list of variables that will ')
    print('                          be excluded (default = exclude_empty.json)')
    print('   --mpi_disable        : Disable mpi mode to run in serial (off by default)')
    print('   ')


def EnsSumMPAS_usage():
    print('\n Creates the summary file for an ensemble of MPAS data. \n')
    print('  ------------------------')
    print('   Args for pyEnsSumMPAS : ')
    print('  ------------------------')
    print('   pyEnsSumMPAS.py')
    print('   -h                   : prints out this usage message')
    print('   --verbose            : prints out in verbose mode (off by default)')
    print('   --sumfile <ofile>    : the output summary data file (default = mpas.ens.summary.nc)')
    print('   --indir <path>       : directory containing all of the ensemble runs (default = ./)')
    print('   --esize  <num>       : Number of ensemble members (default = 200)')
    print('   --tag <name>         : Tag name for the summary metadata (default = tag)')
    print('   --core <name>        : Core name for the summary metadata (default = atmosphere)')
    print('   --mesh <name>        : Mesh name for the summary metadata (default = mesh)')
    print('   --model <name>       : Model name for the summary metadata (default = mpas)')
    print('   --mach <name>        : Machine name used in the metadata (default = derecho)')
    print('   --tslice <num>       : the index into the time dimension (default = 0)')
    print('   --jsonfile <fname>   : Jsonfile to provide that a list of variables that will ')
    print('                          be excluded  (default = empty_excluded.json)')
    print('   --mpi_disable        : Disable mpi mode to run in serial (mpi is enabled by default)')
    print('   ')


#
# Command options for pyEnsSumPop.py
#
def EnsSumPop_usage():
    print('\n Creates the summary file for an ensemble of POP data. \n')
    print('  ------------------------')
    print('   Args for pyEnsSumPop : ')
    print('  ------------------------')
    print('   pyEnsSumPop.py')
    print('   -h                   : prints out this usage message')
    print('   --verbose            : prints out in verbose mode (off by default)')
    print('   --sumfile    <ofile> : the output summary data file (default = pop.ens.summary.nc)')
    print('   --indir      <path>  : directory containing all of the ensemble runs (default = ./)')
    print('   --esize <num>        : Number of ensemble members (default = 40)')
    print('                          (Note: backwards compatible with --npert)')
    print('   --tag <name>         : Tag name used in metadata (default = tag)')
    print('   --compset <name>     : Compset used in metadata (default = G)')
    print('   --res <name>         : Resolution (used in metadata) (default = T62_g17)')
    print('   --mach <name>        : Machine name used in the metadata (default = derecho)')
    print('   --tslice <num>       : the time slice of the variable that we will use (default = 0)')
    print('   --nyear  <num>       : Number of years (default = 1)')
    print('   --nmonth  <num>      : Number of months (default = 12)')
    print('   --jsonfile <fname>   : Jsonfile to provide that a list of variables that will be')
    print('                          included  (RECOMMENDED: default = pop_ensemble.json)')
    print('   --mpi_disable        : Disable mpi mode to run in serial (off by default)')
    print('   ')


#
# Random pick up three files out of a lot files
#
def Random_pickup(ifiles, opts_dict):
    if opts_dict['numRunFile'] > opts_dict['eet']:
        nFiles = opts_dict['numRunFile']
    else:
        nFiles = opts_dict['eet']
    if len(ifiles) > nFiles:
        random_index = random.sample(list(range(0, len(ifiles))), nFiles)
        print('Randomly selected input files:')
    else:
        random_index = list(range(len(ifiles)))
        print('Input files:')
    new_ifiles = []

    for i in random_index:
        new_ifiles.append(ifiles[i])
        print(ifiles[i])

    return new_ifiles


#
# Random pick up opts_dict['npick'] files out of a lot of OCN files
#
def Random_pickup_pop(indir, opts_dict, npick):
    # random_year_range = opts_dict['nyear']
    # random_month_range = opts_dict['nmonth']
    random_case_range = opts_dict['esize']

    pyear = 1
    pmonth = 12

    pcase = random.sample(list(range(0, random_case_range)), npick)

    new_ifiles_temp = []
    not_pick_files = []
    for i in pcase:
        wildname = (
            '*' + str(i).zfill(4) + '*' + str(pyear).zfill(4) + '-' + str(pmonth).zfill(2) + '*'
        )
        print(wildname)
        for filename in os.listdir(opts_dict['indir']):
            if fnmatch.fnmatch(filename, wildname):
                new_ifiles_temp.append(filename)
    for filename in os.listdir(opts_dict['indir']):
        if filename not in new_ifiles_temp:
            not_pick_files.append(filename)
    with open(
        opts_dict['jsondir']
        + 'random_testcase.'
        + str(npick)
        + '.'
        + str(opts_dict['seq'])
        + '.json',
        'wb',
    ) as fout:
        json.dump(
            {'not_pick_files': not_pick_files}, fout, sort_keys=True, indent=4, ensure_ascii=True
        )
    print(sorted(new_ifiles_temp))
    print(sorted(not_pick_files))
    return sorted(new_ifiles_temp)


#
# Check the false positive rate
# (needs updating: this is only for esize 151)
def check_falsepositive(opts_dict, sum_index):
    fp = np.zeros((opts_dict['nPC'],), dtype=np.float32)
    fp[0] = 0.30305
    fp[1] = 0.05069
    fp[2] = 0.005745
    fp[3] = 0.000435
    fp[4] = 5.0e-05
    nPC = 50
    sigMul = 2
    minPCFail = 3
    minRunFail = 2
    numRunFile = 3

    if opts_dict['numRunFile'] > opts_dict['eet']:
        nFiles = opts_dict['numRunFile']
    else:
        nFiles = opts_dict['eet']

    if (
        (nPC == opts_dict['nPC'])
        and (sigMul == opts_dict['sigMul'])
        and (minPCFail == opts_dict['minPCFail'])
        and (minRunFail == opts_dict['minRunFail'])
        and (numRunFile == nFiles)
    ):
        false_positive = fp[len(sum_index) - 1]
    else:
        false_positive = 1.0

    return false_positive


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


#
# Gather arrays from each processor by the file_list to the master processor and make it an array
#
def gather_npArray_pop(npArray, me, array_shape):
    the_array = np.zeros(array_shape, dtype=np.float32)

    if me.get_rank() == 0:
        j = me.get_rank()
        if len(array_shape) == 1:
            the_array[j] = npArray[0]
        elif len(array_shape) == 2:
            the_array[j, :] = npArray[:]
        elif len(array_shape) == 3:
            the_array[j, :, :] = npArray[:, :]
        elif len(array_shape) == 4:
            the_array[j, :, :, :] = npArray[:, :, :]
        elif len(array_shape) == 5:
            the_array[j, :, :, :, :] = npArray[:, :, :, :]
    for i in range(1, me.get_size()):
        if me.get_rank() == 0:
            rank, npArray = me.collect()
            if len(array_shape) == 1:
                the_array[rank] = npArray[0]
            elif len(array_shape) == 2:
                the_array[rank, :] = npArray[:]
            elif len(array_shape) == 3:
                the_array[rank, :, :] = npArray[:, :]
            elif len(array_shape) == 4:
                the_array[rank, :, :, :] = npArray[:, :, :]
            elif len(array_shape) == 5:
                the_array[rank, :, :, :, :] = npArray[:, :, :, :]
    if me.get_rank() != 0:
        me.collect(npArray)
    me.sync()
    return the_array


#
# Use input files from opts_dict['input_globs'] to get timeslices for pop ensemble
# Test file years must start with 0001 (so we can figure out which year to compare)
#
def get_files_from_glob(opts_dict):
    base_year = opts_dict['base_year']
    if base_year > 1:
        print('base year = ', base_year)
    in_files = []
    wildname = '*' + str(opts_dict['input_globs']) + '*'
    if os.path.exists(opts_dict['indir']):
        full_glob_str = os.path.join(opts_dict['indir'], wildname)
        glob_files = glob.glob(full_glob_str)
        in_files.extend(glob_files)
        in_files.sort()
    else:
        print('ERROR: Input directory does not exist => EXITING....')
        sys.exit()
    n_timeslice = []
    for fname in in_files:
        istr = fname.find('.nc')
        temp = (int(fname[istr - 7 : istr - 3]) - base_year) * 12 + int(fname[istr - 2 : istr]) - 1
        n_timeslice.append(temp)
    return n_timeslice, in_files


#
# POP-ECT Compare the testcase with the ensemble summary file
# ifiles are not open
def pop_compare_raw_score(opts_dict, ifiles, timeslice, Var3d, Var2d):
    rmask_var = 'REGION_MASK'
    if not opts_dict['test_failure']:
        nbin = opts_dict['nbin']
    else:
        nbin = 1
    Zscore = np.zeros((len(Var3d) + len(Var2d), len(ifiles), (nbin)), dtype=np.float32)

    failure_count = np.zeros((len(ifiles)), dtype=np.int32)
    sum_file = nc.Dataset(opts_dict['sumfile'], 'r')
    for k, v in sum_file.variables.items():
        if k == 'ens_stddev2d':
            ens_stddev2d = v
        elif k == 'ens_avg2d':
            ens_avg2d = v
        elif k == 'ens_stddev3d':
            ens_stddev3d = v
        elif k == 'ens_avg3d':
            ens_avg3d = v
        elif k == 'time':
            ens_time = v

    # check time slice 0 for zeros....indicating an incomplete summary file
    sum_problem = False
    all_zeros = not np.any(ens_stddev2d[0, :, :])
    if all_zeros:
        print('ERROR: ens_stddev2d field in summary file was not computed.')
        sum_problem = True

    all_zeros = not np.any(ens_avg2d[0, :, :])
    if all_zeros:
        print('ERROR: ens_avg2d field in summary file was not computed.')
        sum_problem = True

    all_zeros = not np.any(ens_stddev3d[0, :, :, :])
    if all_zeros:
        print('ERROR: ens_stddev3d field in summary file was not computed.')
        sum_problem = True

    all_zeros = not np.any(ens_avg3d[0, :, :, :])
    if all_zeros:
        print('ERROR: ens_avg3d field in summary file was not computed.')
        sum_problem = True

    if sum_problem:
        print('=> EXITING....')
        sys.exit()

    npts3d = 0
    npts2d = 0
    is_SE = False
    ens_timeslice = len(ens_time)

    # Get the exact month from the file names
    n_timeslice = []
    in_file_names = []
    if not opts_dict['mpi_enable']:
        n_timeslice, in_file_names = get_files_from_glob(opts_dict)
        temp_list = []
        for i in n_timeslice:
            temp_list.append(i + 1)
        print('STATUS: Checkpoint month(s) = ', temp_list)

    skip_count = 0
    # Compare an individual file with ensemble summary file to get zscore
    for fcount, fid in enumerate(ifiles):
        print(' ')
        # If not in mpi_enable mode, the timeslice will be decided by the month of the input files
        if not opts_dict['mpi_enable']:
            timeslice = n_timeslice[fcount]

        o_fid = nc.Dataset(fid, 'r')
        otimeSeries = o_fid.variables
        rmask = otimeSeries[rmask_var]

        print('**********' + 'Run ' + str(fcount + 1) + ' (file=' + in_file_names[fcount] + '):')

        if timeslice >= ens_timeslice:
            print(
                'WARNING: The summary file contains only ',
                ens_timeslice,
                ' timeslices. Skipping this run evaluation for timeslice = ',
                timeslice,
                '...',
            )
            skip_count = skip_count + 1
            continue
        for vcount, var_name in enumerate(Var3d):
            orig = otimeSeries[var_name][0]
            FillValue = otimeSeries[var_name]._FillValue
            Zscore[vcount, fcount, :], has_zscore = calculate_raw_score(
                var_name,
                orig,
                npts3d,
                npts2d,
                ens_avg3d[timeslice][vcount],
                ens_stddev3d[timeslice][vcount],
                is_SE,
                opts_dict,
                FillValue,
                0,
                rmask,
            )
            if opts_dict['test_failure']:
                temp = Zscore[vcount, fcount, 0]
                print('          ' + '{:>10}'.format(var_name) + ': ' + '{:.2%}'.format(temp))
                if Zscore[vcount, fcount, :] < opts_dict['pop_threshold']:
                    failure_count[fcount] = failure_count[fcount] + 1

        for vcount, var_name in enumerate(Var2d):
            orig = otimeSeries[var_name][0]
            FillValue = otimeSeries[var_name]._FillValue
            # print var_name,timeslice
            Zscore[vcount + len(Var3d), fcount, :], has_zscore = calculate_raw_score(
                var_name,
                orig,
                npts3d,
                npts2d,
                ens_avg2d[timeslice][vcount],
                ens_stddev2d[timeslice][vcount],
                is_SE,
                opts_dict,
                FillValue,
                0,
                rmask,
            )
            if opts_dict['test_failure']:
                temp = Zscore[vcount + len(Var3d), fcount, 0]
                print('          ' + '{:>10}'.format(var_name) + ': ' + '{:.2%}'.format(temp))
                if Zscore[vcount + len(Var3d), fcount, :] < opts_dict['pop_threshold']:
                    failure_count[fcount] = failure_count[fcount] + 1

        if failure_count[fcount] > 0:
            print(
                '**********'
                + str(failure_count[fcount])
                + ' of '
                + str(len(Var3d) + len(Var2d))
                + ' variables failed, resulting in an overall FAIL'
                + '**********'
            )
        else:
            print(
                '**********'
                + str(failure_count[fcount])
                + ' of '
                + str(len(Var3d) + len(Var2d))
                + ' variables failed, resulting in an overall PASS'
                + '**********'
            )

        o_fid.close()

    sum_file.close()

    # give error msg is none of the files were valid
    if skip_count == len(ifiles):
        print('ERROR: no files to process with valid timeslices. Exiting...')
        sys.exit(2)

    if has_zscore:
        return Zscore, n_timeslice
    else:
        Zscore = 0
        return Zscore, n_timeslice


# Get the deficit row number of the standardized global mean matrix
# (AB: no longer used...)
def get_failure_index(the_array):
    mat_rows = the_array.shape[0]
    # mat_cols = the_array.shape[1]

    mat_rank = np.linalg.matrix_rank(the_array)
    deficit = mat_rows - mat_rank
    deficit_row = []
    x = 0
    while deficit > 0:
        for i in range(mat_rows):
            temp_mat = np.delete(the_array, i, axis=0)
            new_rank = np.linalg.matrix_rank(temp_mat)
            if new_rank == mat_rank:
                # print "removing row ", i
                if len(deficit_row) != 0:
                    # print "deficit_row=",deficit_row
                    x = i
                    for num, j in enumerate(deficit_row):
                        if j - num <= i:
                            # print "j=",j,"i=",i
                            x = x + 1
                    deficit_row.append(x)
                else:
                    deficit_row.append(i)

                the_array = temp_mat
                mat_rows = the_array.shape[0]
                mat_rank = new_rank
                deficit = mat_rows - mat_rank
                break
    return deficit_row


#
# Alternative method to get the linearly dependent rows (using QR for faster perf)
def get_dependent_vars_index(a_mat, mytol):
    # initialize
    dv_index = []

    # the_array is nvars x nens
    # nvars = a_mat.shape[0]

    # transpose so vars are the columns
    t_mat = a_mat.transpose()

    # now do a rank-revealing qr (pivots for stability)
    q_mat, r_mat, piv = sla.qr(t_mat, pivoting=True)

    # rank = num of nonzero diag of r
    r_mat_d = np.fabs(r_mat.diagonal())
    # print r_mat_d
    r_diag_rank = len(np.where(r_mat_d >= mytol)[0])
    # print("RANK:  ", "r_diag_rank = ", r_diag_rank, "mytol = ", mytol)

    rank_est = r_diag_rank
    # ind_vars_index = piv[0:rank_est]
    # these are the dependent variables
    dv_index = piv[rank_est:]

    return dv_index


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
