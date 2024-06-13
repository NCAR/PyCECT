# This file contain additional helper functions for the ECT setup framework in as described in Price-Broncucia et al. 2024

import numpy as np

import xarray as xr

from os import listdir
from os.path import isfile, join
import glob

import sklearn

from sklearn.decomposition import PCA
import sklearn.preprocessing


# Read MPAS Ensemble of numpy files. NetCDF files should have already been read, means calculated, then saved as individual numpy files.
def read_MPAS_ensemble_means(folder_name):
#     check for variable name file
    if isfile(folder_name + "/extracted_vars.npy"):
        extracted_var_names = np.load(folder_name + "/extracted_vars.npy")
    else:
        print("No variable name file found")
        return

    # check for gathered means file
    if isfile(folder_name + "/gathered_mpas_means.npy"):
        gathered_mpas_means = np.load(folder_name + "/gathered_mpas_means.npy")
        return gathered_mpas_means, extracted_var_names

    #     if no gathered file exists check that individual means files exist
    else:
        mean_files = glob.glob(folder_name + "/mpas_means_*.npy")
        if len(mean_files) > 0:
#             open first file to get time dimension
            first_mean_file = np.load(folder_name + "/mpas_means_0.npy")
            time_dim = first_mean_file.shape[-1]
            
#             make placeholder numpy array
            gathered_mpas_means = np.zeros((len(mean_files), len(extracted_var_names), time_dim))
            
#             iterate through means 
            for i in range(len(mean_files)):
                gathered_mpas_means[i, :, :] = np.load(folder_name + "/mpas_means_"+str(i)+".npy")
        
#             save gathered mean file
            np.save(folder_name + "/gathered_mpas_means.npy", gathered_mpas_means)
    
            return gathered_mpas_means, extracted_var_names 
#         no means files
        else:
            print("No gathered file found, and no individual mean files found")
            return
        

# Read CESM Ensemble of numpy files. NetCDF files should have already been read, means calculated, then saved as individual numpy files.
def read_CESM_ensemble_means(folder_name):
#     check for variable name file
    if isfile(folder_name + "/extracted_vars.npy"):
        extracted_var_names = np.load(folder_name + "/extracted_vars.npy")
    else:
        print("No variable name file found")
        return

    # check for gathered means file
    if isfile(folder_name + "/gathered_cesm_means.npy"):
        gathered_mpas_means = np.load(folder_name + "/gathered_cesm_means.npy")
        return gathered_mpas_means, extracted_var_names

    #     if no gathered file exists check that indiviual means files exist
    else:
        mean_files = glob.glob(folder_name + "/cesm_means_*.npy")
        if len(mean_files) > 0:
#             open first file to get time dimension
            first_mean_file = np.load(folder_name + "/cesm_means_0.npy")
            time_dim = first_mean_file.shape[-1]
            
#             make placeholder numpy array
            gathered_mpas_means = np.zeros((len(mean_files), len(extracted_var_names), time_dim))
            
#             iterate through means 
            for i in range(len(mean_files)):
                gathered_mpas_means[i, :, :] = np.load(folder_name + "/cesm_means_"+str(i)+".npy")
        
#             save gathered mean file
            np.save(folder_name + "/gathered_cesm_means.npy", gathered_mpas_means)
    
            return gathered_mpas_means, extracted_var_names 
#         no means files
        else:
            print("No gathered file found, and no individual mean files found")
            return

# Function to generate PCA values from "true" ensemble
def ECT_ensemble_step(ensemble_data):
#     use sklearn to scale data, has some functionality to avoid divide by zero etc
    scaler = sklearn.preprocessing.StandardScaler().fit(ensemble_data)
    
#     save mean and scale to be used to transform new data values.
    mean_shift = scaler.mean_
    scale_multiply = scaler.scale_
    
#     transform ensemble data
    ensemble_scaled = scaler.transform(ensemble_data)
    
#     create PCA object
    pca = PCA(svd_solver='full')

#     train PCA with scaled values
    pca.fit(ensemble_scaled)
    
    # save PC scores and vectors
    PC_scores = np.sqrt(pca.explained_variance_)
    PC_vectors = pca.components_
    
    return mean_shift, scale_multiply, PC_vectors, PC_scores


# Function to run test step of ECT
# test_vals should be uncentered and un-normalized matrix of (N_new x N_variables)
# return true on fail, false on non fail
def ECT_test_step(test_vals, mean_shift, scale_multiply, PC_vectors, PC_scores, PC_count, N_new, N_PCfails, N_runFails, m_sigma):

    # Make sure the right number of samples were passed in.
    if test_vals.shape[0] != N_new:
        print("Sample size different from expected based on N_new")
    else:
        # Center and normalize test values according to ensemble
        centered_test_vals = test_vals - mean_shift
        scaled_test_vals = centered_test_vals / scale_multiply

        # print(scaled_test_vals[:, np.array([39, 40, 41, 42, 53])])
        
        # Apply PC_vector matrix to generate transformed samples
        transformed_test_vals = scaled_test_vals @ PC_vectors.T
        # print(f"test_score: {transformed_test_vals[:, 0]}")        
#         Which transformed variables are greater than m_sigma multiples away from the mean
        test_sigma_fails = np.abs(transformed_test_vals / PC_scores) > m_sigma
        # print(np.abs(transformed_test_vals / PC_scores))

#         Which variables failed at least N_runFails
        test_run_fails = np.sum(test_sigma_fails, axis=0) > (N_runFails - 1)
        # print(test_run_fails[0:PC_count])
    
#         Looking at the first PC_count PC variables, do at least N_PCfails variables fail?
        fail_bool = np.sum(test_run_fails[0:PC_count]) > (N_PCfails - 1)
        # print(np.sum(test_run_fails[0:PC_count]))
        # print(N_PCfails)
        # print(fail_bool)

        # print("hello")
        # print(fail_bool)
        # print(np.sum(test_sigma_fails, axis=0))
        # print(np.abs(np.sum(test_sigma_fails, axis=0) @ np.linalg.inv(PC_vectors.T)))
    
        return fail_bool, np.sum(test_sigma_fails, axis=0), np.abs(np.sum(test_sigma_fails, axis=0) @ np.linalg.inv(PC_vectors.T))
        
def anderson_adjustment(estimated_scores, n):
    var_count = len(estimated_scores)
    adjusted_roots = np.zeros(var_count)

    estimated_roots = estimated_scores ** 2
    
    for i in range(var_count):
        dif_term = 0
        for j in range(var_count):
            if j != i:
                dif_term += estimated_roots[j] / (estimated_roots[i] - estimated_roots[j])
            
        adjusted_roots[i] = estimated_roots[i] * (1 - (1/n) * dif_term)

    print(estimated_roots)
    print(adjusted_roots)
    adjusted_scores = np.sqrt(adjusted_roots)
        
    return adjusted_scores