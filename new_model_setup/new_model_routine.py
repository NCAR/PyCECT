# This file contain library functions for the ECT setup framework in as described in Price-Broncucia et al. 2024

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import shapiro
from os import listdir
from os.path import isfile, join
import glob


from sklearn.decomposition import PCA

from functools import reduce

from helper_ect_functions import *

import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
plt.rcParams['figure.dpi']=200

plt.rcParams['lines.linewidth'] = 3

# Step 1, read in sum files
def read_summary_series(folder, file_prefix, file_suffix, starting_timestep, save_interval):
    # get list of *.nc files in directory
    summ_files = glob.glob(folder + f"/{file_prefix}*")
    file_count = len(summ_files)
    
    datasets = []
    variable_lists = []
    
#     There are differing variable counts in the summary files due to exclusions.
#     Identify shared set

    for i in range(file_count):
        datasets.append(xr.open_dataset(f"{folder}/{file_prefix}{i}{file_suffix}", engine='netcdf4'))
        variable_lists.append(datasets[i].vars)
        print(f"{i}: {len(datasets[i].vars)} vars")
    
    shared_vars = reduce(np.intersect1d, variable_lists)
    
    print(f"{len(shared_vars)} shared variables")
    
    n_vars = len(shared_vars)
    ens_size = datasets[0].dims['ens_size']
    
    all_standard_means = np.zeros((file_count, n_vars, ens_size))
    
    for i in range(file_count):
        sorter = np.argsort(datasets[i].vars)
        idx_of_shared_in_dataset = sorter[np.searchsorted(datasets[i].vars, shared_vars, sorter=sorter)]
        
        all_standard_means[i, :, :] = xr.open_dataset(f"{folder}/{file_prefix}{i}{file_suffix}").standardized_gm[idx_of_shared_in_dataset, :]
    
    timesteps = list(range(starting_timestep, file_count * save_interval + starting_timestep, save_interval))
    
    return all_standard_means, timesteps, shared_vars

# Step 2, Calculate Shapiro-Wilks P-score over time
def shapiro_wilks_over_time(all_standard_means, timesteps, shared_vars, title="", mark_timestep=None):
    num_vars = len(shared_vars)
    num_times = len(timesteps)
    
    shap_ps = np.zeros((num_vars, num_times))
    shap_scores = np.zeros((num_vars, num_times))
    
    for j in range(num_times):
        for i in range(num_vars):
            shap_score, shap_p = shapiro(all_standard_means[j, i, :])
            shap_ps[i, j] = shap_p
            shap_scores[i, j] = shap_score
            
    non_normal_vars = np.sum(shap_ps < .05, axis=0)
    non_normal_vars_percentage = non_normal_vars / num_vars
    
    plt.plot(timesteps, non_normal_vars_percentage)
    plt.xlabel("Model Timestep")
    plt.xticks(timesteps[::2], map(str, timesteps[::2]))
    plt.ylabel("Non-Normal Variable Percentage")
    plt.title(title)
    
    if mark_timestep != None:
        plt.axvline(x=mark_timestep, label=f"Timestep: {mark_timestep}", color="red", linestyle="dashed")
        plt.legend()
        
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()

    return (timesteps, non_normal_vars_percentage)

def plot_shapiro_wilks_over_time(timesteps, non_normal_vars_percentage, title="", mark_timestep=None):
    plt.plot(timesteps, non_normal_vars_percentage)
    plt.xlabel("Model Timestep")
    plt.xticks(timesteps[::2], map(str, timesteps[::2]))
    plt.ylabel("Non-Normal Variable Percentage")
    plt.title(title)
    
    if mark_timestep != None:
        plt.axvline(x=mark_timestep, label=f"Timestep: {mark_timestep}", color="red", linestyle="dashed")
        plt.legend()
        
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()
    
# Step 3, Determine Number of PC Dimensions to Use
def timestep_to_idx(timestep, all_timesteps):
    return all_timesteps.index(timestep)

def var_explained_sample(standardized_gms, sample_size, sample_repeats):
    dims = standardized_gms.shape[0]
    total_samples = standardized_gms.shape[1]
    scores_variances = np.zeros((sample_repeats, dims))
    
#     seeds uniquely by default
    rng = np.random.default_rng()
    
    for i in range(sample_repeats):
        pca = PCA()
#         choose samples without replacement
        idxs = rng.choice(total_samples, sample_size, replace=False)
    
        pca.fit(standardized_gms[:, idxs].T)
        scores_variances[i, :] = pca.explained_variance_
    
    cum_sum = np.cumsum(scores_variances, axis=1) / np.sum(scores_variances, axis=1)[:, None]
        
    return scores_variances, cum_sum

def stable_PCs_required(folder, file_prefix, file_suffix, timestep, all_timesteps, variance_explained, title="", vertical_line = None):
    idx = timestep_to_idx(timestep, all_timesteps)
    
    sum_file = xr.open_dataset(f"{folder}/{file_prefix}{idx}{file_suffix}")
    
    standardized_gm = sum_file.standardized_gm
    
    dims = standardized_gm.shape[0]
    
    sample_sizes = [dims]

    test_variances = np.zeros((len(sample_sizes), dims))
    test_cum_variances = np.zeros((len(sample_sizes), dims))
    min_pca_included = [0]
    
    scores_variances, cum_sum = var_explained_sample(standardized_gm, sample_sizes[0], 10)
    test_variances[0, :] = np.mean(scores_variances, axis=0)
    test_cum_variances[0, :] = np.mean(cum_sum, axis=0)
    min_pca_included[0] = np.argwhere(test_cum_variances[0, :] > .95)[0][0]
    
    i = 1
    stable_count = 1
    
    while stable_count < 5:
        print(i)
        sample_sizes.append(sample_sizes[i-1] + 50)
        
        scores_variances, cum_sum = var_explained_sample(standardized_gm, sample_sizes[i], 10)
        test_variances = np.vstack([test_variances, np.mean(scores_variances, axis=0)])
        test_cum_variances = np.vstack([test_cum_variances, np.mean(cum_sum, axis=0)])

        min_pca_included.append(np.argwhere(test_cum_variances[i, :] > variance_explained)[0][0])
        
#         test for 128
#         print("Variance Explained by 128")
#         print(test_cum_variances[i, 127])
        
        if min_pca_included[i] == min_pca_included[i-1]:
            stable_count += 1
        else:
            stable_count = 1
            
        i += 1
        
    print(min_pca_included)
            
    plot_stable_PCs(sample_sizes, min_pca_included, vertical_line=vertical_line, title=title)
    
    return (sample_sizes, min_pca_included)

def plot_stable_PCs(sample_sizes, min_pca_included, vertical_line=None, horizontal_line=None, title=""):
    fig, ax1 = plt.subplots()

    if vertical_line != None:
        plt.axvline(x=vertical_line, linestyle='dashed', color = 'red')
    
    if horizontal_line != None:
        plt.axhline(y=horizontal_line, linestyle='dashed', color = 'red')

    color = 'blue'
    ax1.set_ylabel('Minimum PCA Dimensions to Explain Variance')
    ax1.plot(sample_sizes, min_pca_included)
    ax1.set_xlabel('Ensemble Size')

    fig.tight_layout()
    plt.title(title)
    plt.show()

# How much variance is explained by certain PC dimensions

def var_explained_at_fixed_pc(folder, file_prefix, file_suffix, timestep, all_timesteps, samples, pcs):
    idx = timestep_to_idx(timestep, all_timesteps)
    
    sum_file = xr.open_dataset(f"{folder}/{file_prefix}{idx}{file_suffix}")
    
    standardized_gm = sum_file.standardized_gm
    
    _, cum_sum = var_explained_sample(standardized_gm, samples, 10)

    mean_cum_sum = np.mean(cum_sum, axis=0)

    print(mean_cum_sum[pcs-1])

# Theoretical FPR
from scipy.stats import binom
from scipy.stats import norm
from scipy.optimize import root_scalar

def theoretical_fpr(m, pca_count):
    sigma_rate = norm.cdf(-m) * 2
    p = (sigma_rate**2 * (1 - sigma_rate)) *3 + sigma_rate**3
    return 1 - binom.cdf(2, pca_count, p)

def theoretical_fpr_range(start_pca, end_pca, m):
    PC_range = range(start_pca, end_pca)
    ect_failure_rate_predicted_by_pc_count = [theoretical_fpr(m, i) for i in PC_range]
    return ect_failure_rate_predicted_by_pc_count

def theoretical_fpr_for_opt(m, pca_count, goal):
    return theoretical_fpr(m, pca_count) - goal

def find_theoretical_m(PC_count, goal_FPR):
    sol = root_scalar(theoretical_fpr_for_opt, args=(PC_count, goal_FPR), bracket=[2., 5.])

    new_m = sol.root

    return new_m

#Step 4 Sample Size to Meet FPR
def fpr(standardized_gms, ensemble_size, test_repeats, PC_count, m_sigma):

#     select samples from full summary file to construct ect ensemble
    full_indices = np.arange(standardized_gms.shape[1])
    ensemble_indices = np.random.choice(full_indices, ensemble_size, replace=False)
    full_test_indices = np.delete(full_indices, ensemble_indices)

    ensemble_samples = standardized_gms[:, ensemble_indices].values

    # Step 3: Use samples to characterize PCA distribution
    mean_shift, scale_multiply, PC_vectors, PC_scores = ECT_ensemble_step(ensemble_samples.T)

    sigma_fails = np.zeros(PC_scores.shape[0])

    temp_fails = 0

    for j in range(test_repeats):
        N_new = 3
        N_PCfails = 3
        N_runFails = 2

        test_indices = np.random.choice(full_test_indices, N_new, replace=False)
        test_samples = standardized_gms[:, test_indices].values

        fail_bool, sigma_fail, _ = ECT_test_step(test_samples.T, mean_shift, scale_multiply, 
                    PC_vectors, PC_scores, PC_count, N_new, N_PCfails, N_runFails, m_sigma)

        if fail_bool:
            temp_fails += 1

        sigma_fails += sigma_fail

    return temp_fails/test_repeats

def run_fpr_tests(folder, file_prefix, file_suffix, timestep, all_timesteps, pc_count, sample_size_start, m_sigma, title=""):
    idx = timestep_to_idx(timestep, all_timesteps)
    
    sum_file = xr.open_dataset(f"{folder}/{file_prefix}{idx}{file_suffix}")
    
    standardized_gm = sum_file.standardized_gm
    
    ens_size = standardized_gm.shape[1]
    
    test_repeats = 100
    ensemble_repeats = 100

    sample_sizes = list(range(sample_size_start, ens_size - 50, 50))

    fpr_by_ensemble_size = np.zeros(len(sample_sizes))

    for i, sample_size in enumerate(sample_sizes):
        print(i)
        temp = 0
        for j in range(ensemble_repeats):
            temp += fpr(standardized_gm, sample_size, test_repeats, pc_count, m_sigma)

        fpr_by_ensemble_size[i] = temp/ensemble_repeats
        
    plot_fpr_tests(sample_sizes, fpr_by_ensemble_size, title)
    
    return (sample_sizes, fpr_by_ensemble_size)
    
def plot_fpr_tests(sample_sizes, fpr_by_ensemble_size, title, vertical_line=None):
    plt.axhline(y=.005, color="red", linestyle='dashed', label="Goal FPR=0.5%")
    if vertical_line != None:
        plt.axvline(x=vertical_line, linestyle='dashed', color = 'red')
    
    plt.plot(sample_sizes, fpr_by_ensemble_size)
    plt.title(title)
    plt.xlabel("Ensemble Size")
    plt.ylabel("False Positive Rate (FPR)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

    plt.ylim(bottom=0)
#     plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    
    plt.show()

# For Paper Plotting
def make_cov_plot(means, all_timesteps, single_timesteps, title=""):
    idx = timestep_to_idx(single_timesteps, all_timesteps)
    cov = np.cov(means[idx, :, :], rowvar=True)
    sns.heatmap(cov, vmin=-1, vmax=1, cmap=mpl.colormaps['seismic'])
    
    plt.title(title)
    plt.xlabel("Variable Index")
    plt.ylabel("Variable Index")
    
    plt.show()
    
    return cov

def find_unique_cov_pairs(cov, shared_vars, cutoff):
    vars_1, vars_2 = np.where(np.abs(cov) > cutoff)
    cross = np.where(vars_1 != vars_2)
    big_cor_1 = shared_vars[vars_1[cross]]
    big_cor_2 = shared_vars[vars_2[cross]]
    unique_cor_pairs = []
    for i in range(len(big_cor_1)):
        found = False
        for j in range(len(unique_cor_pairs)):
            if big_cor_1[i] == unique_cor_pairs[j][0]:
                if big_cor_2[i] == unique_cor_pairs[j][1]:
                    found=True
                    break

            if big_cor_2[i] == unique_cor_pairs[j][0]:
                if big_cor_1[i] == unique_cor_pairs[j][1]:
                    found=True
                    break
        if found == False:
            unique_cor_pairs.append([big_cor_1[i], big_cor_2[i]])

    string_unique_cor_pairs = []
    for each in unique_cor_pairs:
        str_1 = each[0].decode('UTF-8').strip()
        str_2 = each[1].decode('UTF-8').strip()
        string_unique_cor_pairs.append([str_1, str_2])

    return string_unique_cor_pairs

def plot_ensemble_spread(means, all_timesteps, shared_vars, var_string, title=""):
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['figure.dpi']=300
    
    idx = np.where(shared_vars == var_string)[0][0]
    
    ensemble_size = means.shape[2]

    bins = 12
    num_time_steps = len(all_timesteps)
    surface_data = np.empty((bins, num_time_steps))
    surface_data[:] = np.nan
    
    for i in range(num_time_steps):
        surface_data[:, i], bin_edges = np.histogram(means[i, idx, :], bins=bins)
    
    X, Y = np.meshgrid(np.arange(bins), all_timesteps)
    
#     fig = plt.figure(figsize=(15, 20))
    ax1 = plt.figure().add_subplot(projection='3d')
    ax1.set_box_aspect(aspect = (1,4,1))
    
    ax1.plot_wireframe(X, Y, surface_data.T/ensemble_size, cstride=0)
    
    ax1.view_init(25, 30)
    
    var_name = var_string.decode('UTF-8').strip()
    
    ax1.set_title(f"MPAS-A Ensemble Distribution \n Variable: '{var_name}'", y=0.8)

    ax1.set_ylabel('Timestep')
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel('Relative Density', rotation=90)
    
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticks(all_timesteps)
    ax1.yaxis.set_ticklabels(map(str, all_timesteps))
    ax1.zaxis.set_ticklabels([])
    
    ax1.yaxis.labelpad=30
    ax1.zaxis.labelpad=-10
    
    
    plt.tight_layout()
    plt.show()
    
from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def plot_continuous_test_results(data_path, scale="log"):
    test_data = pd.read_csv(data_path)
    test_data["perturbations"] = test_data.direction * 10. ** (test_data.order)
    
    
    tested_parameters = pd.unique(test_data.parameter)
    
    for param in tested_parameters:
        var_name = param
        data_slice = test_data[test_data.parameter == param]
        data_slice = data_slice.reset_index(drop=True)
        sorted_data_slice = data_slice.sort_values('perturbations')
        
        test_fail_indices = sorted_data_slice.index[sorted_data_slice.fail == True]
        
        perturbations = np.array([round_to_1(i) for i in sorted_data_slice.perturbations])
        eet_failure_rate = sorted_data_slice.EET/100
        
        y = eet_failure_rate
        x = perturbations
        
        # remove indices of tests where model failed to complete test run
        filtered_x = np.delete(x, test_fail_indices)
        filtered_y = np.delete(y, test_fail_indices)
        
        plt.scatter(filtered_x, filtered_y, label="Test Locations")
        plt.plot(filtered_x, filtered_y, alpha = .2, linestyle="dashed")
        plt.axvline(x=0, color='b', label="Default Variable Value")
        
        for fail_idx in test_fail_indices:
            plt.scatter(x=x[fail_idx], y=y[fail_idx], color='r', marker='x', label="Model Run Failure")
        
        linthresh = np.min(np.abs(perturbations))
        
        if scale == "log":
            plt.xscale("symlog", linthresh=linthresh)
            
        num_neg_test_orders = np.sum(perturbations < 0)
        perturbations_with_zero = np.insert(perturbations, num_neg_test_orders, 0)
        x_labels = [f"{i:.1%}"for i in perturbations_with_zero]
        plt.xticks(perturbations_with_zero, labels=x_labels, rotation = 50)
            
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.ylim(-0.1, 1.1)
        
        title = f"MPAS-A UF-ECT EET Failure Rate vs\n{var_name} Perturbation"
        plt.ylabel("EET Failure Rate")
        plt.xlabel("Perturbation Magnitude")
        file_name = f"{var_name}_eet_perturb_plot"
        
        plt.title(title)
        plt.legend()
        
        plt.show()

    return

def fail_rates(means, test_repeats, ensemble_size, all_timesteps, single_timestep, m_sigma):

    idx = timestep_to_idx(single_timestep, all_timesteps)
    
    full_indices = np.arange(means.shape[2])
    
    ensemble_indices = np.random.choice(full_indices, ensemble_size, replace=False)
    
    full_test_indices = np.delete(full_indices, ensemble_indices)

    ensemble_samples = means[idx, :, ensemble_indices]

    # Step 2: Generate samples to construct PCA samples

    test_fails = 0

    # Step 3: Use samples to characterize PCA distribution
    mean_shift, scale_multiply, PC_vectors, PC_scores = ECT_ensemble_step(ensemble_samples)
#     print(mean_shift)
#     print(scale_multiply)

    # Step 4: Generating new samples from remaining members of mpas run (test_indices), 
    # characterize the false positive rate at various hyperparameters.

    pc_fails = np.zeros(PC_scores.shape[0])
    var_fails = np.zeros(PC_scores.shape[0])

    temp_fails = 0
    
    PC_count = means.shape[1]

    for j in range(test_repeats):
        N_new = 3
        N_PCfails = 3
        N_runFails = 2

        test_indices = np.random.choice(full_test_indices, N_new, replace=False)
        test_samples = means[idx, :, test_indices]

        fail_bool, pc_fail, var_fail = ECT_test_step(test_samples, mean_shift, scale_multiply, 
                    PC_vectors, PC_scores, PC_count, N_new, N_PCfails, N_runFails, m_sigma)

        if fail_bool:
            temp_fails += 1

        pc_fails += pc_fail/N_new
        
        var_fails += var_fail/N_new

    test_fails = temp_fails/N_new

    
    # return test_fails
    return test_fails, pc_fails, var_fails

def fpr_by_pc_and_var(means, test_repeats, ensemble_repeats, ensemble_size, all_timesteps, single_timestep, m_sigma):
    timestep_count, var_count, total_ensemble = means.shape
    
    total_test_fails = 0
    total_pc_fails = np.zeros(var_count)
    total_var_fails = np.zeros(var_count)
    
    for i in range(ensemble_repeats):
#         print(i)
        test_fails, pc_fails, var_fails = fail_rates(means, test_repeats, ensemble_size, all_timesteps, single_timestep, m_sigma)
        
        total_test_fails += test_fails
        total_pc_fails += pc_fails
        total_var_fails += var_fails
        
    mean_test_fails = total_test_fails / (ensemble_repeats * test_repeats)
    mean_pc_fails = total_pc_fails / (ensemble_repeats * test_repeats)
    mean_var_fails = total_var_fails / (ensemble_repeats * test_repeats)
    
#     print(mean_test_fails)
    
    plt.plot(mean_pc_fails)
    plt.xlabel("PC Dimension")
    plt.ylabel("Failure Rate")
    plt.title(f"Average Fails by PC\nMPAS-A 7.3, Timestep {single_timestep}, Ensemble Size {ensemble_size}")
    plt.ylim(bottom=0)
    plt.show()
    
    plt.plot(mean_var_fails)
    plt.xlabel("Variable Index")
    plt.ylabel("Relative Contribution to PC Failure")
    plt.title(f"Average Fails by Variable\nMPAS-A 7.3, Timestep {single_timestep}, Ensemble Size {ensemble_size}")
    plt.ylim(bottom=0)
    plt.show()

def var_explained_by_sample_size(standardized_gms, sample_size, sample_repeats):
    dims = standardized_gms.shape[0]
    total_samples = standardized_gms.shape[1]
    scores_variances = np.zeros((sample_repeats, dims))
    
#     seeds uniquely by default
    rng = np.random.default_rng()
    
    for i in range(sample_repeats):
        pca = PCA()
#         choose samples without replacement
        idxs = rng.choice(total_samples, sample_size, replace=False)
    
        pca.fit(standardized_gms[:, idxs].T)
        scores_variances[i, :] = pca.explained_variance_
    
    cum_sum = np.cumsum(scores_variances, axis=1) / np.sum(scores_variances, axis=1)[:, None]
        
    return scores_variances, cum_sum

def variance_explained_vs_ensemble_size_plot(means, test_repeats, ensemble_sizes, 
                                             all_timesteps, single_timestep, title):
    
    timestep_count, var_count, total_ensemble = means.shape
    idx = timestep_to_idx(single_timestep, all_timesteps)
    
    dims = var_count
    
    test_variances = np.zeros((len(ensemble_sizes), dims))
    test_cum_variances = np.zeros((len(ensemble_sizes), dims))
    
    for i, ensemble_size in enumerate(ensemble_sizes):
        scores_variances, cum_sum = var_explained_sample(means[idx, :, :], ensemble_size, test_repeats)
        test_variances[i, :] = np.mean(scores_variances, axis=0)
        test_cum_variances[i, :] = np.mean(cum_sum, axis=0)

    plt.rcParams['lines.linewidth'] = 2
    plt.plot(test_cum_variances.T)
    plt.axhline(y=.95, linestyle='dashed', color = 'red')
    
    z = list(plt.yticks()[0])[:-1] + [.95]
    plt.yticks(z)
    plt.legend(ensemble_sizes, title="Ensemble Size")
    plt.title(title)
    plt.xlabel("Principal Components Included")
    plt.ylabel("Cummulative Variance Explained")
    plt.show()