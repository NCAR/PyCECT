# To be run after test jobs have finished
import json
import f90nml
import numpy as np
from matplotlib import pyplot as plt
import sys, os



def main(argv):

    # read in testing parameter files
    with open(argv, 'r') as f:
        test_params = json.load(f)

    print(test_params)

    mpas_src = test_params["file_paths"]["mpas_src"]
    init_dir = test_params["file_paths"]["init_dir"]
    namelist_name = test_params["file_paths"]["namelist_name"]

    init_copy_dir = test_params["file_paths"]["init_copy_dir"]
    test_output_dir = test_params["file_paths"]["test_output_dir"]

    true_sum_file = test_params["file_paths"]["true_sum_file"]

    verify_runs = test_params["verify_runs"]

    test_vars = test_params["test_vars"]

    t_slice = test_params["output_timestep"]
    PCA_dims = test_params["ect_pca_dims"]

    orig_namelist = f90nml.read(f"{init_dir}/{namelist_name}")

    for each in test_vars:
        var_name = each["var_name"]
        namelist_preface = each["namelist_preface"]

        # TODO update dtype to float after initial run
        neg_test_orders = np.array(each["neg_test_orders"])
        pos_test_orders = np.array(each["pos_test_orders"])

        default_var_value = orig_namelist[namelist_preface][var_name]

        print(f"Starting post ECT steps for {var_name}")

        # this reflects perturbations away from zero where test value = default_val * (1 + perturbations[i])
        perturbations = np.append(-10**np.array(neg_test_orders, dtype=float), 10**np.array(pos_test_orders, dtype=float))

        test_vals = default_var_value * (1 + perturbations)

        avg_pca_fails = np.empty(len(perturbations))

        avg_eet_fails = np.empty(len(perturbations))

        for i, order in enumerate(neg_test_orders):
            # test folder name (change from negative to positive)
            test_folder = f"{var_name}_perturb_neg{order}"

            # check if pca file exists and read in pca and eet files if so
            if os.path.isfile(f"{test_output_dir}/{test_folder}/pca.npy"):
                print(f"PCA fail file found for {var_name}_perturb_neg{order}")

                pca_fail_file = np.load(f"{test_output_dir}/{test_folder}/pca.npy")

                avg_pca_fails[i] = np.mean(pca_fail_file.sum(axis=0))

                eet_file = np.load(f"{test_output_dir}/{test_folder}/eet.npy")

                avg_eet_fails[i] = (eet_file[1] - eet_file[0])/eet_file[1]


            else:
                print(f"PCA fail file not found for {var_name}_perturb_neg{order}")

        for i, order in enumerate(pos_test_orders):
            # test folder name (positive)
            test_folder = f"{var_name}_perturb_{order}"

            # check if pca file exists and read in pca and eet files if so
            if os.path.isfile(f"{test_output_dir}/{test_folder}/pca.npy"):
                print(f"PCA fail file found for {var_name}_perturb_{order}")

                pca_fail_file = np.load(f"{test_output_dir}/{test_folder}/pca.npy")

                avg_pca_fails[i + len(neg_test_orders)] = np.mean(pca_fail_file.sum(axis=0))

                eet_file = np.load(f"{test_output_dir}/{test_folder}/eet.npy")

                avg_eet_fails[i + len(neg_test_orders)] = (eet_file[1] - eet_file[0])/eet_file[1]


            else:
                print(f"PCA fail file not found for {var_name}_perturb_{order}")

        # sort outputs in order so they plot correctly
        perturbations, test_vals, avg_pca_fails, avg_eet_fails = map(list, zip(*sorted(zip(perturbations, test_vals, avg_pca_fails, avg_eet_fails))))

        print(f"Outputs for {var_name}")
        print(f"Perturbations: {perturbations}")
        print(f"Resulting test values: {test_vals}")
        print(f"Average PCA failures: {avg_pca_fails}")
        print(f"EET rate: {avg_eet_fails}")

        # plot perturbation outputs
        plot_dir_exists = os.path.exists(f"{test_output_dir}/plots")
        if not plot_dir_exists:
                # create plot directory
            os.makedirs(f"{test_output_dir}/plots")

        # log perturbation plot
        plt.plot(perturbations, avg_pca_fails/PCA_dims)
        plt.xscale("symlog")
        plt.title(f"MPAS UF-ECT PCA Fails vs\n {var_name} Perturbation")
        plt.ylabel("PCA Fail Percent")
        plt.xlabel("Perturbation Factor")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_log_perturb_plot.png")
        plt.clf()
        
        # log value plot
        plt.plot(test_vals, avg_pca_fails/PCA_dims)
        plt.xscale("symlog")
        plt.title(f"MPAS UF-ECT PCA Fails vs\n {var_name} Value")
        plt.ylabel("PCA Fail Percent")
        plt.xlabel(f"{var_name}")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_log_val_plot.png")
        plt.clf()

        # non-log value plot
        plt.plot(test_vals, avg_pca_fails/PCA_dims)
        plt.title(f"MPAS UF-ECT PCA Fails vs\n {var_name} Value")
        plt.ylabel("PCA Fail Percent")
        plt.xlabel(f"{var_name}")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_val_plot.png")
        plt.clf()

        # log EET vs perturbation plot
        plt.plot(perturbations, avg_eet_fails)
        plt.xscale("symlog")
        plt.title(f"MPAS UF-ECT EET Rate vs\n {var_name} Perturbation")
        plt.ylabel("EET Fail Percent")
        plt.xlabel(f"Perturbation Factor")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_log_eet_perturb_plot.png")
        plt.clf()

        # log EET vs value plot
        plt.plot(test_vals, avg_eet_fails)
        plt.xscale("symlog")
        plt.title(f"MPAS UF-ECT EET Rate vs\n {var_name} Value")
        plt.ylabel("EET Fail Percent")
        plt.xlabel(f"{var_name}")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_log_eet_value_plot.png")
        plt.clf()

        # log EET vs value plot
        plt.plot(test_vals, avg_eet_fails)
        plt.title(f"MPAS UF-ECT EET Rate vs\n {var_name} Value")
        plt.ylabel("EET Fail Percent")
        plt.xlabel(f"{var_name}")
        plt.savefig(f"{test_output_dir}/plots/{var_name}_eet_value_plot.png")
        plt.clf()




if __name__ == '__main__':
    main(sys.argv[1])