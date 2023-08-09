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

        neg_test_orders = np.array(each["neg_test_orders"], dtype=float)
        pos_test_orders = np.array(each["pos_test_orders"], dtype=float)

        default_var_value = orig_namelist[namelist_preface][var_name]

        print(f"Starting post ECT steps for {var_name}")

        # this reflects perturbations away from zero where test value = default_val * (1 + perturbations[i])
        perturbations = np.append(-10**np.array(neg_test_orders, dtype=float), 10**np.array(pos_test_orders, dtype=float))

        test_vals = default_var_value * (1 + perturbations)

        avg_pca_fails = np.empty(len(perturbations))

        avg_eet_fails = np.empty(len(perturbations))

        model_run_failures = np.full(len(perturbations), False)

        for i, order in enumerate(neg_test_orders):
            # test folder name (change from negative to positive)
            test_folder = f"{var_name}_perturb_neg{order}"

            # check if pca file exists and read in pca and eet files if so
            if os.path.isfile(f"{test_output_dir}/{test_folder}/pca.npy"):
                print(f"PCA failure file found for {var_name}_perturb_neg{order}")

                pca_fail_file = np.load(f"{test_output_dir}/{test_folder}/pca.npy")

                # with np.printoptions(threshold=np.inf):
                #     print(pca_fail_file)

                avg_pca_fails[i] = np.mean(pca_fail_file.sum(axis=0))

                eet_file = np.load(f"{test_output_dir}/{test_folder}/eet.npy")

                avg_eet_fails[i] = (eet_file[1] - eet_file[0])/eet_file[1]


            # check if fail file is found
            elif os.path.isfile(f"{test_output_dir}/{test_folder}/fail.txt"):
                print(f"Model fail file found for {var_name}_perturb_neg{order}")
                model_run_failures[i] = 1

            else:
                print(f'''Neither PCA failure file nor Model run failure file found found for {var_name}_perturb_neg{order} \n
                      Did you run the post_run_script.py?''')

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

            # check if fail file is found
            elif os.path.isfile(f"{test_output_dir}/{test_folder}/fail.txt"):
                print(f"Model fail file found for {var_name}_perturb_neg{order}")
                model_run_failures[i + len(neg_test_orders)] = 1

            else:
                print(f'''Neither PCA failure file nor Model run failure file found found for {var_name}_perturb_{order} \n
                      Did you run the post_run_script.py?''')

        # sort outputs in order so they plot correctly
        perturbations, test_vals, avg_pca_fails, avg_eet_fails = map(np.array, zip(*sorted(zip(perturbations, test_vals, avg_pca_fails, avg_eet_fails))))

        print(f"Outputs for {var_name}")
        print(f"Perturbations: {perturbations}")
        print(f"Resulting test values: {test_vals}")
        print(f"Average PCA failures: {avg_pca_fails}")
        print(f"EET rate: {avg_eet_fails}")

        # plot perturbation outputs
        plot_data = (var_name, model_run_failures, len(neg_test_orders), avg_pca_fails/PCA_dims, avg_eet_fails, perturbations, test_vals, default_var_value)
        for pca_arg in [True, False]:
            for perturb_arg in [True, False]:
                for log_arg in ["log", "linear"]:
                    plot_test_results(plot_data, test_output_dir, plot_pca=pca_arg, plot_perturbations=perturb_arg, scale = log_arg)
        


def plot_test_results(plot_data, file_path, scale="log", plot_pca = True, plot_perturbations = True):
    var_name, model_run_failures, num_neg_test_orders, pca_failure_rate, eet_failure_rate, perturbations, test_vals, default_val = plot_data

    plot_dir_exists = os.path.exists(f"{file_path}/plots")
    if not plot_dir_exists:
        # create plot directory
        os.makedirs(f"{file_path}/plots")

    file_path = file_path + "/plots"

    if plot_pca:
        y = pca_failure_rate
    else:
        y = eet_failure_rate

    if plot_perturbations:
        x = perturbations
        linthresh = np.min(np.abs(perturbations))
    else:
        x = test_vals - default_val
        linthresh = np.min(np.abs(perturbations)) * default_val

    # remove indices of tests where model failed to complete test run
    filtered_x = np.delete(x, model_run_failures)
    filtered_y = np.delete(y, model_run_failures)

    # plt.plot(x, y)

    plt.scatter(filtered_x, filtered_y, label="Test Locations")
    plt.plot(filtered_x, filtered_y, alpha = .2, linestyle="dashed")
    plt.axvline(x=0, color='b', label="Default Variable Value")

    for i, model_failure in enumerate(model_run_failures):
        if model_failure:
            plt.axvline(x=x[i], color='r', label="Model Run Failure")


    if scale == "log":
        plt.xscale("symlog", linthresh=linthresh)

    if plot_perturbations:
        perturbations_with_zero = np.insert(perturbations, num_neg_test_orders, 0)
        plt.xticks(perturbations_with_zero, rotation = 50)
    else:
        test_vals_with_default = np.insert(test_vals, num_neg_test_orders, default_val)
        plt.xticks(test_vals_with_default - default_val, labels = ["%.5f" % z for z in test_vals_with_default], rotation = 50)
    
    if plot_pca and plot_perturbations:
        title = f"MPAS UF-ECT PCA Failure Rate vs\n{var_name} Perturbation"
        plt.ylabel("PCA Failure Rate")
        plt.xlabel("Perturbation Magnitude")
        file_name = f"{var_name}_pca_perturb_plot"
    elif plot_pca and not plot_perturbations:
        title = f"MPAS UF-ECT PCA Failure Rate vs\n{var_name} Value"
        plt.ylabel("PCA Failure Rate")
        plt.xlabel(f"{var_name} Value")
        file_name = f"{var_name}_pca_var_value_plot"
    elif not plot_pca and plot_perturbations:
        title = f"MPAS UF-ECT EET Failure Rate vs\n{var_name} Perturbation"
        plt.ylabel("EET Failure Rate")
        plt.xlabel("Perturbation Magnitude")
        file_name = f"{var_name}_eet_perturb_plot"
    else:
        title = f"MPAS UF-ECT EET Failure Rate vs\n{var_name} Value"
        plt.ylabel("EET Failure Rate")
        plt.xlabel(f"{var_name} Value")
        file_name = f"{var_name}_eet_var_value_plot"

    if scale == "log":
        file_name = file_name + "_log"

    plt.title(title)
    plt.legend()
    plt.savefig(file_path + "/" + file_name, bbox_inches="tight")

    plt.clf()

    return


if __name__ == '__main__':
    main(sys.argv[1])