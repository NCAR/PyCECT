# To be run after test jobs have finished
import json
import f90nml
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyCECT import main as ECT


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

    # orig_namelist = f90nml.read(f"{init_dir}/{namelist_name}")

    for each in test_vars:
        var_name = each["var_name"]
        namelist_preface = each["namelist_preface"]

        # TODO update dtype to float after initial run
        neg_test_orders = np.array(each["neg_test_orders"])
        pos_test_orders = np.array(each["pos_test_orders"])

        # default_var_value = orig_namelist[namelist_preface][var_name]

        print(f"Starting postrun steps for {var_name}")

        for order in neg_test_orders:
            # test folder name (change from negative to positive)
            test_folder = f"{var_name}_perturb_neg{order}"

            # Create symlinks to history files
            output_folder = test_output_dir + "/" + test_folder

            command = f"find {test_output_dir}/{test_folder}/{test_folder}* -name \"history_full*\" -exec cp -s '{{}}' {test_output_dir}/{test_folder}/history_files/ \;"
            
            os.system(command)
            # print(command)

            # Run PyCECT
            args_for_ECT = [f'--sumfile={true_sum_file}', f"--indir={test_output_dir}/{test_folder}/history_files", f"--tslice={t_slice}", f"--nPC={PCA_dims}", "--mpas", f"--eet={verify_runs}", f"--savePCAMat={test_output_dir}/{test_folder}/pca.npy", f"--saveEET={test_output_dir}/{test_folder}/eet.npy", "mpi_enable"]
            ECT(args_for_ECT)



if __name__ == '__main__':
    main(sys.argv[1])
    