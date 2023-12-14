# This script will create folders for each test combination (variable type and perturbation amount).
# It is intended for float namelist variables.
# Perturbations are created based on orders of 10, with exponents specified for both negative and positive directions from the default values. This allows asymmetrical tests like if there is floor to valid values.

# Test parameter should be saved in a json file so they can be easily edited and shared between steps of the procedure.

import json
import os, sys
import f90nml
import numpy as np
import copy
import shutil

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

    verify_runs = test_params["verify_runs"]

    test_vars = test_params["test_vars"]

    orig_namelist = f90nml.read(f"{init_dir}/{namelist_name}")

    # print("Original Namelist:")
    # print(orig_namelist)

    for each in test_vars:
        print(f'Test type: {each["test_type"]}')

        # reset directories in case they has been changed by a test
        mpas_src = test_params["file_paths"]["mpas_src"]
        init_dir = test_params["file_paths"]["init_dir"]

        # binary test
        if each["test_type"] == "binary_test":
            # set test specific directories
            if len(each["mod_mpas_src"]) > 0:
                mpas_src = each["mod_mpas_src"]
            if len(each["mod_mpas_init_dir"]) > 0:
                init_dir = each["mod_mpas_init_dir"]

            print(f"MPAS source: {mpas_src}")
            print(f"MPAS Init Directory: {init_dir}")

            test_name = each["test_name"]
            print(f"Creating directories for {test_name} test")

            init_copy_folder = f"{init_copy_dir}/{test_name}"
            command = f"cp -aTf {init_dir}/ {init_copy_folder}"
            os.system(command)

            # create empty directories for outputs
            output_folder = test_output_dir + f"/{test_name}"

            if os.path.exists(output_folder):
                try:
                    shutil.rmtree(output_folder)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

            os.mkdir(output_folder)
            os.mkdir(output_folder + "/history_files")

            # submit jobs
            run_cmd = f"python {mpas_src}/ensemble.py -rd {output_folder} -c {init_copy_folder} --verify_size {verify_runs} -s --verify"

            os.system(run_cmd)

        # Namelist Float test
        else:
            var_name = each["var_name"]
            namelist_preface = each["namelist_preface"]

            # read perturb orders in and set as float to ensure consistent formatting
            neg_test_orders = np.array(each["neg_test_orders"], dtype=float)
            pos_test_orders = np.array(each["pos_test_orders"], dtype=float)

            default_var_value = orig_namelist[namelist_preface][var_name]

            print(f"Creating directories for {var_name}. Default value: {default_var_value}")
            print(f"Perturbation orders: negative {neg_test_orders} and positive {pos_test_orders}")

            mod_nml = copy.deepcopy(orig_namelist)

            for order in neg_test_orders:
                # create initial conditions copy directories (helpful if initial directory restart files are absolute symlinks for size)
                init_copy_folder = f"{init_copy_dir}/{var_name}_perturb_neg{order}"
                command = f"cp -af {init_dir}/ {init_copy_folder}"
                os.system(command)

                # create empty directories for outputs
                output_folder = test_output_dir + f"/{var_name}_perturb_neg{order}"
                
                if os.path.exists(output_folder):
                    try:
                        shutil.rmtree(output_folder)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                os.mkdir(output_folder)
                os.mkdir(output_folder + "/history_files")

                # modify namelist params
                mod_nml[namelist_preface][var_name] = default_var_value * (1 - 10.**order)
                mod_nml.write(f"{init_copy_folder}/{namelist_name}", force=True)

                # Create readme
                with open(test_output_dir + f"/{var_name}_perturb_neg{order}/history_files/readme.txt", 'w') as f:
                    print(f"Output files in the directory were created using perturbed values", file=f)
                    print(f"{var_name} changed from default value of {default_var_value} to {mod_nml[namelist_preface][var_name]}", file=f)

                # submit jobs
                run_cmd = f"python {mpas_src}/ensemble.py -rd {output_folder} -c  {init_copy_folder} --verify_size {verify_runs} -s --verify"

                os.system(run_cmd)

            for order in pos_test_orders:
                # create initial conditions copy directories (helpful if initial directory restart files are absolute symlinks for size)
                init_copy_folder = f"{init_copy_dir}/{var_name}_perturb_{order}"
                command = f"cp -af {init_dir}/ {init_copy_folder}"
                os.system(command)

                # create empty directories for outputs
                output_folder = test_output_dir + f"/{var_name}_perturb_{order}"

                if os.path.exists(output_folder):
                    try:
                        shutil.rmtree(output_folder)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                os.mkdir(output_folder)
                os.mkdir(output_folder + "/history_files")

                # modify namelist params
                mod_nml[namelist_preface][var_name] = default_var_value * (1 + 10.**order)
                mod_nml.write(f"{init_copy_folder}/{namelist_name}", force=True)

                # Create readme
                with open(test_output_dir + f"/{var_name}_perturb_{order}/history_files/readme.txt", 'w') as f:
                    print(f"Output files in the directory were created using perturbed values", file=f)
                    print(f"{var_name} changed from default value of {default_var_value} to {mod_nml[namelist_preface][var_name]}", file=f)

                # submit jobs
                run_cmd = f"python {mpas_src}/ensemble.py -rd {output_folder} -c  {init_copy_folder} --verify_size {verify_runs} -s --verify"
                os.system(run_cmd)

if __name__ == '__main__':
    main(sys.argv[1])