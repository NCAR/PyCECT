# This is to be run after test jobs have finished
import argparse
import json
import os
import sys
from glob import glob

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from pyCECT import main as ECT


def main(argv):
    # # read in testing parameter files
    # with open(argv, 'r') as f:
    #     test_params = json.load(f)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_params_json', help='Path to the json file containing the test parameters'
    )
    parser.add_argument(
        '--force',
        help='Force a remake of the ECT save-results files (and thus remake spatial means from history files) even if the ECT save-result files already exist',
        action='store_true',
    )

    args = parser.parse_args()
    test_params_file = args.test_params_json
    with open(test_params_file, 'r') as f:
        test_params = json.load(f)

    print('Test parameters:')
    print(test_params)

    test_output_dir = test_params['file_paths']['test_output_dir']

    true_sum_file = test_params['file_paths']['true_sum_file']

    verify_runs = test_params['verify_runs']

    test_vars = test_params['test_vars']

    t_slice = test_params['output_timestep']
    PCA_dims = test_params['ect_pca_dims']

    for each in test_vars:
        print(f'Test type: {each["test_type"]}')

        # binary test
        if each['test_type'] == 'binary_test':
            print(f"Test name: {each['test_name']}")

            test_name = each['test_name']

            # file paths
            indir = f'{test_output_dir}/{test_name}/history_files'
            eet_filepath = f'{test_output_dir}/{test_name}/eet.npy'
            savefile_path = f'{test_output_dir}/{test_name}/savefile.nc'

            # check if pca values have already been calculated for this variable perturbation combo?
            if os.path.isfile(savefile_path) and not args.force:
                print(f'Existing save file found for {test_name} test')
                print('Running ECT from save-results file')

                # Run PyCECT
                run_ECT(
                    true_sum_file,
                    indir,
                    t_slice,
                    PCA_dims,
                    verify_runs,
                    saveEET_file=eet_filepath,
                    use_saveResults_file=savefile_path,
                )

            else:
                # Check and make sure history files were written for all verify runs before trying to copy
                if len(glob(f'{test_output_dir}/{test_name}/**/history_full*')) < verify_runs:
                    print(
                        f'Insufficient output files for {test_name}, categorized as model failure.'
                    )

                    with open(f'{test_output_dir}/{test_name}/fail.txt', 'w') as f:
                        f.write(
                            'This variable/perturbation combination did not produce output files for all verify members and is thus categorized as a model failure.'
                        )

                else:
                    # Create symlinks to history files
                    command = f"find {test_output_dir}/{test_name}/{test_name}* -name \"history_full*\" -exec cp -sf '{{}}' {test_output_dir}/{test_name}/history_files/ \\;"

                    os.system(command)
                    # print(command)

                    # Run PyCECT
                    run_ECT(
                        true_sum_file,
                        indir,
                        t_slice,
                        PCA_dims,
                        verify_runs,
                        saveEET_file=eet_filepath,
                        saveResults_file=savefile_path,
                    )

        # Namelist Float test
        else:
            print(f'Test name: {each["var_name"]}')
            var_name = each['var_name']
            #            namelist_preface = each['namelist_preface']

            neg_test_orders = np.array(each['neg_test_orders'], dtype=float)
            pos_test_orders = np.array(each['pos_test_orders'], dtype=float)

            # default_var_value = orig_namelist[namelist_preface][var_name]

            print(f'Starting postrun steps for {var_name}')

            for order in neg_test_orders:
                # test folder name (change from negative to positive)
                float_test_name = f'{var_name}_perturb_neg{order}'
                test_folder = float_test_name

                # file paths
                indir = f'{test_output_dir}/{test_folder}/history_files'
                eet_filepath = f'{test_output_dir}/{test_folder}/eet.npy'
                savefile_path = f'{test_output_dir}/{test_folder}/savefile.nc'

                # check if save file have already been calculated for this variable perturbation combo?
                if os.path.isfile(savefile_path) and not args.force:
                    print(f'Existing save file found for {float_test_name}')
                    print('Running ECT from save-results file')

                    # Run PyCECT
                    run_ECT(
                        true_sum_file,
                        indir,
                        t_slice,
                        PCA_dims,
                        verify_runs,
                        saveEET_file=eet_filepath,
                        use_saveResults_file=savefile_path,
                    )

                else:
                    # Check and make sure history files were written for all verify runs before trying to copy
                    if len(glob(f'{test_output_dir}/{test_folder}/**/history_full*')) < verify_runs:
                        print(
                            f'Insufficient output files for {test_folder}, categorized as model failure.'
                        )

                        with open(f'{test_output_dir}/{test_folder}/fail.txt', 'w') as f:
                            f.write(
                                'This variable/perturbation combination did not produce output files for all verify members and is thus categorized as a model failure.'
                            )

                    else:
                        # Create symlinks to history files
                        command = f"find {test_output_dir}/{test_folder}/{test_folder}* -name \"history_full*\" -exec cp -sf '{{}}' {test_output_dir}/{test_folder}/history_files/ \\;"

                        os.system(command)
                        # print(command)

                        # Run PyCECT
                        run_ECT(
                            true_sum_file,
                            indir,
                            t_slice,
                            PCA_dims,
                            verify_runs,
                            saveEET_file=eet_filepath,
                            saveResults_file=savefile_path,
                        )

            for order in pos_test_orders:
                # test folder name (positive)
                float_test_name = f'{var_name}_perturb_{order}'
                test_folder = float_test_name

                # file paths
                indir = f'{test_output_dir}/{test_folder}/history_files'
                eet_filepath = f'{test_output_dir}/{test_folder}/eet.npy'
                savefile_path = f'{test_output_dir}/{test_folder}/savefile.nc'

                # check if pca values have already been calculated for this variable perturbation combo?
                if os.path.isfile(savefile_path) and not args.force:
                    print(f'Existing save file found for {float_test_name}')
                    print('Running ECT from save-results file')

                    # Run PyCECT
                    run_ECT(
                        true_sum_file,
                        indir,
                        t_slice,
                        PCA_dims,
                        verify_runs,
                        saveEET_file=eet_filepath,
                        use_saveResults_file=savefile_path,
                    )

                else:
                    # Check and make sure history files were written for all verify runs before trying to copy
                    if len(glob(f'{test_output_dir}/{test_folder}/**/history_full*')) < verify_runs:
                        print(
                            f'Insufficient output files for {test_folder}, categorized as model failure.'
                        )

                        with open(f'{test_output_dir}/{test_folder}/fail.txt', 'w') as f:
                            f.write(
                                'This variable/perturbation combination did not produce output files for all verify members and is thus categorized as a model failure.'
                            )

                    else:
                        # Create symlinks to history files
                        command = f"find {test_output_dir}/{test_folder}/{test_folder}* -name \"history_full*\" -exec cp -sf '{{}}' {test_output_dir}/{test_folder}/history_files/ \\;"

                        os.system(command)
                        # print(command)

                        # Run PyCECT
                        run_ECT(
                            true_sum_file,
                            indir,
                            t_slice,
                            PCA_dims,
                            verify_runs,
                            saveEET_file=eet_filepath,
                            saveResults_file=savefile_path,
                        )


def run_ECT(
    sumfile,
    indir,
    tslice,
    nPC,
    eet_count,
    saveEET_file=None,
    saveResults_file=None,
    use_saveResults_file=None,
):
    args_for_ECT = [
        f'--sumfile={sumfile}',
        f'--indir={indir}',
        f'--tslice={tslice}',
        f'--nPC={nPC}',
        '--mpas',
        f'--eet={eet_count}',
        '--mpi_enable',
    ]
    if saveEET_file is not None:
        args_for_ECT.append(f'--saveEET={saveEET_file}')
    if saveResults_file is not None:
        args_for_ECT.append(f'--saveResults={saveResults_file}')
    if use_saveResults_file is not None:
        args_for_ECT.append(f'--useSavedResults={use_saveResults_file}')

    # print(args_for_ECT)

    ECT(args_for_ECT)


if __name__ == '__main__':
    main(sys.argv[1])
