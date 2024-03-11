# To be run after test jobs have finished
import json
import os
import sys
from glob import glob

import f90nml
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pyCECT import main as ECT


def main(argv):
    # read in testing parameter files
    with open(argv, 'r') as f:
        test_params = json.load(f)

    print(test_params)

    mpas_src = test_params['file_paths']['mpas_src']
    init_dir = test_params['file_paths']['init_dir']
    namelist_name = test_params['file_paths']['namelist_name']

    init_copy_dir = test_params['file_paths']['init_copy_dir']
    test_output_dir = test_params['file_paths']['test_output_dir']

    true_sum_file = test_params['file_paths']['true_sum_file']

    verify_runs = test_params['verify_runs']

    test_vars = test_params['test_vars']

    t_slice = test_params['output_timestep']
    PCA_dims = test_params['ect_pca_dims']

    # orig_namelist = f90nml.read(f"{init_dir}/{namelist_name}")

    for each in test_vars:
        print(f'Test type: {each["test_type"]}')

        # reset directories in case they has been changed by a test
        mpas_src = test_params['file_paths']['mpas_src']
        init_dir = test_params['file_paths']['init_dir']

        # binary test
        if each['test_type'] == 'binary_test':
            print(f'Test name: {each["test_name"]}')
            # set test specific directories
            if len(each['mod_mpas_src']) > 0:
                mpas_src = each['mod_mpas_src']
            if len(each['mod_mpas_init_dir']) > 0:
                init_dir = each['mod_mpas_init_dir']

            test_name = each['test_name']

            # check if pca values have already been calculated for this variable perturbation combo?
            if os.path.isfile(f'{test_output_dir}/{test_name}/pca.npy'):
                print(f'Existing PCA file found for {test_name} test')

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
                    command = f"find {test_output_dir}/{test_name}/{test_name}* -name \"history_full*\" -exec cp -s '{{}}' {test_output_dir}/{test_name}/history_files/ \;"

                    os.system(command)
                    # print(command)

                    # Run PyCECT
                    args_for_ECT = [
                        f'--sumfile={true_sum_file}',
                        f'--indir={test_output_dir}/{test_name}/history_files',
                        f'--tslice={t_slice}',
                        f'--nPC={PCA_dims}',
                        '--mpas',
                        f'--eet={verify_runs}',
                        f'--savePCAMat={test_output_dir}/{test_name}/pca.npy',
                        f'--saveEET={test_output_dir}/{test_name}/eet.npy',
                        '--mpi_enable',
                        '--printVars',
                    ]
                    ECT(args_for_ECT)

        # Namelist Float test
        else:
            print(f'Test name: {each["var_name"]}')
            var_name = each['var_name']
            namelist_preface = each['namelist_preface']

            neg_test_orders = np.array(each['neg_test_orders'], dtype=float)
            pos_test_orders = np.array(each['pos_test_orders'], dtype=float)

            # default_var_value = orig_namelist[namelist_preface][var_name]

            print(f'Starting postrun steps for {var_name}')

            for order in neg_test_orders:
                # test folder name (change from negative to positive)
                test_folder = f'{var_name}_perturb_neg{order}'

                # check if pca values have already been calculated for this variable perturbation combo?
                if os.path.isfile(f'{test_output_dir}/{test_folder}/pca.npy'):
                    print(f'Existing PCA file found for {var_name}_perturb_neg{order}')

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
                        command = f"find {test_output_dir}/{test_folder}/{test_folder}* -name \"history_full*\" -exec cp -s '{{}}' {test_output_dir}/{test_folder}/history_files/ \;"

                        os.system(command)
                        # print(command)

                        # Run PyCECT
                        args_for_ECT = [
                            f'--sumfile={true_sum_file}',
                            f'--indir={test_output_dir}/{test_folder}/history_files',
                            f'--tslice={t_slice}',
                            f'--nPC={PCA_dims}',
                            '--mpas',
                            f'--eet={verify_runs}',
                            f'--savePCAMat={test_output_dir}/{test_folder}/pca.npy',
                            f'--saveEET={test_output_dir}/{test_folder}/eet.npy',
                            'mpi_enable',
                        ]
                        ECT(args_for_ECT)

            for order in pos_test_orders:
                # test folder name (positive)
                test_folder = f'{var_name}_perturb_{order}'

                # check if pca values have already been calculated for this variable perturbation combo?
                if os.path.isfile(f'{test_output_dir}/{test_folder}/pca.npy'):
                    print(f'Existing PCA file found for {var_name}_perturb_neg{order}')

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
                        command = f"find {test_output_dir}/{test_folder}/{test_folder}* -name \"history_full*\" -exec cp -s '{{}}' {test_output_dir}/{test_folder}/history_files/ \;"

                        os.system(command)
                        # print(command)

                        # Run PyCECT
                        args_for_ECT = [
                            f'--sumfile={true_sum_file}',
                            f'--indir={test_output_dir}/{test_folder}/history_files',
                            f'--tslice={t_slice}',
                            f'--nPC={PCA_dims}',
                            '--mpas',
                            f'--eet={verify_runs}',
                            f'--savePCAMat={test_output_dir}/{test_folder}/pca.npy',
                            f'--saveEET={test_output_dir}/{test_folder}/eet.npy',
                            'mpi_enable',
                        ]
                        ECT(args_for_ECT)


if __name__ == '__main__':
    main(sys.argv[1])
