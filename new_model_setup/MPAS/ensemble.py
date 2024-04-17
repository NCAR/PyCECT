#!/usr/bin/python
import argparse
import getopt
import os
import random
import sys
import xml.etree.ElementTree as ET

# ==============================================================================
# create ensembles members from a control case
# ==============================================================================

# generate <num_pick> positive random integers in [0, end-1]
# can't have any duplicates


def random_pick(num_pick, end):
    ar = range(0, end)
    rand_list = random.sample(ar, num_pick)
    return rand_list


# get the pertlim corressponding to the random int
# modified 10/23 to go to 2000 (previously only 1st 900 unique)
# modified 3/24 to go to 3996 (previously 1998 unique)
def get_pertlim_uf(rand_num):
    i = rand_num
    if i == 0:
        ptlim = '0'
    elif i > 3996:
        print("don't support sizes > 3996")
    else:  # generate perturbation
        if i > 1998:
            orig = i
            i = orig - 1998
        else:
            orig = 0

        if i <= 1800:  # [1 - 1800]
            if i <= 900:  # [1-900]
                j = 2 * int((i - 1) / 100) + 101
            elif i <= 1000:  # [901 - 1000]
                j = 2 * int((i - 1) / 100) + 100
            elif i <= 1800:  # [1001-1800]
                j = 2 * int((i - 1001) / 100) + 102
            k = (i - 1) % 100  # this is just last 2 digits of i-1
            if i % 2 != 0:  # odd
                ll = j + int(k / 2) * 18
                ippt = str(ll).zfill(3)
                ptlim = '0.' + ippt + 'd-13'
            else:  # even
                ll = j + int((k - 1) / 2) * 18
                ippt = str(ll).zfill(3)
                ptlim = '-0.' + ippt + 'd-13'
        else:  # [1801 - 2000]
            if i <= 1900:  # [1801-1900]
                j = 1
            else:  # [1901-2000]
                j = 2
            k = (i - 1) % 100
            if i % 2 != 0:  # odd
                ll = j + int(k / 2) * 2
                ippt = str(ll).zfill(3)
                ptlim = '0.' + ippt + 'd-13'
            else:  # even
                ll = j + int((k - 1) / 2) * 2
                ippt = str(ll).zfill(3)
                ptlim = '-0.' + ippt + 'd-13'

        if orig > 0:
            # adjust
            if i % 2 != 0:  # odd
                ptlim = '1.' + ippt + 'd-13'
            else:  # even
                ptlim = '-1.' + ippt + 'd-13'

    return ptlim


def main(argv):
    # directory with the executable (where this script is at the moment)
    stat_dir = os.path.dirname(os.path.realpath(__file__))
    print('STATUS: stat_dir = ' + stat_dir)

    # parse arguments
    parser = argparse.ArgumentParser(description='Create and submit mpas ensemble runs.')
    parser.add_argument(
        '-rd',
        '--run_dir',
        default=None,
        dest='run_dir_root',
        help='Directory to create run\
    directories in (default: None).',
        required=True,
    )
    parser.add_argument(
        '-c',
        '--control_run',
        default=None,
        dest='control_run',
        help='Path to the control run\
    that will be duplicated for the ensemble members (default = None).',
        required=True,
    )
    parser.add_argument(
        '-v',
        '--verify',
        default=False,
        dest='verify',
        action='store_true',
        help='To create \
    random verifications runs (instead of an ensemble).',
    )
    parser.add_argument(
        '-es',
        '--ens_size',
        default=10,
        dest='ens_size',
        help='Total number of ensemble \
    members (default = 10).',
        type=int,
    )
    parser.add_argument(
        '-vs',
        '--verify_size',
        default=3,
        dest='verify_size',
        help='Total number of ensemble \
    members (default = 3).',
        type=int,
    )
    parser.add_argument(
        '-st',
        '--start',
        default=0,
        dest='ens_start',
        help='Ensemble run to start with\
    (default = 0).',
        type=int,
    )
    parser.add_argument(
        '-rs',
        '--run_script',
        default='chey-run.sh',
        dest='run_script',
        help='Run script name to\
    submit to PBS queue in the control run directory (default: chey-run.sh).',
    )
    parser.add_argument(
        '-s',
        '--submit',
        default=False,
        dest='submit',
        action='store_true',
        help='Indicates\
    that jobs should be submitted to the queue (default: False).',
    )

    args = parser.parse_args()
    print(args)

    submit = args.submit
    run_script = args.run_script
    run_dir_root = args.run_dir_root
    control_run = args.control_run
    ens_size = args.ens_size
    ens_start = args.ens_start
    verify = args.verify
    verify_size = args.verify_size

    if verify:
        run_type = 'verify'
        run_size = verify_size
    else:
        run_type = 'ensemble'
        run_size = ens_size

    # some parameter checking
    if run_size > 0:
        if run_size > 3996:
            print('Error: cannot have an ensemble size greater than 3996.')
            sys.exit()
        print('STATUS: ensemble size = ' + str(run_size))
    else:
        print('Error: cannot have an ensemble size less than 1.')
        sys.exit()

    if not os.path.exists(run_dir_root):
        print(
            'ERROR: the specified directory for the new runs  (',
            run_dir_root,
            ') does not exist. Exiting',
        )
        sys.exit()

    if not os.path.exists(control_run):
        print('ERROR: the specified control run  (', control_run, ') does not exist. Exiting')
        sys.exit()

    # generate random pertlim(s) for verify
    if run_type == 'verify':
        end_range = 200
        rand_ints = random_pick(verify_size, end_range)

    # assume that the control case has the right run length and output stream settings
    # assume it has a submission script

    base_dir_name = os.path.basename(os.path.normpath(control_run))
    print('STATUS: base_dir_name = ', base_dir_name)
    for i in range(ens_start, run_size):
        # run dir name
        iens = '{0:04d}'.format(i)

        new_run = base_dir_name + '.' + iens

        # does this exist already?
        checkdir = run_dir_root + '/' + new_run
        if os.path.exists(checkdir):
            print('WARNING: Directory ', checkdir, ' already exists ... skipping ...')
            continue
        else:
            print('STATUS: creating run ' + new_run)

        # make a copy of control run in run_dir_root
        os.chdir(run_dir_root)
        command = 'cp -r ' + control_run + ' ' + new_run
        os.system(command)
        os.chdir(new_run)
        # thisdir = os.getcwd()

        # set pertlim in namelist.atmosphere file
        if run_type == 'verify':
            this_pert = get_pertlim_uf(rand_ints[i])
        else:  # full ensemble
            this_pert = get_pertlim_uf(i)

        new_line = 'config_pertlim = ' + str(this_pert) + '\n'
        # get old line if i==ens_start
        if i == ens_start:
            with open('namelist.atmosphere', 'r') as f:
                all_lines = f.readlines()
                old_line = 'nope'
                for line in all_lines:
                    if line.find('config_pertlim') >= 0:
                        old_line = line

                if old_line == 'nope':
                    print('Error: no pertlim found!')
                    sys.exit()

        with open('namelist.atmosphere', 'r') as r:
            text = r.read().replace(old_line, new_line)
        with open('namelist.atmosphere', 'w') as w:
            w.write(text)

        # set run length and output freq (currently assume correct in the control case)

        # modify the run script for derecho
        new_name = '#PBS -N  mpas.' + iens + '\n'
        execute_line_1 = f'mpiexec_mpt {stat_dir}/atmosphere_model'
        execute_line_2 = f'mpiexec {stat_dir}/atmosphere_model'
        with open(run_script, 'r') as f:
            all_lines = f.readlines()

            for line in all_lines:
                # find job name
                if line.find('#PBS -N') >= 0:
                    old_name = line

                # find execute mpt line
                if line.find('mpiexec_mpt') >= 0:
                    old_execute_line = line
                    execute_line = execute_line_1

                # find execute mpt line
                elif line.find('mpiexec') >= 0:
                    old_execute_line = line
                    execute_line = execute_line_2

        with open(run_script, 'r') as r:
            text = r.read().replace(old_name, new_name)
            text = text.replace(old_execute_line, execute_line)

        with open(run_script, 'w') as w:
            w.write(text)

        # modify the output file name in streams.atmosphere xml file
        tree = ET.parse('streams.atmosphere')
        root = tree.getroot()
        for i in root.iter('stream'):
            name = i.attrib['name']
            if name == 'output' or name == 'custom_output' or name == 'diagnostics':
                # print(i.attrib)
                fname = i.attrib['filename_template']
                sp = os.path.splitext(fname)
                nname = sp[0] + '.' + iens + sp[1]
                i.set('filename_template', nname)
                # break

        tree.write('streams.atmosphere')

        # submit the run script?
        if submit:
            command = 'qsub ' + run_script
            os.system(command)

    # Final output
    if run_type == 'verify':
        print('STATUS: ---MPAS-ECT VERIFICATION CASES COMPLETE---')
        print('Set up ', run_size, ' cases using the following pertlim values:')
        for i in range(0, run_size):
            print('     ', get_pertlim_uf(rand_ints[i]))
    else:
        print('STATUS: --MPAS ENSEMBLE CASES COMPLETE---')


if __name__ == '__main__':
    main(sys.argv[1:])
