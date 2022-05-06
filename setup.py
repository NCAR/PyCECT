#!/usr/bin/env python3

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open('README.rst') as f:
    long_description = f.read()

CLASSIFIERS = [
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

setup(
    name='PyCECT',
    description='A library providing a statistical measurement of consistency between an accepted ensemble and a test set of simulations.',
    long_description=long_description,
    python_requires='>=3.6',
    maintainer='Allison Baker',
    maintainer_email='abaker@ucar.edu',
    classifiers=CLASSIFIERS,
    url='https://github.com/NCAR/PyCECT',
    project_urls={
        'Documentation': 'https://pycect.readthedocs.io',
        'Source': 'https://github.com/NCAR/PyCECT',
        'Tracker': 'https://github.com/NCAR/PyCECT/issues',
    },
    packages=find_packages(exclude=('tests',)),
    package_dir={'pycect': 'pycect'},
    include_package_data=True,
    install_requires=install_requires,
    license='Apache 2.0',
    zip_safe=False,
    keywords='PyCECT',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
)
