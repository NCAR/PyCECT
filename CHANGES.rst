PyCECT Change Log
=====================

Copyright 2020 University Corporation for Atmospheric Research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


VERSION 3.4.0
-------------

- Added MOM6 ensemble consistency testing (MOM-ECT). The new pyEnsSumMom6.py
  creates the ensemble summary file, and pyCECT.py --mom tests runs against it.
  The variables to test are listed in a json file (see mom6_ensemble.json).

- pyEnsSumMom6.py uses monthly files by default (--nyear, --nmonth, --start_year,
  --start_month), or files with several time steps each via --ntsteps. With more
  members than --esize, --rand and --seed select the members at random.

- New pyCECT.py options for MOM-ECT: --mom_tol, --mom_threshold, --tslice (time
  step to test), --rand (test one randomly selected file) and --noOutfile. MOM-ECT
  results are written to --outfile as a labeled table.

- The MOM6 summary file stores the time units and calendar, so pyCECT.py can
  test runs whose time uses a different reference date (test times are converted
  to the summary file's units before matching).

- Reduced memory use in pyEnsSumPop.py and pyEnsSumMom6.py (only the root task
  allocates the gathered results).

- pyCECT.py now runs in serial only. Removed the pyCECT.py options --mpi_enable,
  --casejson and --npick, and the pyEnsSumPop.py options --rand, --nrand, --seq
  and --jsondir.

- pyCECT.py exits with an error if more than one of --pop, --mom, --mpas and
  --cam is given, and error exits now return status 2.

- Bug fix: --saveResults did not save sum_std_mean correctly.


VERSION 3.2.0
-------------

- Migrated from Python 2 to Python 3.

- Added improved documentation via ReadtheDocs.


VERSION 3.1.1
--------------

- Minor bug fixes and I/O update for web_enabled interface.

VERSION 3.1.0
--------------

- Minor bug fixes.

- Removed pyNIO and pyNGL dependencies.

- Modified CAM variable exclusion process to potentially exclude more variables (via larger tolerance in rank calculation and identification of variables taking only a few constant values).

- Updated CAM option to print plots.


VERSION 3.0.7
-------------

- Added web_enabled mode and pbs submission script.



VERSION 3.0.5
-------------

- Minor release to address data cast problem in area_avg.

VERSION 3.0.2
-------------

- Minor release to remove tabs.


VERSION 3.0.1
-------------

- Minor release that can generate ensemble summary on 3D variable having dimension ilev, create boxplot on ensemble members, and can process with excluded variable list or included variable list.

VERSION 3.0.0
--------------

- "Ultra-fast", UF-CAM-ECT, tool released.


VERSIONS 2.0.1 - 2.0.3
--------------------

- Bug fixes.

VERSION 2.0.0
-------------

- Tools for POP (Ocean component) are released.


VERSION 1.0.0
-------------

- Initial release.

- Includes CAM (atmosphere compnent) tools: CECT and PyEnsSum.
