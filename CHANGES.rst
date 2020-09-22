PyReshaper Change Log
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

-"Ultra-fast", UF-CAM-ECT, tool released.


VERSIONS 2.0.1 - 2.0.3
--------------------

- Bug fixes.

  VERSION 2.0.0
-------------

 -Tools for POP (Ocean component) are released.


VERSION 1.0.0
-------------

 - Initial release.

 - Includes CAM (atmosphere compnent) tools: CECT and PyEnsSum.
   
   


 - Bugfixes
 - Disabling use of PyNIO backend (for now) due to unknown problems
 - Fixes and ease of use with I/O backend utilities
 
VERSION 1.0.2
-------------

 - Bugfixes
 
VERSION 1.0.1
-------------

 - Bugfixes

VERSION 1.0.0
-------------

 - New I/O backend manager.  Can work with netCDF4 or PyNIO.
 - Removing hierarchy to Reshaper classes, as there is only 1 (similarly
   removing hierarchy of matching Specifier classes)
 - No longer requires PyNIO to install...but you need netCDF4 or PyNIO to
   run the reshaper!
 - Adding ability to treat all 1D time-dependent variables as metadata
   with only one command-line option
 - Adding ability to extract out only some time-series files, instead of
   requiring all time-series files be extracted.  This option should be
   used sparingly and with caution!
 - Adding the ability to "chunk" over user-specified dimensions when
   reading and writing.  This finally allows for some control over memory
   use!


VERSION 0.9.10
--------------

 - Python 2.6 back-porting


VERSION 0.9.6
-------------

 - Split 'slice2series' script into 2 separate scripts: 's2smake' to generate
   specifier object files (specfiles), and 's2srun' to run a reshaper job
   with a given specifier object file
 - Now uses 'write mode' to determing if overwriting output files or skipping
   existing output files
 - Added capability to append to existing output files


VERSION 0.9.5
-------------

 - Fix bug in the 'slice2series' script
 - Adds a write to file option for the Specifiers
 - Modifying output message verbosity settings


VERSION 0.9.4
-------------

 - Newly automated versioning information
 - Bugfixes, performance improvements
 - New yellowstone testing code
 - Now measures read/write times separately in diagnostic data
 - Performance improvement (no explicit loop over time-steps in a time-slice
   file needed)
 - Enabling user-defined compression level for NetCDF4 output


VERSION 0.9.3
-------------

 - Bugfix: Now installs properly from PyPI


VERSION 0.9.2
-------------

 - Tagging new branch for version 0.9.2 release.
 - Restructured source tree
 - Installation of LICENSE file with package
 - Updates for upload to PyPI


VERSION 0.9.1
-------------
  
 - Added many new helper tools in the Yellowstone testing directory.
 - Perform a sort of the time-series variables by size before partitioning
   them across processors.  Since the partition strides through the list of
   variables names, by sorting the variables by size first, we do a reasonable
   job of selecting as many small variables as large on every processor
 - A few minor typo bugfixes.
 - Added the ability to output all metadata to a "once" file.  This metadata
   includes ALL variables that are NOT the time-series variables being written
   to file.  Hence, this can produce time-series files that are not entirely
   self-describing.
 - Fixed a bug in which a job hangs when using more ranks than variables
 - Switched to using the identified 'unlimited' dimension name instead of
   explicitly using the string 'time'.
 - Added the ability to process time-slice files with multiple time steps
   in each slice file. 
 - Added new plotting utility and module.  Some changes to the getsizes
   utility in the Yellowstone testing area.  Some more tweaks here, too.
 - Updated the PyReshaper to use the newly created PyTools/trunk.  (Will
   modify the externals to use a tag once the new PyTools has been tested and
   verified).
 - Renamed any ATM/LND 0.1 degree tests in the tests/yellowstone directory to 
   0.25 degree, which they actually are.  Similarly, any 0.25 degree OCN/ICE
   tests were named 0.1 degree.
 - Updated the Specifier and Reshaper Doxygen comments to use Doxypypy
   formatting.


VERSION 0.9.0
-------------

 - Initial release.  (Previously in prototype repo.)
 - Improvements to the Yellowstone testing scripts
 - Added new UCAR license agreement
