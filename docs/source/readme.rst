
Overview
==============================

The Community Earth System Model Ensemble
Consistency Test (CESM-ECT or CECT) suite was developed as an
alternative to requiring bitwise identical output for quality
assurance. This objective test provides a statistical measurement
of consistency between an accepted ensemble created
by small initial temperature perturbations and a test set of
CESM simulations.  Recently the CECT framework has been extended to work
with MPAS-A (https://mpas-dev.github.io/atmosphere/atmosphere.html).

The pyCECT  package, or *python CESM Ensemble Consistency Test*
package contains the necessary tools to to compare the results of a set of new (modified)
CESM (or MPAS-A) simulations against the accepted ensemble (pyCECT) as well as the tools to
create the ensemble summary files (pyEnsSum, pyEnsSumPop, and pyEnsMPAS). These
modules will be explained in more detail.

CESM notes:
---------------------
1. The pyCECT package is also (optionally) included in CESM (Community Earth System
   Model) via Externals.cfg.  See:

    https://github.com/ESCOMP/CESM/

2. Creating the ensemble summaries (via pyEnsSum or pyEnsSumPop) is
    typically done by the CESM software developers.  See:

    http://www.cesm.ucar.edu/models/cesm2/python-tools/

3. A web-based interface to this tool is available here:

   http://www.cesm.ucar.edu/models/cesm2/verification/

MPAS notes:
---------------------

1. MPAS-A summary files may be generated via pyEnsSumMPAS.  As this functionality
   is new, summary files are not yet available in the MPAS-A repo.

2. Please contact us with any questions/issues.


Setup Framework notes:
--------------------------

See the README file PyCECT/new_model_setup for information on examples of how to use the setup framework for determining parameters for your own model.  (This approach is described in Price-Broncucia 2025, listed below.)


Relevant publications:
-------------------------

Teo Price-Broncucia, Allison Baker, Dorit Hammerling, Michael Duda, and Rebecca Morrison, “The Ensemble Consistency Test: From CESM to MPAS and Beyond”, Geoscientific Model Development, 18, pp. 2349-2372, 2025.
https://gmd.copernicus.org/articles/18/2349/2025/

Daniel J. Milroy, Allison H. Baker, Dorit M. Hammerling, and Elizabeth R. Jessup, “Nine time steps: ultra-fast statistical consistency testing of the Community Earth System Model (pyCECT v3.0)”, Geoscientific Model Development, 11, pp. 697-711, 2018.
https://gmd.copernicus.org/articles/11/697/2018/

A.H. Baker, Y. Hu, D.M. Hammerling, Y. Tseng, X. Hu, X. Huang, F.O. Bryan, and G. Yang, “Evaluating Statistical Consistency in the Ocean Model Component of the Community Earth System Model (pyCECT v2.0).” Geoscientific Model Development, 9, pp. 2391-2406, 2016.
https://gmd.copernicus.org/articles/9/2391/2016/

A.H. Baker, D.M. Hammerling, M.N. Levy, H. Xu, J.M. Dennis, B.E. Eaton, J. Edwards, C. Hannay, S.A. Mickelson, R.B. Neale, D. Nychka, J. Shollenberger, J. Tribbia, M. Vertenstein, and D. Williamson, “A new ensemble-based consistency test for the community earth system model (pyCECT v1.0).” Geoscientific Model Development, 8, pp. 2829–2840, 2015.
https://gmd.copernicus.org/articles/8/2829/2015/
