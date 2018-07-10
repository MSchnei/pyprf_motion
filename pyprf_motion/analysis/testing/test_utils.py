"""Test utility functions."""

import os
from os.path import isfile, join
import numpy as np
from pyprf_motion.analysis import pyprf_main
from pyprf_motion.analysis import utils_general as util
from pyprf_motion.analysis.cython_leastsquares_setup_call import setup_cython

# Compile cython code:
setup_cython()

# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))

# Decimal places to round before comparing template and test results:
varRnd = 3


def test_main():
    """Run main pyprf_motion function and compare results with template."""

    # --------------------------------------------------------------------------
    # *** Test numpy version, no cross-validation, cartesian coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_np_noxval_crt_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy_noxval_crt.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_np_noxval_crt_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestNpNoxvalCrt = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test numpy version, with cross-validation, cartesian coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_np_xval_crt_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy_xval_crt.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_np_xval_crt_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestNpXvalCrt = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test numpy version, no cross-validation, polar coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_np_noxval_pol_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy_noxval_pol.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_np_noxval_pol_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestNpNoxvalPol = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test numpy version, with cross-validation, polar coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_np_xval_pol_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy_xval_pol.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_np_xval_pol_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestNpXvalPol = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version, no cross-validation, cartesian coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_cy_noxval_crt_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython_noxval_crt.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_cy_noxval_crt_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestCyNoxvalCrt = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version, with cross-validation, cartesian coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_cy_xval_crt_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython_xval_crt.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_cy_xval_crt_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestCyXvalCrt = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version, no cross-validation, polar coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_cy_noxval_pol_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython_noxval_pol.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_cy_noxval_pol_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestCyNoxvalPol = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version, with cross-validation, polar coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_cy_xval_pol_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython_xval_pol.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_cy_xval_pol_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestCyXvalPol = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test tensorflow version, no cross-validation, cartesian coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_tf_noxval_crt_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgTf = (strDir + '/config_testing_tensorflow_noxval_crt.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgTf, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_tf_noxval_crt_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestTfNoxvalCrt = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test tensorflow version, no cross-validation, polar coordinates

    # Load template result:
    aryTmplR2, _, _ = util.load_nii(
            strDir + '/pRF_tmpl_results_tf_noxval_pol_R2.nii.gz')

    # Round template results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgTf = (strDir + '/config_testing_tensorflow_noxval_pol.csv')

    # Call main pyprf_motion function:
    pyprf_main.pyprf(strCsvCnfgTf, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = util.load_nii(
            strDir + '/result/' + 'pRF_test_results_tf_noxval_pol_R2.nii.gz')

    # Round test results:
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestTfNoXvalPol = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Clean up

    # Path of directory with results:
    strDirRes = strDir + '/result/'

    # Get list of files in results directory:
    lstFls = [f for f in os.listdir(strDirRes) if isfile(join(strDirRes, f))]

    # Delete results of test:
    for strTmp in lstFls:
        if '.nii' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))
        elif '.npy' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))
    # --------------------------------------------------------------------------

    assert (lgcTestNpNoxvalCrt and
            lgcTestNpXvalCrt and
            lgcTestNpNoxvalPol and
            lgcTestNpXvalPol and
            lgcTestCyNoxvalCrt and
            lgcTestCyXvalCrt and
            lgcTestCyNoxvalPol and
            lgcTestCyXvalPol and
            lgcTestTfNoxvalCrt and
            lgcTestTfNoXvalPol)
