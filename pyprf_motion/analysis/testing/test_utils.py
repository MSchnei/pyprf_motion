"""Test utility functions."""

import os
from os.path import isfile, join
import numpy as np
from pyprf_motion.analysis.load_config import load_config
from pyprf_motion.analysis.model_creation_main import model_creation
from pyprf_motion.analysis import pyprf_main
from pyprf_motion.analysis import utils_general as util
from pyprf_motion.analysis.cython_leastsquares_setup_call import setup_cython

# Compile cython code:
setup_cython()

# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))

# Decimal places to round before comparing template and test results:
varRnd = 3


def test_model_creation():

    # --------------------------------------------------------------------------
    # *** Test model creation, cartesian coordinates

    # Load template model:
    mdlTmplCrt = np.load(
            strDir + '/pRF_tmpl_model_tc_crt.npz')
    mdlTmplCrt = mdlTmplCrt['model']

    # Round template results:
    mdlTmplCrt = np.around(mdlTmplCrt.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgCrtMdl = (strDir + '/config_testing_crt_model.csv')

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfgCrtMdl, lgcTest=True)

    # Call function for model creation:
    model_creation(dicCnfg)

    # Load test result:
    mdlTestCrt = np.load(
            strDir + '/result/' + 'pRF_test_model_tc_crt.npy')

    # Round test results:
    mdlTestCrt = np.around(mdlTestCrt.astype(np.float32), decimals=varRnd)

    # Test whether the template and test models correspond:
    lgcTestTmplCrt = np.array_equal(mdlTmplCrt, mdlTestCrt)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test model creation, polar coordinates

    # Load template model:
    mdlTmplPol = np.load(
            strDir + '/pRF_tmpl_model_tc_pol.npz')
    mdlTmplPol = mdlTmplPol['model']

    # Round template results:
    mdlTmplPol = np.around(mdlTmplPol.astype(np.float32), decimals=varRnd)

    # Path of config file for tests:
    strCsvCnfgPolMdl = (strDir + '/config_testing_pol_model.csv')

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfgPolMdl, lgcTest=True)

    # Call function for model creation:
    model_creation(dicCnfg)

    # Load test result:
    mdlTestPol = np.load(
            strDir + '/result/' + 'pRF_test_model_tc_pol.npy')

    # Round test results:
    mdlTestPol = np.around(mdlTestPol.astype(np.float32), decimals=varRnd)

    # Test whether the template and test models correspond:
    lgcTestTmplPol = np.array_equal(mdlTmplPol, mdlTestPol)
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
    assert (lgcTestTmplCrt and
            lgcTestTmplPol)


def test_model_fitting():
    """Run main pyprf_motion function and compare results with template."""
    # --------------------------------------------------------------------------
    # *** Put template models in results folder for use in model fitting

    # Load template model for crt coordinates:
    mdlTmplCrt = np.load(
            strDir + '/pRF_tmpl_model_tc_crt.npz')
    mdlTmplCrt = mdlTmplCrt['model']

    # save template model in results folder as test model
    np.save(strDir + '/result/' + '/pRF_test_model_tc_crt.npz', mdlTmplCrt)

    # Load template model for pol coordinates:
    mdlTmplPol = np.load(
            strDir + '/pRF_tmpl_model_tc_pol.npz')
    mdlTmplPol = mdlTmplPol['model']

    # save template model in results folder as test model
    np.save(strDir + '/result/' + '/pRF_test_model_tc_pol.npz', mdlTmplPol)

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
