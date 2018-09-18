# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields."""

# Part of pyprf_motion library
# Copyright (C) 2018  Marian Schneider, Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import time
import numpy as np
import multiprocessing as mp
from pyprf_motion.analysis.utils_general import export_nii, joinRes
from pyprf_motion.analysis.model_creation_main import model_creation
from pyprf_motion.analysis.model_creation_utils import crt_mdl_prms
from pyprf_motion.analysis.prepare import prep_models, prep_func


def pyprf(cfg, lgcTest=False, strExpSve=''):
    """
    Main function for pRF mapping.

    Parameters
    ----------
    cfg : namespace
        Namespace containing variables from config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.
    strExpSve : str
        String to add to path for export of results in nii. This is used to
        differentiate different results from different exponents.
    """

    # %% Check time
    print('---pRF analysis')
    varTme01 = time.time()

    # %% Preparations

    # Conditional imports:
    if cfg.strVersion == 'gpu':
        from pyprf_motion.analysis.find_prf_gpu import find_prf_gpu
    if ((cfg.strVersion == 'cython') or (cfg.strVersion == 'numpy')):
        from pyprf_motion.analysis.find_prf_cpu import find_prf_cpu

    # Convert preprocessing parameters (for temporal smoothing)
    # from SI units (i.e. [s]) into units of data array (volumes):
    cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)

    # Create or load pRF time course models
    aryPrfTc = model_creation(cfg)

    # %% Preprocessing

    # The model time courses will be preprocessed such that they are smoothed
    # (temporally) with same factor as the data and that they will be z-scored:
    aryPrfTc = prep_models(aryPrfTc, varSdSmthTmp=cfg.varSdSmthTmp)

    # The functional data will be masked and demeaned:
    aryLgcMsk, aryLgcVar, hdrMsk, aryAff, aryFunc, tplNiiShp = prep_func(
        cfg.strPathNiiMask, cfg.lstPathNiiFunc)

    # set the precision of the header to np.float32 so that the prf results
    # will be saved in this precision later
    hdrMsk.set_data_dtype(np.float32)

    # %% Find pRF models for voxel time courses

    print('------Find pRF models for voxel time courses')

    # Number of voxels for which pRF finding will be performed:
    varNumVoxInc = aryFunc.shape[0]

    print('---------Number of voxels on which pRF finding will be performed: '
          + str(varNumVoxInc))

    print('---------Preparing parallel pRF model finding')

    # For the GPU version, we need to set down the parallelisation to 1 now,
    # because no separate CPU threads are to be created. We may still use CPU
    # parallelisation for preprocessing, which is why the parallelisation
    # factor is only reduced now, not earlier.
    if cfg.strVersion == 'gpu':
        cfg.varPar = 1

    # Make sure that if gpu fitting is used, the number of cross-validations is
    # set to 1, not higher
    if cfg.strVersion == 'gpu':
        strErrMsg = 'Stopping program. ' + \
            'Cross-validation on GPU is currently not supported. ' + \
            'Set varNumXval equal to 1 in csv file in order to continue. '
        assert cfg.varNumXval == 1, strErrMsg

    # Get array with all possible model parameter combination:
    # [x positions, y positions, sigmas]
    aryMdlParams = crt_mdl_prms((int(cfg.varVslSpcSzeX),
                                 int(cfg.varVslSpcSzeY)), cfg.varNum1,
                                cfg.varExtXmin, cfg.varExtXmax, cfg.varNum2,
                                cfg.varExtYmin, cfg.varExtYmax,
                                cfg.varNumPrfSizes, cfg.varPrfStdMin,
                                cfg.varPrfStdMax, cfg.lstExp, kwUnt='deg',
                                kwCrd=cfg.strKwCrd)

    # Empty list for results (parameters of best fitting pRF model):
    lstPrfRes = [None] * cfg.varPar

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Change type of functional data to float 32:
    aryFunc = aryFunc.astype(np.float32)
    # Create list with chunks of functional data for the parallel processes:
    lstFunc = np.array_split(aryFunc, cfg.varPar)
    # We don't need the original array with the functional data anymore:
    del(aryFunc)

    # check whether we need to crossvalidate
    if np.greater(cfg.varNumXval, 1):
        cfg.lgcXval = True
    elif np.equal(cfg.varNumXval, 1):
        cfg.lgcXval = False
    else:
        print("Please set number of crossvalidation folds (numXval) to 1 \
              (meaning no cross validation) or greater than 1 (meaning number \
              of cross validation folds)")

    # CPU version (using numpy or cython for pRF finding):
    if ((cfg.strVersion == 'numpy') or (cfg.strVersion == 'cython')):

        print('---------pRF finding on CPU')

        print('---------Creating parallel processes')

        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=find_prf_cpu,
                                         args=(idxPrc,
                                               lstFunc[idxPrc].T,
                                               aryPrfTc,
                                               aryMdlParams,
                                               cfg.strVersion,
                                               cfg.lgcXval,
                                               cfg.varNumXval,
                                               queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    # GPU version (using tensorflow for pRF finding):
    elif cfg.strVersion == 'gpu':

        print('---------pRF finding on GPU')

        # Create processes:
        for idxPrc in range(0, cfg.varPar):
            lstPrcs[idxPrc] = mp.Process(target=find_prf_gpu,
                                         args=(idxPrc,
                                               aryMdlParams,
                                               lstFunc[idxPrc].T,
                                               aryPrfTc,
                                               queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Delete reference to list with function data (the data continues to exists
    # in child process):
    del(lstFunc)

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstPrfRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    # %% Sort and prepare output from model fitting
    print('---------Prepare pRF finding results for export')

    # Put output into correct order:
    lstPrfRes = sorted(lstPrfRes)

    # collect results from parallelization
    aryBstXpos = joinRes(lstPrfRes, cfg.varPar, 1, inFormat='1D')
    aryBstYpos = joinRes(lstPrfRes, cfg.varPar, 2, inFormat='1D')
    aryBstSd = joinRes(lstPrfRes, cfg.varPar, 3, inFormat='1D')
    aryBstExp = joinRes(lstPrfRes, cfg.varPar, 4, inFormat='1D')
    aryBstR2 = joinRes(lstPrfRes, cfg.varPar, 5, inFormat='1D')
    if np.greater(cfg.varNumXval, 1):
        aryBstR2Single = joinRes(lstPrfRes, cfg.varPar, 6, inFormat='2D')

    # %% Calculate polar angle and eccentricity

    # Calculate polar angle map:
    aryPlrAng = np.arctan2(aryBstYpos, aryBstXpos)
    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryEcc = np.sqrt(np.add(np.square(aryBstXpos),
                            np.square(aryBstYpos)))

    # %% Export each map of best parameters as a 3D nii file

    print('---------Exporting results')

    # Xoncatenate all the best voxel maps
    aryBstMaps = np.stack([aryBstXpos, aryBstYpos, aryBstSd, aryBstExp,
                           aryBstR2, aryPlrAng, aryEcc], axis=1)

    # List with name suffices of output images:
    lstNiiNames = ['_x_pos',
                   '_y_pos',
                   '_SD',
                   '_exp',
                   '_R2',
                   '_polar_angle',
                   '_eccentricity']

    # Create full path names from nii file names and output path
    lstNiiNames = [cfg.strPathOut + strNii + strExpSve + '.nii.gz' for strNii
                   in lstNiiNames]

    # export map results as seperate 3D nii files
    export_nii(aryBstMaps, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp,
               aryAff, hdrMsk, outFormat='3D')

    # %% Save R2 maps from crossvalidation (saved for every run) as nii:

    if np.greater(cfg.varNumXval, 1):

        # truncate extremely negative R2 values
        aryBstR2Single[np.where(np.less_equal(aryBstR2Single, -1.0))] = -1.0

        # List with name suffices of output images:
        lstNiiNames = ['_R2_single']

        # Create full path names from nii file names and output path
        lstNiiNames = [cfg.strPathOut + strNii + strExpSve + '.nii.gz' for
                       strNii in lstNiiNames]

        # export R2 maps as a single 4D nii file
        export_nii(aryBstR2Single, lstNiiNames, aryLgcMsk, aryLgcVar,
                   tplNiiShp, aryAff, hdrMsk, outFormat='4D')

    # %% Report time

    varTme02 = time.time()
    varTme03 = varTme02 - varTme01
    print('---Elapsed time: ' + str(varTme03) + ' s')
    print('---Done.')
