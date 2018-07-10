# -*- coding: utf-8 -*-
"""Main function for pRF finding using CPU."""

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

import numpy as np
from sklearn.model_selection import KFold
from pyprf_motion.analysis.cython_leastsquares import cy_lst_sq, cy_lst_sq_xval


def find_prf_cpu(idxPrc, aryFuncChnk, aryPrfTc, aryMdlParams, strVersion,
                 lgcXval, varNumXval, queOut):
    """
    Find best fitting pRF model for voxel time course, using the CPU.

    Parameters
    ----------
    idxPrc : int
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryFunc : np.array
        2D array with functional MRI data, with shape aryFunc[voxel, time].
    aryPrfTc : np.array
        Array with pRF model time courses, with shape
        aryPrfTc[x-pos*y-pos*SD, number of volumes]
    aryMdlParams : np.array
        2D array with all pRF model parameter combinations.
    strVersion : str
        Which version to use for pRF finding; 'numpy' or 'cython'.
    lgcXval: boolean
        Logical to determine whether we cross-validate.
    varNumXval: int
        Number of folds for k-fold cross-validation.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------
    lstOut : list
        List containing the following objects:
        idxPrc : int
            Process ID of the process calling this function (for CPU
            multi-threading). In GPU version, this parameter is 0.
        vecBstXpos : np.array
            1D array with best fitting x-position for each voxel, with shape
            vecBstXpos[voxel].
        vecBstYpos : np.array
            1D array with best fitting y-position for each voxel, with shape
            vecBstYpos[voxel].
        vecBstSd : np.array
            1D array with best fitting pRF size for each voxel, with shape
            vecBstSd[voxel].
        vecBstR2 : np.array
            1D array with R2 value of 'winning' pRF model for each voxel, with
            shape vecBstR2[voxel].

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue. This version performs the model finding on the CPU,
    using numpy or cython (depending on the value of `strVersion`).
    """

    # Number of models in the visual space:
    varNumMdls = aryPrfTc.shape[0]

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Vectors for pRF finding results [number-of-voxels times one]:
    # make sure they have the same precision as aryMdlParams, since this
    # is important for later comparison
    vecBstXpos = np.zeros(varNumVoxChnk, dtype=aryMdlParams.dtype)
    vecBstYpos = np.zeros(varNumVoxChnk, dtype=aryMdlParams.dtype)
    vecBstSd = np.zeros(varNumVoxChnk, dtype=aryMdlParams.dtype)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk), np.inf).astype(np.float32)

    # In case we cross-validate we also save and replace the best
    # residual values for every fold (not only mean across folds):
    if lgcXval:
        aryBstResFlds = np.zeros((varNumVoxChnk, varNumXval), dtype=np.float32)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # if lgc for Xval is true we already prepare indices for xvalidation
    if lgcXval:
        # obtain iterator for cross-validation
        itXval = KFold(n_splits=varNumXval)
        vecSplts = np.arange(aryPrfTc.shape[-1], dtype=np.int32)

        # prepare lists that will hold indices for xvalidation
        lstIdxTrn = []
        lstIdxtst = []
        # Loop over the cross-validations to put indcies in array
        for idxTrn, idxTst in itXval.split(vecSplts):
            lstIdxTrn.append(idxTrn)
            lstIdxtst.append(idxTst)
        # trun lists into array
        aryIdxTrn = np.stack(lstIdxTrn, axis=-1).astype(np.int32)
        aryIdxTst = np.stack(lstIdxtst, axis=-1).astype(np.int32)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrf = np.ceil(vecStatPrf)
        vecStatPrf = vecStatPrf.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # There can be pRF model time courses with a variance of zero (i.e. pRF
    # models that are not actually responsive to the stimuli). For time
    # efficiency, and in order to avoid division by zero, we ignore these
    # model time courses.
    aryPrfTcVar = np.var(aryPrfTc, axis=1)

    # Zero with float32 precision for comparison:
    varZero32 = np.array(([0.0])).astype(np.float32)[0]

    # Loop through pRF models:
    for idxMdl in range(0, varNumMdls):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Status indicator:
            if varCntSts02 == vecStatPrf[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('------------Progress: ' +
                             str(vecStatPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatPrf[varCntSts01]) +
                             ' pRF models out of ' +
                             str(varNumMdls))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # Only fit pRF model if variance is not zero:
        if np.greater(aryPrfTcVar[idxMdl], varZero32):

            # Check whether we need to crossvalidate
            if lgcXval:
                # We do crossvalidate. In this case, we loop through
                # the different folds of the crossvalidation and
                # calculate the cross-validation error for the current
                # model for all voxel time courses.

                # Cython version:
                if strVersion == 'cython':

                    vecMdl = aryPrfTc[idxMdl, :]

                    # A cython function is used to loop over the folds
                    # of the cross-validation, and within each fold to
                    # calculate the parameter estimates of the current
                    # model and the crossvalidation error:
                    aryResXval = cy_lst_sq_xval(vecMdl,
                                                aryFuncChnk,
                                                aryIdxTrn,
                                                aryIdxTst)

                # Numpy version:
                elif strVersion == 'numpy':

                    # pre-allocate ary to collect cross-validation
                    # error for every xval fold
                    aryResXval = np.empty((varNumVoxChnk,
                                           varNumXval),
                                          dtype=np.float32)

                    # loop over cross-validation folds
                    for idxXval in range(varNumXval):
                        # Get pRF time course models for trn and tst:
                        vecMdlTrn = aryPrfTc[idxMdl,
                                             aryIdxTrn[:, idxXval]]
                        vecMdlTst = aryPrfTc[idxMdl,
                                             aryIdxTst[:, idxXval]]
                        # Get functional data for trn and tst:
                        aryFuncChnkTrn = aryFuncChnk[
                            aryIdxTrn[:, idxXval], :]
                        aryFuncChnkTst = aryFuncChnk[
                            aryIdxTst[:, idxXval], :]

                        # Reshape pRF time course model so it has the
                        # required shape for np.linalg.lstsq
                        vecMdlTrn = np.reshape(vecMdlTrn, (-1, 1))

                        # Numpy linalg.lstsq is used to calculate the
                        # parameter estimates of the current model:
                        vecTmpPe = np.linalg.lstsq(vecMdlTrn,
                                                   aryFuncChnkTrn,
                                                   rcond=-1)[0]

                        # calculate model prediction time course
                        aryMdlPrdTc = np.dot(
                            np.reshape(vecMdlTst, (-1, 1)),
                            np.reshape(vecTmpPe, (1, -1)))

                        # calculate residual sum of squares between
                        # test data and model prediction time course
                        aryResXval[:, idxXval] = np.sum(
                            (np.subtract(aryFuncChnkTst,
                                         aryMdlPrdTc))**2, axis=0)

                # calculate the average cross validation error across
                # all folds
                vecTmpRes = np.mean(aryResXval, axis=1)

            else:
                # We do not crossvalidate. In this case, we calculate
                # the ratio of the explained variance (R squared)
                # for the current model for all voxel time courses.

                # Cython version:
                if strVersion == 'cython':

                    # A cython function is used to calculate the
                    # residuals of the current model:
                    vecTmpRes = cy_lst_sq(
                        aryPrfTc[idxMdl, :],
                        aryFuncChnk)

                # Numpy version:
                elif strVersion == 'numpy':

                    # Reshape pRF time course model so it has the
                    # required shape for np.linalg.lstsq
                    vecDsgn = np.reshape(
                        aryPrfTc[idxMdl, :], (-1, 1))

                    # Numpy linalg.lstsq is used to calculate the
                    # residuals of the current model:
                    vecTmpRes = np.linalg.lstsq(vecDsgn, aryFuncChnk,
                                                rcond=-1)[1]

            # Check whether current crossvalidation error (xval=True)
            # or residuals (xval=False) are lower than previously
            # calculated ones:
            vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

            # Replace best x and y position values, and SD values.
            vecBstXpos[vecLgcTmpRes] = aryMdlParams[idxMdl, 0]
            vecBstYpos[vecLgcTmpRes] = aryMdlParams[idxMdl, 1]
            vecBstSd[vecLgcTmpRes] = aryMdlParams[idxMdl, 2]

            # Replace best mean residual values:
            vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]
            # In case we cross-validate we also save and replace the best
            # residual values for every fold (not only mean across folds):
            if lgcXval:
                aryBstResFlds[vecLgcTmpRes, :] = aryResXval[vecLgcTmpRes, :]

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # After finding the best fitting model for each voxel, we still have to
    # calculate the average correlation coefficient between predicted and
    # measured time course (xval=True) or the coefficient of determination
    # (xval=False) for each voxel.

    if lgcXval:

        # create vector that allows to check whether every voxel is visited
        # exactly once
        vecVxlTst = np.zeros(aryFuncChnk.shape[1])

        # Since we did not do this during finding the best model, we still need
        # to calculate deviation from a mean model for every voxel and fold
        # arySsTotXval

        # concatenate vectors with best x, y, sigma params
        aryBstPrm = np.stack((vecBstXpos, vecBstYpos, vecBstSd), axis=1)
        # find unique rows
        vecUnqIdx = np.ascontiguousarray(aryBstPrm).view(
            np.dtype((np.void, aryBstPrm.dtype.itemsize * aryBstPrm.shape[1])))
        _, vecUnqIdx = np.unique(vecUnqIdx, return_index=True)
        # get rows with all best-fitting model parameter combinations found
        aryUnqRows = aryBstPrm[vecUnqIdx]

        # calculate deviation from a mean model for every voxel and fold
        arySsTotXval = np.zeros((aryBstResFlds.shape),
                                dtype=aryBstResFlds.dtype)

        # loop over all best-fitting model parameter combinations found
        for vecPrm in aryUnqRows:
            # get logical for voxels for which this prm combi was the best
            lgcVxl = np.isclose(aryBstPrm, vecPrm, atol=1e-04).all(axis=1)
            if np.all(np.invert(lgcVxl)):
                print('------------No voxel found, process ' + str(idxPrc))
            # mark those voxels that were visited
            vecVxlTst[lgcVxl] += 1

            # get voxel time course
            aryVxlTc = aryFuncChnk[:, lgcVxl]
            # loop over cross-validation folds
            for idxXval in range(varNumXval):
                # Get functional data for tst:
                aryFuncChnkTst = aryVxlTc[
                    aryIdxTst[:, idxXval], :]
                # Deviation from the mean for each datapoint:
                aryFuncDev = np.subtract(aryFuncChnkTst,
                                         np.mean(aryFuncChnkTst,
                                                 axis=0)[None, :])
                # Sum of squares:
                vecSsTot = np.sum(np.power(aryFuncDev,
                                           2.0),
                                  axis=0)
                arySsTotXval[lgcVxl, idxXval] = vecSsTot

        # check that every voxel was visited exactly once
        errMsg = 'At least one voxel visited more than once for SStot calc'
        assert len(vecVxlTst) == np.sum(vecVxlTst), errMsg

        # Calculate coefficient of determination by comparing:
        # aryBstResFlds vs. arySsTotXval

        # get logical to check that arySsTotXval is greater than zero in all
        # voxels and folds
        lgcExclZeros = np.all(np.greater(arySsTotXval,  np.array([0.0])),
                              axis=1)
        print('------------Nr of voxels: ' + str(len(lgcExclZeros)))
        print('------------Nr of voxels avove 0: ' + str(np.sum(lgcExclZeros)))

        # Calculate R2 for every crossvalidation fold seperately
        aryBstR2fld = np.subtract(
            1.0, np.divide(aryBstResFlds,
                           arySsTotXval))

        # Calculate mean R2 across folds here
        vecBstR2 = np.subtract(
            1.0, np.mean(np.divide(aryBstResFlds,
                                   arySsTotXval),
                         axis=1))

        # Output list:
        lstOut = [idxPrc,
                  vecBstXpos,
                  vecBstYpos,
                  vecBstSd,
                  vecBstR2,
                  aryBstR2fld]

        queOut.put(lstOut)

    else:
        # To calculate the coefficient of determination, we start with the
        # total sum of squares (i.e. the deviation of the data from the mean).
        # The mean of each time course:
        vecFuncMean = np.mean(aryFuncChnk, axis=0)
        # Deviation from the mean for each datapoint:
        aryFuncDev = np.subtract(aryFuncChnk, vecFuncMean[None, :])
        # Sum of squares:
        vecSsTot = np.sum(np.power(aryFuncDev,
                                   2.0),
                          axis=0)
        # Coefficient of determination:
        vecBstR2 = np.subtract(1.0,
                               np.divide(vecBstRes,
                                         vecSsTot))

        # Output list:
        lstOut = [idxPrc,
                  vecBstXpos,
                  vecBstYpos,
                  vecBstSd,
                  vecBstR2]

        queOut.put(lstOut)
