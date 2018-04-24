# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Omer Faruk Gulban & Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# *****************************************************************************
# *** Import modules & adjust cython settings for speedup

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************

# *****************************************************************************
# *** Main function for least squares solution, with cross validation

cpdef np.ndarray[np.float32_t, ndim=2] cy_lst_sq_xval(
    np.ndarray[np.float32_t, ndim=1] vecPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk,
    np.ndarray[np.int32_t, ndim=2] aryIdxTrn,
    np.ndarray[np.int32_t, ndim=2] aryIdxTst
    ):
    """
    Cythonised least squares GLM model fitting with cross validation.

    Parameters
    ----------
    vecPrfTc : np.array
        1D numpy array, at float32 precision, containing a single pRF model
        time course (along time dimension).
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].
    aryIdxTrn : np.array
        2D numpy array, at int32 precision, containing a trainings indices for
        cross-validation.
    aryIdxTst : np.array
        2D numpy array, at int32 precision, containing a test indices for
        cross-validation.

    Returns
    -------
    aryResXval : np.array
        2D numpy array with cross validation error for all voxels in the chunk of
        functional data and all cross validation folds.
        Dimensionality: aryResXval[voxel, varNumXval]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses with k-fold cross validation.
    Assumes removal of the mean from the functional data and the model.
    Needs to be compiled before execution (see `cython_leastsquares_setup.py`).
    """

    cdef float[:] vecPrfTc_view = vecPrfTc
    cdef float [:, :] aryFuncChnk_view = aryFuncChnk
    cdef int [:, :] aryIdxTrn_view = aryIdxTrn
    cdef int [:, :] aryIdxTst_view = aryIdxTst

    cdef unsigned long varNumVoxChnk, idxVox
    cdef unsigned int idxVol, idxXval, varNumXval
    cdef int[:] vecIdxTrn

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])
    # Number of cross-validations:
    varNumXval = int(aryIdxTrn.shape[-1])

    # Define 2D array for residuals (here crossvalidation error) of least
    # squares solution), initialized with all zeros here:
    cdef np.ndarray[np.float32_t, ndim=2] aryResXval = np.zeros((varNumVoxChnk,
                                                                 varNumXval),
                                                                dtype=np.float32)
    # Memory view on array for residuals (here crossvalidation error)
    cdef float[:, :] aryResXval_view = aryResXval

    # Define 1D array for variances in training model time courses across folds,
    # initialized with all zeros here
    cdef np.ndarray[np.float32_t, ndim=1] vecVarY = np.zeros(varNumXval,
                                                             dtype=np.float32)
    # Memory view on array for variances in training model time courses:
    cdef float[:] vecVarY_view = vecVarY

    # Calculate variance of training pRF model time course (i.e. variance in
    # the model) - separately for every fold:
    for idxXval in range(varNumXval):
        # get vector with volumes for training
        vecIdxTrn = aryIdxTrn_view[:, idxXval]
        for idxVol in vecIdxTrn:
            vecVarY_view[idxXval] += vecPrfTc_view[idxVol] ** 2

    # Call optimised cdef function for calculation of residuals:
    aryResXval_view = func_cy_res_xval(vecPrfTc_view,
                                       aryFuncChnk_view,
                                       aryIdxTrn_view,
                                       aryIdxTst_view,
                                       aryResXval_view,
                                       varNumXval,
                                       varNumVoxChnk,
                                       vecVarY_view)

    # Convert memory view to numpy array before returning it:
    aryResXval = np.asarray(aryResXval_view)

    return aryResXval

# *****************************************************************************

# *****************************************************************************
# *** Function for fast calculation of residuals, with cross-validation

cdef float[:, :] func_cy_res_xval(float[:] vecPrfTc_view,
                                  float[:, :] aryFuncChnk_view,
                                  int[:, :] aryIdxTrn_view,
                                  int[:, :] aryIdxTst_view,
                                  float[:, :] aryResXval_view,
                                  unsigned int varNumXval,
                                  unsigned long varNumVoxChnk,
                                  float[:] vecVarY_view):

    cdef float varVarY, varCovXy, varRes, varSlope, varXhat
    cdef unsigned int idxVol, idxXval
    cdef unsigned long idxVox
    cdef int[:] vecIdxTrn_view, vecIdxTst_view

    # Loop through cross-validations
    for idxXval in range(varNumXval):

        # get vector with current training and test volumes
        vecIdxTrn_view = aryIdxTrn_view[:, idxXval]
        vecIdxTst_view = aryIdxTst_view[:, idxXval]

        # Loop through voxels:
        for idxVox in range(varNumVoxChnk):

            # Covariance and residuals of current voxel:
            varCovXy = 0
            varRes = 0

            # Loop through trainings volumes and calculate covariance between
            # the training model and the current voxel:
            for idxVol in vecIdxTrn_view:
                varCovXy += (aryFuncChnk_view[idxVol, idxVox]
                             * vecPrfTc_view[idxVol])
            # get the variance in trainings model time courses for this fold
            varVarY = vecVarY_view[idxXval]
            # Obtain the slope of the regression of the model on the data:
            varSlope = varCovXy / varVarY

            # calculate model prediction time course
            for idxVol in vecIdxTst_view:
                # The predicted voxel time course value:
                varXhat = vecPrfTc_view[idxVol] * varSlope
                # Mismatch between prediction and actual voxel value (variance):
                varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

            aryResXval_view[idxVox, idxXval] = varRes

    # Return memory view
    return aryResXval_view
# *****************************************************************************
