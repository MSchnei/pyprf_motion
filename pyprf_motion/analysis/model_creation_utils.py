# -*- coding: utf-8 -*-
"""Utilities for pRF model creation."""

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
import multiprocessing as mp
from pyprf_motion.analysis.utils_hrf import spmt, dspmt, ddspmt, cnvl_tc
from pyprf_motion.analysis.utils_general import (rmp_rng, map_pol_to_crt,
                                                 cnvl_2D_gauss)
from pyprf_motion.analysis.utils_hrf import create_boxcar


def rmp_deg_pixel_x_y_s(vecX, vecY, vecPrfSd, tplPngSize,
                        varExtXmin, varExtXmax, varExtYmin, varExtYmax):
    """Remap x, y, sigma parameters from degrees to pixel.

    Parameters
    ----------
    vecX : 1D numpy array
        Array with possible x parametrs in degree
    vecY : 1D numpy array
        Array with possible y parametrs in degree
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in degree
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space in pixel (width, height).
    varExtXmin : float
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float
        Extent of visual space from centre in positive x-direction (width)
    varExtYmin : int
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float
        Extent of visual space from centre in positive y-direction (height)
    Returns
    -------
    vecX : 1D numpy array
        Array with possible x parametrs in pixel
    vecY : 1D numpy array
        Array with possible y parametrs in pixel
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in pixel
    """
    # Remap modelled x-positions of the pRFs:
    vecXpxl = rmp_rng(vecX, 0.0, (tplPngSize[0] - 1), varOldThrMin=varExtXmin,
                      varOldAbsMax=varExtXmax)

    # Remap modelled y-positions of the pRFs:
    vecYpxl = rmp_rng(vecY, 0.0, (tplPngSize[1] - 1), varOldThrMin=varExtYmin,
                      varOldAbsMax=varExtYmax)

    # We calculate the scaling factor from degrees of visual angle to
    # pixels separately for the x- and the y-directions (the two should
    # be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in ' + \
        'stimulus space (in degrees of visual angle) and the ' + \
        'ratio of X and Y dimensions in the upsampled visual space' + \
        'do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Convert prf sizes from degrees of visual angles to pixel
    vecPrfSdpxl = np.multiply(vecPrfSd, varDgr2PixX)

    return vecXpxl, vecYpxl, vecPrfSdpxl


def crt_mdl_prms(tplPngSize, varNum1, varExtXmin,  varExtXmax, varNum2,
                 varExtYmin, varExtYmax, varNumPrfSizes, varPrfStdMin,
                 varPrfStdMax, kwUnt='pix', kwCrd='crt'):
    """Create an array with all possible model parameter combinations

    Parameters
    ----------
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space (width, height).
    varNum1 : int, positive
        Number of x-positions to model
    varExtXmin : float
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float
        Extent of visual space from centre in positive x-direction (width)
    varNum2 : float, positive
        Number of y-positions to model.
    varExtYmin : int
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float
        Extent of visual space from centre in positive y-direction (height)
    varNumPrfSizes : int, positive
        Number of pRF sizes to model.
    varPrfStdMin : float, positive
        Minimum pRF model size (standard deviation of 2D Gaussian)
    varPrfStdMax : float, positive
        Maximum pRF model size (standard deviation of 2D Gaussian)
    kwUnt: str
        Keyword to set the unit for model parameter combinations; model
        parameters can be in pixels ["pix"] or degrees of visual angles ["deg"]
    kwCrd: str
        Keyword to set the coordinate system for model parameter combinations;
        parameters can be in cartesian ["crt"] or polar ["pol"] coordinates
    Returns
    -------
    aryMdlParams : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, 3]
        Model parameters (x, y, sigma) for all models.
    Reference
    ---------
    [1]
    """

    # Number of pRF models to be created (i.e. number of possible
    # combinations of x-position, y-position, and standard deviation):
    varNumMdls = varNum1 * varNum2 * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for
    # which pRF model time courses are going to be created, where the
    # columns correspond to: (1) the x-position, (2) the y-position, and
    # (3) the standard deviation. The parameters are in units of the
    # upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 3), dtype=np.float32)

    # Counter for parameter array:
    varCntMdlPrms = 0

    if kwCrd == 'crt':

        # Vector with the moddeled x-positions of the pRFs:
        vecX = np.linspace(varExtXmin, varExtXmax, varNum1, endpoint=True)

        # Vector with the moddeled y-positions of the pRFs:
        vecY = np.linspace(varExtYmin, varExtYmax, varNum2, endpoint=True)

        # Vector with standard deviations pRF models (in degree of vis angle):
        vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                               endpoint=True)

        if kwUnt == 'deg':
            # since parameters are already in degrees of visual angle,
            # we do nothing
            pass

        elif kwUnt == 'pix':
            # convert parameters to pixels
            vecX, vecY, vecPrfSd = rmp_deg_pixel_x_y_s(vecX, vecY, vecPrfSd,
                                                       tplPngSize, varExtXmin,
                                                       varExtXmax, varExtYmin,
                                                       varExtYmax)

        else:
            print('Unknown keyword provided for possible model parameter ' +
                  'combinations: should be either pix or deg')

        # Put all combinations of x-position, y-position, and standard
        # deviations into the array:

        # Loop through x-positions:
        for idxX in range(0, varNum1):

            # Loop through y-positions:
            for idxY in range(0, varNum2):

                # Loop through standard deviations (of Gaussian pRF models):
                for idxSd in range(0, varNumPrfSizes):

                    # Place index and parameters in array:
                    aryMdlParams[varCntMdlPrms, 0] = vecX[idxX]
                    aryMdlParams[varCntMdlPrms, 1] = vecY[idxY]
                    aryMdlParams[varCntMdlPrms, 2] = vecPrfSd[idxSd]

                    # Increment parameter index:
                    varCntMdlPrms += 1

    elif kwCrd == 'pol':

        # Vector with the radial position:
        vecRad = np.linspace(0.0, varExtXmax, varNum1, endpoint=True)

        # Vector with the angular position:
        vecTht = np.linspace(0.0, 2*np.pi, varNum2, endpoint=False)

        # get all possible combinations on the grid, using matrix indexing ij
        # of output
        aryRad, aryTht = np.meshgrid(vecRad, vecTht, indexing='ij')

        # faltten arrays to be able to combine them with meshgrid
        vecRad = aryRad.flatten()
        vecTht = aryTht.flatten()

        # convert from polar to cartesian
        vecX, vecY = map_pol_to_crt(vecTht, vecRad)

        # Vector with standard deviations pRF models (in degree of vis angle):
        vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                               endpoint=True)

        if kwUnt == 'deg':
            # since parameters are already in degrees of visual angle,
            # we do nothing
            pass

        elif kwUnt == 'pix':
            # convert parameters to pixels
            vecX, vecY, vecPrfSd = rmp_deg_pixel_x_y_s(vecX, vecY, vecPrfSd,
                                                       tplPngSize, varExtXmin,
                                                       varExtXmax, varExtYmin,
                                                       varExtYmax)

        # Put all combinations of x-position, y-position, and standard
        # deviations into the array:

        # Loop through x-positions:
        for idxXY in range(0, varNum1*varNum2):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = vecX[idxXY]
                aryMdlParams[varCntMdlPrms, 1] = vecY[idxXY]
                aryMdlParams[varCntMdlPrms, 2] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms += 1

    else:
        print('Unknown keyword provided for coordinate system for model ' +
              'parameter combinations: should be either crt or pol')

    return aryMdlParams


def crt_mdl_rsp(arySptExpInf, tplPngSize, aryMdlParams, varPar):
    """Create responses of 2D Gauss models to spatial conditions.

    Parameters
    ----------
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space (width, height).
    aryMdlParams : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, 3]
        Model parameters (x, y, sigma) for all models.
    varPar : int, positive
        Number of cores to parallelize over.
    Returns
    -------
    aryMdlCndRsp : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions.
    Reference
    ---------
    [1]
    """

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = np.array_split(aryMdlParams, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for results from parallel processes (for pRF model responses):
    lstMdlTc = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    print('---------Running parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvl_2D_gauss,
                                     args=(idxPrc, lstMdlParams[idxPrc],
                                           arySptExpInf, tplPngSize, queOut
                                           )
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstMdlTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')
    # Put output arrays from parallel process into one big array
    lstMdlTc = sorted(lstMdlTc)
    aryMdlCndRsp = np.empty((0, arySptExpInf.shape[-1]))
    for idx in range(0, varPar):
        aryMdlCndRsp = np.concatenate((aryMdlCndRsp, lstMdlTc[idx][1]), axis=0)

    # Clean up:
    del(lstMdlParams)
    del(lstMdlTc)

    return aryMdlCndRsp.astype('float16')


def crt_nrl_tc(aryMdlRsp, aryTmpExpInf, varTr, varNumVol, varTmpOvsmpl):
    """Create temporally upsampled neural time courses.

    Parameters
    ----------
    aryMdlRsp : 2d numpy array, shape [n_x_pos * n_y_pos * n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions.
    aryTmpExpInf : 2d numpy array, shape [unknown, 3]
        Info about nature, onset and duration of conditions during experiments.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment
    varNumVol : float, positive
        Number of data point (volumes) in the (fMRI) data
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.
    Returns
    -------
    aryNrlTc : 2d numpy array,
               shape [n_x_pos * n_y_pos * n_sd, varNumVol*varTmpOvsmpl]
        Neural time course models in temporally upsampled space
    Reference
    ---------
    [1]
    """
    # adjust the input, if necessary, such that input is 2D
    tplInpShp = aryMdlRsp.shape
    aryMdlRsp = aryMdlRsp.reshape((-1, aryMdlRsp.shape[-1]))

    # create boxcar functions in temporally upsampled space
    print('---------Create boxcar functions for spatial condtions')
    aryBxCarTmp = create_boxcar(aryTmpExpInf[:, 0], aryTmpExpInf[:, 1],
                                aryTmpExpInf[:, 2], varTr, varNumVol,
                                aryExclCnd=np.array([0.]),
                                varTmpOvsmpl=varTmpOvsmpl).T

    # pre-allocate pixelwise boxcar array
    aryNrlTc = np.zeros((aryMdlRsp.shape[0], aryBxCarTmp.shape[-1]),
                        dtype='float16')
    print('---------Insert predicted condition values for all models')
    # loop through boxcar functions of conditions
    for ind, vecCndOcc in enumerate(aryBxCarTmp):
        # get response predicted by models for this specific spatial condition
        rspValPrdByMdl = aryMdlRsp[:, ind]
        # insert predicted response value several times using broad-casting
        aryNrlTc[..., vecCndOcc.astype('bool')] = rspValPrdByMdl[:, None]

    # determine output shape
    tplOutShp = tplInpShp[:-1] + (int(varNumVol*varTmpOvsmpl), )

    return aryNrlTc.reshape(tplOutShp).astype('float16')


def crt_prf_tc(aryNrlTc, varNumVol, varTr, varTmpOvsmpl, switchHrfSet,
               tplPngSize, varPar):
    """Convolve every neural time course with HRF function.

    Parameters
    ----------
    aryNrlTc : 4d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_vol]
        Temporally upsampled neural time course models.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varTmpOvsmpl : int, positive
        Factor by which the data hs been temporally upsampled.
    switchHrfSet : int, (1, 2, 3)
        Switch to determine which hrf basis functions are used
    tplPngSize : tuple
        Pixel dimensions of the visual space (width, height).
    varPar : int, positive
        Description of input 1.
    Returns
    -------
    aryNrlTcConv : 5d numpy array,
                   shape [n_x_pos, n_y_pos, n_sd, n_hrf_bases, varNumVol]
        Neural time courses convolved with HRF basis functions
    Reference
    ---------
    [1]
    """

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryNrlTc.shape
    aryNrlTc = np.reshape(aryNrlTc, (-1, aryNrlTc.shape[-1]))

    # Put input data into chunks:
    lstNrlTc = np.array_split(aryNrlTc, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstConv = [None] * varPar

    print('---------Running parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvl_tc,
                                     args=(idxPrc,
                                           lstNrlTc[idxPrc],
                                           lstHrf,
                                           varTr,
                                           varNumVol,
                                           varTmpOvsmpl,
                                           queOut)
                                     )

        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstConv[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')
    # Put output into correct order:
    lstConv = sorted(lstConv)
    # Concatenate convolved pixel time courses (into the same order
    aryNrlTcConv = np.zeros((0, switchHrfSet, varNumVol), dtype=np.float32)
    for idxRes in range(0, varPar):
        aryNrlTcConv = np.concatenate((aryNrlTcConv, lstConv[idxRes][1]),
                                      axis=0)
    # clean up
    del(aryNrlTc)
    del(lstConv)

    # Reshape results:
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (varNumVol, )

    # Return:
    return np.reshape(aryNrlTcConv, tplOutShp).astype(np.float32)


def fnd_unq_rws(A, return_index=False, return_inverse=False):
    """Find unique rows in 2D array.

    Parameters
    ----------
    A : 2d numpy array
        Array for which unique rows should be identified.
    return_index : bool
        Bool to decide whether I is returned.
    return_inverse : bool
        Bool to decide whether J is returned.
    Returns
    -------
    B : 1d numpy array,
        Unique rows
    I: 1d numpy array, only returned if return_index is True
        B = A[I,:]
    J: 2d numpy array, only returned if return_inverse is True
        A = B[J,:]
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                  return_index=return_index,
                  return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')