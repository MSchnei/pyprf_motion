# -*- coding: utf-8 -*-
"""Utilities for pRF model creation."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Marian Schneider, Ingo Marquardt
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
from PIL import Image
from pyprf_motion.analysis.utils_hrf import spmt, dspmt, ddspmt, cnvl_tc
from pyprf_motion.analysis.utils_general import cnvl_2D_gauss


def load_png(varNumVol, strPathPng, tplVslSpcSze=(200, 200), varStrtIdx=0,
             varZfill=3):
    """
    Load PNGs with stimulus information for pRF model creation.

    Parameters
    ----------
    varNumVol : int
        Number of PNG files.
    strPathPng : str
        Parent directory of PNG files. PNG files need to be organsied in
        numerical order (e.g. `file_001.png`, `file_002.png`, etc.).
    tplVslSpcSze : tuple
        Pixel size (x, y) at which PNGs are sampled. In case of large PNGs it
        is useful to sample at a lower than the original resolution.
    varStrtIdx : int
        Start index of PNG files. For instance, `varStrtIdx = 0` if the name of
        the first PNG file is `file_000.png`, or `varStrtIdx = 1` if it is
        `file_001.png`.
    varZfill : int
        Zero padding of PNG file names. For instance, `varStrtIdx = 3` if the
        name of PNG files is `file_007.png`, or `varStrtIdx = 4` if it is
        `file_0007.png`.

    Returns
    -------
    aryPngData : np.array
        3D Numpy array with the following structure:
        aryPngData[x-pixel-index, y-pixel-index, PngNumber]

    Notes
    -----
    Part of py_pRF_mapping library.
    """
    # Create list of png files to load:
    lstPngPaths = [None] * varNumVol
    for idx01 in range(0, varNumVol):
        lstPngPaths[idx01] = (strPathPng +
                              str(idx01 + varStrtIdx).zfill(varZfill) +
                              '.png')

    # The png data will be saved in a numpy array of the following order:
    # aryPngData[x-pixel, y-pixel, PngNumber].
    aryPngData = np.zeros((tplVslSpcSze[0],
                           tplVslSpcSze[1],
                           varNumVol))

    # Open first image in order to check dimensions (greyscale or RGB, i.e. 2D
    # or 3D).
    objIm = Image.open(lstPngPaths[0])
    aryTest = np.array(objIm.resize((objIm.size[0], objIm.size[1]),
                                    Image.ANTIALIAS))
    varNumDim = aryTest.ndim
    del(aryTest)

    # Loop trough PNG files:
    for idx01 in range(0, varNumVol):

        # Old version of reading images with scipy
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :, 0]
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :]

        # Load & resize image:
        objIm = Image.open(lstPngPaths[idx01])
        objIm = objIm.resize((tplVslSpcSze[0],
                              tplVslSpcSze[1]),
                             resample=Image.NEAREST)

        # Casting of array depends on dimensionality (greyscale or RGB, i.e. 2D
        # or 3D):
        if varNumDim == 2:
            aryPngData[:, :, idx01] = np.array(objIm.resize(
                (objIm.size[0], objIm.size[1]), Image.ANTIALIAS))[:, :]
        elif varNumDim == 3:
            aryPngData[:, :, idx01] = np.array(objIm.resize(
                (objIm.size[0], objIm.size[1]), Image.ANTIALIAS))[:, :, 0]
        else:
            # Error message:
            strErrMsg = ('ERROR: PNG files for model creation need to be RGB '
                         + 'or greyscale.')
            raise ValueError(strErrMsg)

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 200).astype(np.int8)

    return aryPngData


def load_ev_txt(strPthEv):
    """Load information from event text file.

    Parameters
    ----------
    input1 : str
        Path to event text file
    Returns
    -------
    aryEvTxt : 2d numpy array, shape [n_measurements, 3]
        Array with info about conditions: type, onset, duration
    Notes
    -----
    Part of py_pRF_mapping library.
    """
    aryEvTxt = np.loadtxt(strPthEv, dtype='float', comments='#', delimiter=' ',
                          skiprows=0, usecols=(0, 1, 2))
    return aryEvTxt

def crt_pw_bxcr_fn(aryPngData, varNumVol, vecMtDrctn=False, aryPresOrd=False):
    """Create pixel-wise boxcar functions.

    Parameters
    ----------
    input1 : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    input2 : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Create pixel-wise boxcar functions')
    aryBoxCar = np.empty(aryPngData.shape[0:2] + (len(vecMtDrctn),) +
                         (varNumVol,), dtype='int64')
    for ind, num in enumerate(vecMtDrctn):
        aryCondTemp = np.zeros((aryPngData.shape), dtype='int64')
        lgcTempMtDrctn = [aryPresOrd == num][0]
        aryCondTemp[:, :, lgcTempMtDrctn] = np.copy(
            aryPngData[:, :, lgcTempMtDrctn])
        aryBoxCar[:, :, ind, :] = aryCondTemp

    return aryBoxCar


def crt_nrl_tc(aryBoxCar, varNumMtDrctn, varNumVol, tplPngSize, varNumX,
               varExtXmin,  varExtXmax, varNumY, varExtYmin, varExtYmax,
               varNumPrfSizes, varPrfStdMin, varPrfStdMax, varPar):
    """Create neural model time courses from pixel-wise boxcar functions.

    Parameters
    ----------
    aryBoxCar : 4d numpy array, shape [n_x_pix, n_y_pix, n_mtn_dir, n_vol]
        Description of input 1.
    varNumMtDrctn : float, positive
        Description of input 2.
    varNumVol : float, positive
        Description of input 2.
    tplPngSize : tuple
        Description of input 2.
    varNumX : float, positive
        Description of input 2.
    varExtXmin : float, positive
        Description of input 2.
    varExtXmax : float, positive
        Description of input 2.
    varNumY : float, positive
        Description of input 2.
    varExtYmin : float, positive
        Description of input 2.
    varExtYmax : float, positive
        Description of input 2.
    varNumPrfSizes : float, positive
        Description of input 2.
    varPrfStdMin : float, positive
        Description of input 2.
    varPrfStdMax : float, positive
        Description of input 2.
    varPar : float, positive
        Description of input 2.
    Returns
    -------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Create neural time course models')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0, (tplPngSize[0] - 1), varNumX, endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0, (tplPngSize[1] - 1), varNumY, endpoint=True)

    # We calculate the scaling factor from degrees of visual angle to pixels
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in stimulus ' + \
        'space (in degrees of visual angle) and the ratio of X and Y ' + \
        'dimensions in the upsampled visual space do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector containing pRF sizes with the scaling factors.
    # Now the vector with the pRF sizes can be used directly for creation of
    # Gaussian pRF models in visual space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = np.array_split(aryMdlParams, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstPrfTc = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvl_2D_gauss,
                                     args=(idxPrc, aryBoxCar,
                                           lstMdlParams[idxPrc], tplPngSize,
                                           varNumVol, queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')
    # Put output arrays from parallel process into one big array
    lstPrfTc = sorted(lstPrfTc)
    aryPrfTc = np.empty((0, varNumMtDrctn, varNumVol))
    for idx in range(0, varPar):
        aryPrfTc = np.concatenate((aryPrfTc, lstPrfTc[idx][1]), axis=0)

    # check that all the models were collected correctly
    assert aryPrfTc.shape[0] == varNumMdls

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryNrlTc = np.zeros([varNumX, varNumY, varNumPrfSizes, varNumMtDrctn,
                         varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryNrlTc[idxX, idxY, idxSd, :, :] = aryPrfTc[
                    varCntMdlPrms, :, :]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    return aryNrlTc


def crt_prf_tc(aryNrlTc, varNumVol, varTr, tplPngSize, varNumMtDrctn,
               switchHrfSet, varPar,):
    """Convolve every neural time course with HRF function.

    Parameters
    ----------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Description of input 1.
    varNumVol : float, positive
        Description of input 2.
    varTr : float, positive
        Description of input 1.
    tplPngSize : tuple
        Description of input 1.
    varNumMtDrctn : int, positive
        Description of input 1.
    switchHrfSet :
        Description of input 1.
    varPar : int, positive
        Description of input 1.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
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

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvl_tc,
                                     args=(idxPrc,
                                           lstNrlTc[idxPrc],
                                           lstHrf,
                                           varTr,
                                           varNumVol,
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
    aryNrlTcConv = np.zeros((0, switchHrfSet, varNumVol))
    for idxRes in range(0, varPar):
        aryNrlTcConv = np.concatenate((aryNrlTcConv, lstConv[idxRes][1]),
                                      axis=0)
    # clean up
    del(aryNrlTc)
    del(lstConv)

    # Reshape results:
    tplOutShp = tplInpShp[:-2] + (varNumMtDrctn * len(lstHrf), ) + \
        (tplInpShp[-1], )

    # Return:
    return np.reshape(aryNrlTcConv, tplOutShp)
