# -*- coding: utf-8 -*-
"""pRF finding function definitions."""

# Part of py_pRF_motion library
# Copyright (C) 2016  Ingo Marquardt
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

import os
import numpy as np
import scipy as sp
import nibabel as nb


def load_nii(strPathIn, varSzeThr=5000.0):
    """
    Load nii file.

    Parameters
    ----------
    strPathIn : str
        Path to nii file to load.
    varSzeThr : float
        If the nii file is larger than this threshold (in MB), the file is
        loaded volume-by-volume in order to prevent memory overflow. Default
        threshold is 1000 MB.

    Returns
    -------
    aryNii : np.array
        Array containing nii data. 32 bit floating point precision.
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    Notes
    -----
    If the nii file is larger than the specified threshold (`varSzeThr`), the
    file is loaded volume-by-volume in order to prevent memory overflow. The
    reason for this is that nibabel imports data at float64 precision, which
    can lead to a memory overflow even for relatively small files.
    """
    # Load nii file (this does not load the data into memory yet):
    objNii = nb.load(strPathIn)

    # Get size of nii file:
    varNiiSze = os.path.getsize(strPathIn)

    # Convert to MB:
    varNiiSze = np.divide(float(varNiiSze), 1000000.0)

    # Load volume-by-volume or all at once, depending on file size:
    if np.greater(varNiiSze, float(varSzeThr)):

        # Load large nii file

        print(('---------Large file size ('
              + str(np.around(varNiiSze))
              + ' MB), reading volume-by-volume'))

        # Get image dimensions:
        tplSze = objNii.shape

        # Create empty array for nii data:
        aryNii = np.zeros(tplSze, dtype=np.float32)

        # Loop through volumes:
        for idxVol in range(tplSze[3]):
            aryNii[..., idxVol] = np.asarray(
                  objNii.dataobj[..., idxVol]).astype(np.float32)

    else:

        # Load small nii file

        # Load nii file (this doesn't load the data into memory yet):
        objNii = nb.load(strPathIn)

        # Load data into array:
        aryNii = np.asarray(objNii.dataobj).astype(np.float32)

    # Get headers:
    objHdr = objNii.header

    # Get 'affine':
    aryAff = objNii.affine

    # Output nii data (as numpy array), header, and 'affine':
    return aryNii, objHdr, aryAff


def crt_2D_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1]
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX)) + np.square((aryY - varPosY))) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * np.square(varSd))

    return aryGauss


def cnvl_2D_gauss(idxPrc, aryBoxCar, aryMdlParamsChnk, tplPngSize, varNumVol,
                  queOut):
    """Spatially convolve boxcar functions with 2D Gaussian.

    Parameters
    ----------
    idxPrc : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    aryBoxCar : float, positive
      Description of input 2.
    aryMdlParamsChnk : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    tplPngSize : float, positive
      Description of input 2.
    varNumVol : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    queOut : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Determine number of motion directions
    varNumMtnDrtn = aryBoxCar.shape[2]

    # Output array with pRF model time courses:
    aryOut = np.zeros([varChnkSze, varNumMtnDrtn, varNumVol])

    # Loop through different motion directions:
    for idxMtn in range(0, varNumMtnDrtn):
        # Loop through combinations of model parameters:
        for idxMdl in range(0, varChnkSze):

            # Spatial parameters of current model:
            varTmpX = aryMdlParamsChnk[idxMdl, 1]
            varTmpY = aryMdlParamsChnk[idxMdl, 2]
            varTmpSd = aryMdlParamsChnk[idxMdl, 3]

            # Create pRF model (2D):
            aryGauss = crt_2D_gauss(tplPngSize[0],
                                    tplPngSize[1],
                                    varTmpX,
                                    varTmpY,
                                    varTmpSd)

            # Multiply pixel-time courses with Gaussian pRF models:
            aryPrfTcTmp = np.multiply(aryBoxCar[:, :, idxMtn, :],
                                      aryGauss[:, :, None])

            # Calculate sum across x- and y-dimensions - the 'area under the
            # Gaussian surface'. This is essentially an unscaled version of the
            # pRF time course model (i.e. not yet scaled for size of the pRF).
            aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1))

            # Put model time courses into function's output with 2d Gaussian
            # arrray:
            aryOut[idxMdl, idxMtn, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    lstOut = [idxPrc,
              aryOut]

    # Put output to queue:
    queOut.put(lstOut)


class cls_set_config(object):
    """
    Set config parameters from dictionary into local namespace.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.
    """

    def __init__(self, dicCnfg):
        """Set config parameters from dictionary into local namespace."""
        self.__dict__.update(dicCnfg)
