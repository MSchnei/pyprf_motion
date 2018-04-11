# -*- coding: utf-8 -*-
"""pRF model creation."""

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
import nibabel as nb
from pyprf_motion.analysis.utils_general import cls_set_config
from pyprf_motion.analysis.model_creation_utils import load_png
from pyprf_motion.analysis.model_creation_utils import (crt_pw_bxcr_fn,
                                                        crt_nrl_tc,
                                                        crt_prf_tc)


def model_creation(dicCnfg):
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing config parameters.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    """
    # *************************************************************************
    # *** Load parameters from config file

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)
    # *************************************************************************

    if cfg.lgcCrteMdl:  #noqa

        # *********************************************************************
        # *** Load spatial stimulus information from PNG files:

        print('------Load spatial stimulus information from PNG files')

        aryPngData = load_png(cfg.varNumVol,
                              cfg.strPathPng,
                              cfg.tplVslSpcSze,
                              varStrtIdx=cfg.varStrtIdx,
                              varZfill=cfg.varZfill)
        # *********************************************************************

        # *********************************************************************
        # *** Create pixel-wise boxcar functions

        print('------Create pixel-wise boxcar functions')

        aryBoxCar = crt_pw_bxcr_fn(aryPngData, cfg.varNumVol)
#        del(aryPngData)

        # *********************************************************************

        # *********************************************************************
        # *** Create neural time course models

        print('------Create neural time course models')

        aryNrlTc = crt_nrl_tc(aryBoxCar, cfg.varNumMtDrctn, cfg.varNumVol,
                              cfg.tplPngSize, cfg.varNumX, cfg.varExtXmin,
                              cfg.varExtXmax, cfg.varNumY, cfg.varExtYmin,
                              cfg.varExtYmax, cfg.varNumPrfSizes,
                              cfg.varPrfStdMin, cfg.varPrfStdMax,
                              cfg.varPar)
#        del(aryBoxCar)
        # *********************************************************************

        # *********************************************************************
        # *** convolve every neural time course model with hrf function(s)

        print('------Create pRF time course models by convolution')

        aryPrfTc = crt_prf_tc(aryNrlTc, cfg.varNumVol, cfg.varTr,
                              cfg.tplPngSize, cfg.varNumMtDrctn,
                              cfg.switchHrfSet, cfg.varPar)
#        del(aryNrlTc)
        # *********************************************************************

        # *********************************************************************
        # Debugging feature:
        # np.save('/home/john/Desktop/aryPixConv.npy', aryPixConv)
        # *********************************************************************

        # *********************************************************************
        # *** Save pRF time course models

        print('------Save pRF time course models to disk')

        # Save the 4D array as '*.npy' file:
        np.save(cfg.strPathMdl,
                aryPrfTc)

        # Save 4D array as '*.nii' file (for debugging purposes):
        niiPrfTc = nb.Nifti1Image(aryPrfTc, np.eye(4))
        nb.save(niiPrfTc, cfg.strPathMdl)
        # *********************************************************************

    else:

        # *********************************************************************
        # %% Load existing pRF time course models

        print('------Load pRF time course models from disk')

        # Load the file:
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model matrix has the expected
        # dimensions:
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for correct dimensions:
        strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                     'models do not agree with specified model parameters')
        assert vecPrfTcShp[0] == cfg.varNumX and \
            vecPrfTcShp[1] == cfg.varNumY and \
            vecPrfTcShp[2] == cfg.varNumPrfSizes and \
            vecPrfTcShp[3] == cfg.varNumMtDrctn, strErrMsg

    # *************************************************************************

    return aryPrfTc
