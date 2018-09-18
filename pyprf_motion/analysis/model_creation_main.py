# -*- coding: utf-8 -*-
"""pRF model creation."""

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
from pyprf_motion.analysis.utils_general import cls_set_config
from pyprf_motion.analysis.model_creation_utils import (crt_mdl_prms,
                                                        crt_mdl_rsp,
                                                        crt_nrl_tc,
                                                        crt_prf_tc)


def model_creation(cfg):
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    cfg : namespace
        Namespace containing variables from config file.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    """

    # %% Create model time courses
    # If desired by user in csv file or if exponents have been prvided via
    # command line
    if cfg.lgcCrteMdl:

        # %% Load spatial condition information

        print('------Load spatial condition information')

        arySptExpInf = np.load(cfg.strSptExpInf)

        # Here we assume scientific convention and orientation of images where
        # the origin should fall in the lower left corner, the x-axis occupies
        # the width and the y-axis occupies the height dimension of the screen.
        # We also assume that the first dimension that the user provides
        # indexes x and the second indexes the y-axis. Since python is column
        # major (i.e. first indexes columns, only then rows), we need to rotate
        # arySptExpInf by 90 degrees rightward. This will insure that with the
        # 0th axis we index the scientific x-axis and higher values move us to
        # the right on that x-axis. It will also ensure that the 1st
        # python axis indexes the scientific y-axis and higher values will
        # move us up.
        arySptExpInf = np.rot90(arySptExpInf, k=3)

        # %% Load temporal condition information

        print('------Load temporal condition information')

        aryTmpExpInf = np.load(cfg.strTmpExpInf)

        # %% Create model parameter combination, for now in pixel.

        aryMdlParams = crt_mdl_prms((int(cfg.varVslSpcSzeX),
                                     int(cfg.varVslSpcSzeY)), cfg.varNum1,
                                    cfg.varExtXmin, cfg.varExtXmax,
                                    cfg.varNum2, cfg.varExtYmin,
                                    cfg.varExtYmax, cfg.varNumPrfSizes,
                                    cfg.varPrfStdMin, cfg.varPrfStdMax,
                                    cfg.lstExp, kwUnt='pix',
                                    kwCrd=cfg.strKwCrd)

        # %% Create 2D Gauss model responses to spatial conditions.

        print('------Create 2D Gauss model responses to spatial conditions')

        aryMdlRsp = crt_mdl_rsp(arySptExpInf, (int(cfg.varVslSpcSzeX),
                                               int(cfg.varVslSpcSzeY)),
                                aryMdlParams, cfg.varPar)
        del(arySptExpInf)

        # %% Create pRF time courses

        # Because first upsampling and then convolving the time course models
        # is a very memory-intense process, we divide it into batches and loop
        varNumMdls = aryMdlRsp.shape[0]
        # Set the maximum batch size, this is to not explode RAM
        varBtchMaxSze = 150000.0
        # Split aryMdlRsp into bacthes
        lstMdlRsp = np.array_split(aryMdlRsp, int(varNumMdls/varBtchMaxSze))
        # Delete array to save memory
        del(aryMdlRsp)
        # Prepare list to collect pRF time courses
        lstPrfTc = []

        # Loop over batches
        for indBtc, aryMdlRsp in enumerate(lstMdlRsp):
            print('------Create pRF time courses, Batch ' + str(indBtc) +
                  ' out of ' + str(len(lstMdlRsp)))

            # Create neural time courses in temporally upsampled space
            aryNrlTc = crt_nrl_tc(aryMdlRsp, aryTmpExpInf, cfg.varTr,
                                  cfg.varNumVol, cfg.varTmpOvsmpl)

            # Convolve every neural time course model with hrf function(s)
            # And append outcome to list
            lstPrfTc.append(crt_prf_tc(aryNrlTc, cfg.varNumVol, cfg.varTr,
                            cfg.varTmpOvsmpl, cfg.switchHrfSet,
                            (int(cfg.varVslSpcSzeX), int(cfg.varVslSpcSzeY)),
                            cfg.varPar).astype(np.float16))
        del(aryTmpExpInf)
        del(aryMdlRsp)
        del(aryNrlTc)

        # Turn list into array
        aryPrfTc = np.concatenate(lstPrfTc, axis=0)
        aryPrfTc = aryPrfTc.astype(np.float32)

        # %% Save pRF time course models

        print('------Save pRF time course models to disk')

        # The data will come out of the convolution process with an extra
        # dimension, since in principle different basis functions in addition
        # to the canonical HRF can be used. But for now the model fitting can
        # only handle option 1 (canonical convolution). Therefore, we
        # check that user has set the switchHrfSet in the csv file to 1
        strErrMsg = 'Stopping program. ' + \
            'Only canonical hrf fitting is currently supported. ' + \
            'Set switchHrfSet equal to 1 in csv file in order to continue. '
        assert cfg.switchHrfSet == 1, strErrMsg
        # we reduce the dimensions by squeezing
        aryPrfTc = np.squeeze(aryPrfTc)

        # Save the 4D array as '*.npy' file:
        np.save(cfg.strPathMdl, aryPrfTc)
        # Save the corresponding model parameters
        np.save(cfg.strPathMdl + "_params", aryMdlParams)
        del(aryMdlParams)

        # %% Load existing pRF time course models
    else:
        print('------Load pRF time course models from disk')

        # Load the file:
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model matrix has the expected
        # dimensions:
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for correct dimensions:
        strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                     'models do not agree with specified model parameters')
        assert vecPrfTcShp[0] == cfg.varNum1 * \
            cfg.varNum2 * cfg.varNumPrfSizes * len(cfg.lstExp) and \
            vecPrfTcShp[1] == cfg.varNumVol, strErrMsg

    return aryPrfTc
