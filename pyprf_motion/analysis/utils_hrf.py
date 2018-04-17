# -*- coding: utf-8 -*-

"""Main functions for hrf."""

# Part of pyprf_motion library
# Copyright (C) 2016  Marian Schneider, Ingo Marquardt
#
# Functions spm_hrf_compat, spmt, dspmt, ddspmt were copied from nipy
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

from __future__ import division
from copy import deepcopy
from functools import partial
import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d


# %% Hrf functions for convultion taken from nipy
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio=6,
                   normalize=True,
                   ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and
    dispersion `peak_disp`), minus an *undershoot* gamma PDF (with location
    `under_delay` and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning. SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


def spmt(t):
    """ SPM canonical HRF, HRF values for time values `t`

    This is the canonical HRF function as used in SPM. It
    has the following defaults:
                                                defaults
                                                (seconds)
    delay of response (relative to onset)         6
    delay of undershoot (relative to onset)      16
    dispersion of response                        1
    dispersion of undershoot                      1
    ratio of response to undershoot               6
    onset (seconds)                               0
    length of kernel (seconds)                   32
    """
    return spm_hrf_compat(t, normalize=True)


def dspmt(t):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

    This is the canonical HRF derivative function as used in SPM.

    It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1
    """
    t = np.asarray(t)
    return spmt(t) - spmt(t - 1)


_spm_dd_func = partial(spm_hrf_compat, normalize=True, peak_disp=1.01)


def ddspmt(t):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    This is the canonical HRF dispersion derivative function as used in SPM.

    It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).
    """
    return (spmt(t) - _spm_dd_func(t)) / 0.01


def create_boxcar(conditions, onsets, durations, varTr, varNumVol,
                  excl_cond=None, varTmpOvsmpl=1000.):
    """
    Creation of condition time courses in temporally upsampled space.
    """
    if excl_cond is not None:
        for cond in excl_cond:
            onsets = onsets[conditions != cond]
            durations = durations[conditions != cond]
            conditions = conditions[conditions != cond]

    resolution = varTr / float(varTmpOvsmpl)
    conditions = np.asarray(conditions)
    onsets = np.asarray(onsets, dtype=np.float)
    unique_conditions = np.sort(np.unique(conditions))
    boxcar = []

    for c in unique_conditions:
        tmp = np.zeros(int(varNumVol * varTr/resolution))
        onset_c = onsets[conditions == c]
        duration_c = durations[conditions == c]
        onset_idx = np.round(onset_c / resolution).astype(np.int)
        duration_idx = np.round(duration_c / resolution).astype(np.int)
        aux = np.arange(int(varNumVol * varTr/resolution))
        for start, dur in zip(onset_idx, duration_idx):
            lgc = np.logical_and(aux >= start, aux < start + dur)
            tmp = tmp + lgc
        assert np.all(np.less(tmp, 2))
        boxcar.append(tmp)
    boxcar_out = np.array(boxcar).T
    if boxcar_out.shape[1] == 1:
        boxcar_out = np.squeeze(boxcar_out)
    return boxcar_out.astype('float16')


def cnvl_tc(idxPrc, aryPrfTcChunk, lstHrf, varTr, varNumVol, varTmpOvsmpl,
            queOut, varHrfLen=32.):
    """
    Convolution of time courses with HRF model.
    """

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = deepcopy(aryPrfTcChunk.shape)
    aryPrfTcChunk = aryPrfTcChunk.reshape((-1, aryPrfTcChunk.shape[-1]))

    # prepare hrf basis functions
    lstBse = []
    for fnHrf in lstHrf:
        # needs to be a multiple of varTmpOvsmpl
        vecTmpBse = fnHrf(np.linspace(0, varHrfLen,
                                      (varHrfLen // varTr) * varTmpOvsmpl))
        # normalise HRF so that the sum of values is 1 (see FSL)
        # otherwise, after convolution values for predictors are very high
        vecTmpBse = np.divide(vecTmpBse, np.sum(vecTmpBse))

        lstBse.append(vecTmpBse)

    # get frame times, i.e. start point of every volume in seconds
    vecFrms = np.arange(0, varTr * varNumVol, varTr)
    # get supersampled frames times, i.e. start point of every volume in
    # upsampled res, since convolution takes place in temp. upsampled space
    vecFrmTms = np.arange(0, varTr * varNumVol, varTr / varTmpOvsmpl)

    # Prepare an empty array for ouput
    aryConv = np.zeros((aryPrfTcChunk.shape[0], len(lstHrf), varNumVol),
                       dtype='float16')
    print("---------Process " + str(idxPrc) +
          ": Convolve")
    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course (already in upsampled space):
        vecTcUps = aryPrfTcChunk[idxTc, :]

        # *** convolve
        for indBase, base in enumerate(lstBse):
            # perform the convolution
            col = np.convolve(base, vecTcUps, mode='full')[:vecTcUps.size]
            # get function for downsampling
            f = interp1d(vecFrmTms, col)
            # downsample to original resoltuion to match res of data
            # take the value from the centre of each volume's period (see FSL)
            aryConv[idxTc, indBase, :] = f(vecFrms + varTr/2.)

    # determine output shape
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (varNumVol, )

    # Create list containing the convolved timecourses, and the process ID:
    lstOut = [idxPrc,
              aryConv.reshape(tplOutShp)]

    # Put output to queue:
    queOut.put(lstOut)
