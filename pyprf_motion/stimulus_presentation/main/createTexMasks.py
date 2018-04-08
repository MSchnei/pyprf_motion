#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to creature textures and masks for the motion localiser.
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
from PIL import Image
import config as cfg
from utils import (carrierPattern, createBinCircleMask,  getDistIma,
                   assignBorderVals)

# %% create the carrier pattern (texture for the bars)

# create template
barTexture = np.zeros((cfg.pix, cfg.pix, cfg.nFrames/cfg.cycPerSec))
# set the phase of the carrier pattern
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec)
lamb = np.sin(phase)/4. + 0.5
# create the texture over time frames
for ind, (t, d) in enumerate(zip(phase, lamb)):
    ima = carrierPattern(d, t, cfg.numSquaresBars, cfg.pix)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    barTexture[..., ind] = ima

# save array as images, if wanted
for ind in np.arange(barTexture.shape[2]):
    im = Image.fromarray(255*barTexture[..., ind].astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
            + "Ima_" + str(ind) + ".png")

# assign parts of the texture to get the bars
horiBar = np.zeros((barTexture.shape))
vertiBar = np.zeros((barTexture.shape))
horiBar[0:cfg.pix/cfg.numSquaresBars, :, :] = np.copy(
    barTexture[0:cfg.pix/cfg.numSquaresBars, :, :])
vertiBar[:, 0:cfg.pix/cfg.numSquaresBars, :] = np.copy(
    barTexture[:, 0:cfg.pix/cfg.numSquaresBars, :])

# %% create masks (for the bars)

# create templates for horizontal bar
horiBarMasks = np.zeros((cfg.pix, cfg.pix, cfg.barSteps),
                        dtype='float16')
horiBarShape = np.zeros((cfg.pix, cfg.pix))
horiBarShape[0:cfg.pix/cfg.numSquaresBars, :] = 1
horiBarMasksFitting = np.zeros((cfg.pix, cfg.pix, cfg.barSteps),
                               dtype='float16')
# create templates for vertical bar
vertiBarMasks = np.zeros((cfg.pix, cfg.pix, cfg.barSteps),
                         dtype='float16')
vertiBarShape = np.zeros((cfg.pix, cfg.pix))
vertiBarShape[:, 0:cfg.pix/cfg.numSquaresBars] = 1
vertiBarMasksFitting = np.zeros((cfg.pix, cfg.pix, cfg.barSteps),
                                dtype='float16')
# create overall circular aperture for all stimuli
circleAperture = createBinCircleMask(cfg.fovHeight, cfg.pix, rLow=cfg.minR,
                                     rUp=cfg.fovHeight/2., thetaMin=0,
                                     thetaMax=360)
# determine the bar positions in pixels
barSizeinPix = cfg.pix*(cfg.barSize/cfg.fovHeight)
posInPix = np.linspace(0, cfg.pix-barSizeinPix, cfg.barSteps)
# combine the horiBarShape with the circleAperture
for ind, pos in enumerate(posInPix):
    # roll forward
    ima = np.roll(horiBarShape, int(pos), axis=0).astype('bool')
    # combine with circle aperture
    ima = np.logical_and(ima, circleAperture)
    # save inbetween result for later fitting
    horiBarMasksFitting[:, :, ind] = ima[::-1, ...]
    # ramp the borders with a cosine function
    if np.any(ima == 1):
        print "image " + str(ind)
        # get its distance image
        distIma = getDistIma(ima, cfg.fovHeight, cfg.pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        ima = assignBorderVals(ima, distIma, borderRange=cfg.borderRange)
    # roll back
    ima = np.roll(ima, -int(pos), axis=0)
    horiBarMasks[:, :, ind] = ima
# combine the vertiBarShape with the circleAperture
for ind, pos in enumerate(posInPix):
    # roll forward
    ima = np.roll(vertiBarShape, int(pos), axis=1).astype('bool')
    # combine with circle aperture
    ima = np.logical_and(ima, circleAperture)
    # save inbetween result for later fitting
    vertiBarMasksFitting[:, :, ind] = ima[::-1, ...]
    # ramp the borders with a cosine function
    if np.any(ima == 1):
        print "image " + str(ind)
        # get its distance image
        distIma = getDistIma(ima, cfg.fovHeight, cfg.pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        ima = assignBorderVals(ima, distIma, borderRange=cfg.borderRange)
    # roll back
    ima = np.roll(ima, -int(pos), axis=1)
    vertiBarMasks[:, :, ind] = ima

# adjust range for psychopy
horiBarMasks = horiBarMasks * 2 - 1
vertiBarMasks = vertiBarMasks * 2 - 1

# %% create the carrier pattern (texture for the wedge)

# create template
wedgeTexture = np.zeros((cfg.pix, cfg.pix, cfg.nFrames/cfg.cycPerSec))
# create the texture over time frames
for ind, (t, d) in enumerate(zip(phase, lamb)):
    ima = carrierPattern(d, t, cfg.numSquaresWedge, cfg.pix)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    wedgeTexture[..., ind] = ima

# %% create the masks for the wedge

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, cfg.wedgeSteps, endpoint=False)
maxTheta = minTheta + cfg.wedgeWidth
# get combinations of angles and radis
combis = zip(np.zeros(len(minTheta)), np.ones(len(minTheta))*cfg.fovHeight/2.,
             minTheta, maxTheta)
# create the wedgeMasks
wedgeMasks = np.empty((cfg.pix, cfg.pix, len(combis)), dtype='float16')
for ind, combi in enumerate(combis):
    wedgeMasks[..., ind] = createBinCircleMask(cfg.fovHeight, cfg.pix,
                                               rLow=combi[0],
                                               rUp=combi[1],
                                               thetaMin=combi[2],
                                               thetaMax=combi[3],
                                               rMin=cfg.minR)
# save the wedgeMask for later fitting
wedgeMasksFitting = np.copy(wedgeMasks[::-1, ...])

# ramp the borders of wedgeMasks with a cosine function
for i in range(wedgeMasks.shape[-1]):
    # get a single mask
    binMask = wedgeMasks[..., i]
    # check whether there is at least one element of 1
    if np.any(binMask == 1):
        print "image " + str(i)
        # get its distance image
        distIma = getDistIma(binMask, cfg.fovHeight, cfg.pix)
        # assign raised cosine values to pixels less than borderRange
        wedgeMasks[..., i] = assignBorderVals(binMask, distIma,
                                              borderRange=cfg.borderRange)
    else:
        # assign old contrast mask
        wedgeMasks[..., i] = binMask

# save wedge masks as images, if wanted
for ind in np.arange(wedgeMasks.shape[2]):
    im = Image.fromarray(255*wedgeMasks[..., ind].astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/wedgeMasks/"
            + "Ima_" + str(ind) + ".png")

# adjust range for psychopy
wedgeMasks = wedgeMasks*2-1

# %%  save textures
strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

fileName = os.path.join(strPathParentUp, 'maskTextures', 'Textures_MotLoc')
np.savez(fileName, horiBar=horiBar.astype('int8'),
         vertiBar=vertiBar.astype('int8'), wedge=wedgeTexture.astype('int8'))

# %% save masks
fileName = os.path.join(strPathParentUp, 'maskTextures', 'Masks_MotLoc')
np.savez(fileName, horiBarMasks=horiBarMasks.astype('float16'),
         vertiBarMasks=vertiBarMasks.astype('float16'),
         wedgeMasks=wedgeMasks.astype('float16'),
         horiBarMasksFitting=horiBarMasksFitting.astype('int8'),
         vertiBarMasksFitting=vertiBarMasksFitting.astype('int8'),
         wedgeMasksFitting=wedgeMasksFitting.astype('int8'))
