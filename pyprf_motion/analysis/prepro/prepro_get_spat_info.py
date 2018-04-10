#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This pre-processing script does the following:
- load Conditions.npz and load Masks.npz
- save resulting images as pngs or all images as 3D array

Input:
- path to parent folder
- downsampling factor
- Conditions_MotLoc_run01.npz
- Masks_MotLoc.npz

Output
- png files, one per TR
"""

import os
import numpy as np
from PIL import Image

# %% set parameters

# set path to parent folder
strPthPrnt = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02"

# provide name of motLoc files in the order that they were shown
lstMotLoc = [
    "Conditions_MotLoc_run01.npz",
    "Conditions_MotLoc_run02.npz",
    "Conditions_MotLoc_run03.npz",
    "Conditions_MotLoc_run04.npz",
    ]

# set factors for downsampling
factorX = 8
factorY = 8

# value to multipy mask value (1s) with for png format
scaleValue = 255

# %% load conditions files

# deduce path to conditions file
strPthCond = os.path.join(strPthPrnt, '03_MotLoc', 'expInfo')

# Loop through npz files in target directory:
lstCond = []
for cond in lstMotLoc:
    inputFile = os.path.join(strPthCond, cond)
    # extract condition
    aryTmpCnd = np.load(inputFile)["conditions"].astype('int8')
    # append condition to list
    lstCond.append(aryTmpCnd)

# join conditions across runs
arySptCond = np.vstack(lstCond)

# arySptCond is a varNrCond x 2 array. The first column contains the index for
# spatial aperture, the second contains the index for the type of the aperture
# (vertical bar, horizontal bar, wedge)

# %% load mask file

# deduce path to conditions file
strPthMsk = os.path.join(strPthPrnt, '03_MotLoc', 'expInfo',
                         'Masks_MotLoc.npz')

# Load npz file content:
with np.load((strPthMsk)) as objMsks:
    mskHoriBar = objMsks["horiBarMasksFitting"]
    mskVertiBar = objMsks["vertiBarMasksFitting"]
    mskWedge = objMsks["wedgeMasksFitting"]

# %% generate pngs

# load np arrays from dictionary and save their 2D slices as png
for index in np.arange(arySptCond.shape[0]):

    # get the index of the masks and conditions
    keyMask = arySptCond[index, 0]
    keyCond = arySptCond[index, 1]

    if keyCond == 0:
        ima = np.zeros(mskWedge.shape[:2])
    elif keyCond == 1:
        ima = mskHoriBar[..., keyMask-1]
    elif keyCond == 2:
        ima = mskVertiBar[..., keyMask-1]
    elif keyCond == 3:
        ima = mskWedge[..., keyMask-1]

    # if desired, downsample
    if factorX > 1 or factorY > 1:
        ima = ima[::factorX, ::factorY]

    im = Image.fromarray(scaleValue * ima.astype(np.uint8))
    if index > 999:
        filename = ("frame" + '' + str(index) + '.png')
    elif index > 99:
        filename = ("frame" + '0' + str(index) + '.png')
    elif index > 9:
        filename = ("frame" + '00' + str(index) + '.png')
    else:
        filename = ("frame" + '000' + str(index) + '.png')

    im.save((os.path.join(strPthPrnt, '03_MotLoc', 'expInfo', 'sptInfo',
                          filename)))
    print("Save ima: " + os.path.join(strPthPrnt, '03_MotLoc', 'expInfo',
                                      'sptInfo', filename))
