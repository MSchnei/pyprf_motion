#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This pre-processing script does the following:
- load Conditions.npz and load Masks.npz
- save resulting images as pngs or all images as 3D array
"""

import os
import numpy as np
from PIL import Image

# %% set parameters

# set path to parent folder
strPathParent = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02"

# provide name of motLoc files in the order that they were shown
lstMotLoc = [
    "Conditions_MotLoc_run02.npz",
    "Conditions_MotLoc_run03.npz",
    "Conditions_MotLoc_run04.npz",
    "Conditions_MotLoc_run01.npz",
    ]

factorX = 8
factorY = 8

# value to multipy mask value (1s) with for png format
scaleValue = 255

# %% load conditions files

# deduce path to conditions file
strPthCond = os.path.join(strPathParent, '03_MotLoc', 'expInfo')

# Loop through npz files in target directory:
lstCond = []
for cond in lstMotLoc:
    inputFile = os.path.join(strPthCond, cond)
    # extract condition
    conditions = np.load(inputFile)["conditions"].astype('int8')
    # append condition to list
    lstCond.append(conditions)

# join conditions across runs
aryCond = np.vstack(lstCond)

# aryCond is a varNrCond x 2 array. The first column contains the index for the
# spatial aperture, the second contains the index for the type of the aperture
# (vertical bar, horizontal bar, wedge)

# %% load mask file

# deduce path to conditions file
strPthMsk = os.path.join(strPathParent, '03_MotLoc', 'expInfo',
                         'Masks_MotLoc.npz')

# Load npz file content:
with np.load((strPthMsk)) as objMsks:
    mskHoriBar = objMsks["horiBarMasksFitting"]
    mskVertiBar = objMsks["vertiBarMasksFitting"]
    mskWedge = objMsks["wedgeMasksFitting"]

# %% generate pngs

# load np arrays from dictionary and save their 2D slices as png
for index in np.arange(aryCond.shape[0]):

    # get the index of the masks and conditions
    keyMask = aryCond[index, 0]
    keyCond = aryCond[index, 1]

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

    im.save((os.path.join(strPathParent, '03_MotLoc', 'expInfo', 'pngs',
                          filename)))
    print("Save ima: " + os.path.join(strPathParent, '03_MotLoc', 'expInfo',
                                      'pngs', filename))
