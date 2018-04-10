#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This pre-processing script does the following:
- load conditions.npz
- run cumulative sum on durations
- write a single text file per run
- write a cumulative text file for all runs
"""

import os
import numpy as np

# %% set parameters
strPathParent = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02"
varRun = 1
strPathCond = os.path.join(strPathParent, '03_MotLoc', 'expInfo',
                           'Conditions_MotLoc_run0' + str(varRun) + '.npz')

# %% load conditions file
npzfile = np.load(strPathCond)
aryCond = npzfile["conditions"].astype('int8')

# aryCond is a varNrCond x 2 array. The first column contains the index for the
# spatial aperture, the second contains the index for the type of the aperture
# (vertical bar, horizontal bar, wedge)