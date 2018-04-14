#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This pre-processing script does the following:
- load conditions.npz
- run cumulative sum on durations
- write a single text file per run
- write a cumulative text file for all runs

Input:
- path to parent folder
- time to repeat (TR) in fMRI experiment
- stimulation time per TR
- Conditions_MotLoc_run01.npz

Output
- single text file per run
- cumulative text file for all runs
"""

import os
import numpy as np

# %% set parameters
strPthPrnt = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02"

# provide name of motLoc files in the order that they were shown
lstMotLoc = [
    "Conditions_MotLoc_run01.npz",
    "Conditions_MotLoc_run02.npz",
    "Conditions_MotLoc_run03.npz",
    "Conditions_MotLoc_run04.npz",
    ]

# provide the TR in seconds
varTr = 2.

# provide the stimulation time
varStmTm = 1.5

# %% load conditions files

# deduce path to conditions file
strPthCond = os.path.join(strPthPrnt, '03_MotLoc', 'expInfo')

# Loop through npz files in target directory:
lstCond = []
for ind, cond in enumerate(lstMotLoc):
    inputFile = os.path.join(strPthCond, cond)
    # extract condition
    aryTmp = np.load(inputFile)["conditions"].astype('int32')
    # create empty array
    aryTmpCond = np.empty((len(aryTmp), 3), dtype='float16')
    # get the condition nr
    aryTmpCond[:, 0] = aryTmp[:, 0] + aryTmp[:, 1] * np.max(aryTmp[:, 0])
    # get remapping to continuous numbers
    aryFrm = np.unique(aryTmpCond[:, 0])
    aryTo = np.argsort(np.unique(aryTmpCond[:, 0]))
    # apply mapping
    aryTmpCond[:, 0] = np.array(
        [aryTo[aryFrm == i][0] for i in aryTmpCond[:, 0]])
    # get the onset time
    aryTmpCond[:, 1] = np.cumsum(np.ones(len(aryTmp))*varTr) - varTr
    # get the duration
    aryTmpCond[:, 2] = np.ones(len(aryTmp))*varStmTm
    # create txt file
    strPthTxtFle = os.path.join(strPthPrnt, '03_MotLoc', 'expInfo', 'tmpInfo',
                                'run_' + str(ind + 1) + '_txt_eventmatrix.txt')
    np.savetxt(strPthTxtFle, aryTmpCond, delimiter=" ", fmt="%1.2f")
    # append condition to list
    lstCond.append(aryTmp)

# join conditions across runs
aryCond = np.vstack(lstCond)

# %% create aryTmpCond

# create empty array
aryTmpCondAll = np.empty((len(aryCond), 3), dtype='float16')
# get the condition nr
aryTmpCondAll[:, 0] = aryCond[:, 0] + aryCond[:, 1] * np.max(aryCond[:, 0])
# get remapping to continuous numbers
aryFrm = np.unique(aryTmpCondAll[:, 0])
aryTo = np.argsort(np.unique(aryTmpCondAll[:, 0]))
# apply mapping
aryTmpCondAll[:, 0] = np.array(
    [aryTo[aryFrm == i][0] for i in aryTmpCondAll[:, 0]])

# get the onset time
aryTmpCondAll[:, 1] = np.cumsum(np.ones(len(aryCond))*varTr) - varTr
# get the duration
aryTmpCondAll[:, 2] = np.ones(len(aryCond))*varStmTm

# aryTmpCondAll is a varNrCond x 3 array. The first column contains the index
# for the condition number, the second contains the onset time of the condition
# with respect to the start of the first run in [s] and the last column
# contains the duration

# %% create txt file
strPthTxtFle = os.path.join(strPthPrnt, '03_MotLoc', 'expInfo', 'tmpInfo',
                            'run_all_txt_eventmatrix.txt')
np.savetxt(strPthTxtFle, aryTmpCondAll, delimiter=" ", fmt="%1.2f")
