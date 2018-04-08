# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import config as cfg
from utils import balancedLatinSquares


# %% set paramters
expectedTR = 2
targetDuration = 0.5
targetDist = targetDuration + 1.2

# total number of conditions
nrOfCond = 3

# set number of blank trials
nrNullTrialStart = 7  # 7
nrNullTrialEnd = 7
nrNullTrialBetw = 14

# prepare vectors for presentation order
barApertures = np.linspace(1, cfg.barSteps, cfg.barSteps)
wedgeApertures = np.linspace(1, cfg.wedgeSteps, cfg.wedgeSteps)

# prepare modules of presentation order
modulesPresOrder = [
    np.hstack((barApertures, barApertures[::-1])),  # horibar
    np.hstack((barApertures, barApertures[::-1])),  # vertiBar
    np.hstack((wedgeApertures[::-1], wedgeApertures)),  # wedge
    np.hstack((wedgeApertures, wedgeApertures[::-1],)),  # wedge other way
    ]

# counterbalance occurance of vertiBar, horiBar and wedge
# add 1 since the wedge appears twice
lst = balancedLatinSquares(nrOfCond+1)
for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
aryPerm = np.array(lst)

# loop through combinations
for ind, indCond in enumerate(aryPerm):
    pass
    presOrder = np.hstack((np.zeros(nrNullTrialStart),
                           modulesPresOrder[indCond[0]-1],
                           np.zeros(nrNullTrialBetw),
                           modulesPresOrder[indCond[1]-1],
                           np.zeros(nrNullTrialBetw),
                           modulesPresOrder[indCond[2]-1],
                           np.zeros(nrNullTrialBetw),
                           modulesPresOrder[indCond[3]-1],
                           np.zeros(nrNullTrialEnd),
                           ))
    apert = np.hstack((np.zeros(nrNullTrialStart),
                       np.ones(len(modulesPresOrder[indCond[0]-1]))*indCond[0],
                       np.zeros(nrNullTrialBetw),
                       np.ones(len(modulesPresOrder[indCond[1]-1]))*indCond[1],
                       np.zeros(nrNullTrialBetw),
                       np.ones(len(modulesPresOrder[indCond[2]-1]))*indCond[2],
                       np.zeros(nrNullTrialBetw),
                       np.ones(len(modulesPresOrder[indCond[3]-1]))*indCond[3],
                       np.zeros(nrNullTrialEnd),
                       ))
    # replace last element with second last element, since both wedge types are
    # represented by the second last element
    apert[apert == nrOfCond+1] = nrOfCond
    # concatenate presOrder and aperture
    conditions = np.vstack((presOrder, apert)).T

    # %% Prepare target times
    targetSwitch = True
    while targetSwitch:
        # prepare targets
        targetTRs = np.zeros(len(conditions)).astype('bool')
        targetPos = np.random.choice(np.arange(3), size=len(conditions),
                                     replace=True,
                                     p=np.array([1/3., 1/3., 1/3.]))
        targetTRs[targetPos == 1] = True
        nrOfTargets = np.sum(targetTRs)

        # prepare random target onset delay
        targetOffsetSec = np.random.uniform(0.1,
                                            expectedTR-targetDuration,
                                            size=nrOfTargets)

        targets = np.arange(0, len(conditions)*expectedTR,
                            expectedTR)[targetTRs]
        targets = targets + targetOffsetSec
        targetSwitch = np.any(np.diff(targets) < targetDist)

    # %% save the results
    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if 2*ind+1 > 9:
        filename = os.path.join(strPathParentUp, 'conditions',
                                'Conditions_MotLoc_run' + str(ind+1))
    else:
        filename = os.path.join(strPathParentUp, 'conditions',
                                'Conditions_MotLoc_run0' + str(ind+1))
    np.savez(filename, conditions=conditions.astype('int8'),
             targets=targets, targetDuration=targetDuration,
             expectedTR=expectedTR)
