# -*- coding: utf-8 -*-
"""Define pRF finding parameters here"""

# %% general stimulus parameters

# set the number of frames
nFrames = 60.
# set the height of the field of view in deg
fovHeight = 24.
# set the dimension (numbe rof pixels for the carrier pattern)
pix = 1024.
# determine cycles per second
cycPerSec = 2.


# %% set parameters for the texture

# set the number of squares that the carrier pattern should be made up of
# for the wedge
numSquaresWedge = 8.
# set the number of steps that the wedge should take
wedgeSteps = 32.
# set the width of the width in deg
wedgeWidth = 45.
# set the minimum radius fo the wedge
minR = 0.4

# set the number of squares that the carrier pattern should be made up of
# for the bars
numSquaresBars = 8.
# set the number of steps that divide the visual field
barSteps = 31.
# set the size of the bar aperture
barSize = 3.

# border range for the mask in deg of visual angle
borderRange = 0.3
