"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.py to set analysis parameters.
"""

import os
import argparse
from pyprf_motion.analysis.pyprf_main import pyprf
from pyprf_motion.analysis.load_config import load_config
from pyprf_motion.analysis.utils_general import cls_set_config, cmp_res_R2
from pyprf_motion import __version__

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))
# ##### DEBUGGING ###############
#strCsvCnfg = "/home/marian/Documents/Testing/test_pyprf_motion/S02_config_MotLoc_lstCmpr.csv"
#lgcTest = False
# ###############################

def main():
    """pyprf_motion entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'pyprf_motion ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - config file path:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    objParser.add_argument('-compr', nargs='+',
                           help='List of exponents that will be used for \
                                 static nonlinearity.',
                           type=float, default=None)

    # Namespace object containing arguments and values:
    objNspc = objParser.parse_args()

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    # Proceed depending on command line input
    if strCsvCnfg is None:
        print('Option 01')
        # If no config argument was provided
        print('Please provide the file path to a config file, e.g.:')
        print('   pyprf_motion -config /path/to/my_config_file.csv')

    elif strCsvCnfg is not None and objNspc.compr is None:
        print('Option 02')
        # If config argument but no compr argument was provided

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False
        # Load config parameters from csv file into dictionary:
        dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest, lgcExpCmd=False)
        # Load config parameters from dictionary into namespace:
        cfg = cls_set_config(dicCnfg)
        # Call to main function, to invoke pRF analysis:
        pyprf(cfg, lgcTest=lgcTest, strExpSve='')

    elif strCsvCnfg is not None and objNspc.compr is not None:
        print('Option 03')
        # If config argument and compr argument were provided

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False
        # Get list with exponents
        lstExp = objNspc.compr

        # Loop over exponents and find best pRF
        for varExp in lstExp:
            # Load config parameters from csv file into dictionary:
            dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest, lgcExpCmd=True)
            # Load config parameters from dictionary into namespace. We do this
            # on every loop so we have a fresh start in case variables are
            # redefined during the prf analysis
            cfg = cls_set_config(dicCnfg)
            # Set model creation to True, so that the pRF models are updated
            # to new exponent
            cfg.lgcCrteMdl = True
            # Set new list of exponents in cfg namespace
            # Put exponent in a list so that it can be used in pRF analysis
            cfg.lstExp = [varExp]
            # Print to command line, so the user knows which exponent is used
            print('---Exponent for nonlinearity: ' + str(cfg.lstExp))
            # Derive str for saving inbetween results as nii for this exponent
            strExpSve = '_' + str(varExp)
            # Call to main function, to invoke pRF analysis:
            pyprf(cfg, lgcTest=lgcTest, strExpSve=strExpSve)

        # List with name suffices of output images:
        lstNiiNames = ['_x_pos',
                       '_y_pos',
                       '_SD',
                       '_exp',
                       '_R2',
                       '_polar_angle',
                       '_eccentricity']

        # Compare results for the different exponents, export nii files based
        # on the results of the comparison and delete in-between results
        cmp_res_R2(lstExp, lstNiiNames, cfg.strPathOut, posR2=4, lgcDel=True)


if __name__ == "__main__":
    main()
