"""
pyprf_motion setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

import numpy as np
from setuptools import setup, Extension

setup(name='pyprf_motion',
      version='1.0.2',
      description=('Population receptive field analysis for motion-sensitive \
                    early- and mid-level visual cortex.'),
      url='https://github.com/MSchnei/pyprf_motion',
      author='Marian Schneider, Ingo Marquardt',
      author_email='marian.schneider@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel',
                        'cython==0.27.1', 'tensorflow==1.4.0',
                        'scikit-learn==0.19.1'],
      # setup_requires=['numpy'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      packages=['pyprf_motion.analysis'],
      py_modules=['pyprf_motion.analysis'],
      entry_points={
          'console_scripts': [
              'pyprf_motion = pyprf_motion.analysis.__main__:main',
              ]},
      ext_modules=[Extension('pyprf_motion.analysis.cython_leastsquares',
                             ['pyprf_motion/analysis/cython_leastsquares.c'],
                             include_dirs=[np.get_include()]
                             )],
      )
