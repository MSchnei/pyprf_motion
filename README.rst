pyprf_motion
============

Population receptive field analysis for motion-sensitive early- and
mid-level visual cortex.

This is an extension of the `pyprf
package <https://github.com/ingo-m/pypRF>`__. Compared to pyprf,
pyprf_motion offers stimuli that were specifically optimized to elicit
responses from motion-sensitive areas. On the analysis side,
pyprf_motion offers some additional features made necessary by the
different stimulation type (model positions defined in polar
coordinates, sub-TR temporal resolution for model creation,
cross-validation for model fitting) at the cost of some speed and
flexibility. There is currently no support for GPU.

Installation
------------

For installation, follow these steps:

0. (Optional) Create conda environment

.. code:: bash

   conda create -n env_pyprf_motion python=2.7
   source activate env_pyprf_motion
   conda install pip

1. Clone repository

.. code:: bash

   git clone https://github.com/MSchnei/pyprf_motion.git

2. Install numpy, e.g. by running:

.. code:: bash

   pip install numpy

3. Install pyprf_motion with pip

.. code:: bash

   pip install /path/to/cloned/pyprf_motion

Dependencies
------------

`Python 2.7 <https://www.python.org/download/releases/2.7/>`__

+----------------------------------------------+----------------+
| Package                                      | Tested version |
+==============================================+================+
| `NumPy <http://www.numpy.org/>`__            | 1.14.0         |
+----------------------------------------------+----------------+
| `SciPy <http://www.scipy.org/>`__            | 1.0.0          |
+----------------------------------------------+----------------+
| `NiBabel <http://nipy.org/nibabel/>`__       | 2.2.1          |
+----------------------------------------------+----------------+
| `cython <http://cython.org/>`__              | 0.27.1         |
+----------------------------------------------+----------------+
| `tensorflow <https://www.tensorflow.org/>`__ | 1.4.0          |
+----------------------------------------------+----------------+
| `scikit-learn <scikit-learn.org/>`__         | 0.19.1         |
+----------------------------------------------+----------------+

How to use
----------

1. Present stimuli and record fMRI data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PsychoPy scripts in the stimulus_presentation folder can be used to
map motion-sensitive visual areas (especially area hMT+) using the pRF
framework.

1. Specify your desired parameters in the config file.

2. Run the createTexMasks.py file to generate relevant masks and
   textures. Masks and textures will be saved as numpy arrays in .npz
   format in the parent folder called MaskTextures.

3. Run the createCond.py file to generate the condition order. Condition
   and target presentation orders will be saved as numpy arrays in .npz
   format in the parent folder called Conditions.

4. Run the stimulus presentation file motLoc.py in PsychoPy. The
   stimulus setup should look like the following screen-shot:

2. Prepare spatial and temporal information for experiment as arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run prepro_get_spat_info.py in the prepro folder to obtain an array
   with the spatial information of the experiment.

2. Run prepro_get_temp_info.py in the prepro folder to obtain an array
   with the temporal information of the experiment.

3. Prepare the input data
~~~~~~~~~~~~~~~~~~~~~~~~~

The input data should be motion-corrected, high-pass filtered and
(optionally) distortion-corrected. If desired, spatial as well as
temporal smoothing can be applied. The PrePro folder contains some
auxiliary scripts to perform some of these functions.

4. Adjust the csv file
~~~~~~~~~~~~~~~~~~~~~~

Adjust the information in the config_default.csv file in the Analysis
folder, such that the provided information is correct. It is recommended
to make a specific copy of the csv file for every subject.

5. Run pyprf_motion
~~~~~~~~~~~~~~~~~~~

Open a terminal and run

::

   pyprf_motion -config path/to/custom_config.csv

References
----------

This application is based on the following work:

-  Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field
   estimates in human visual cortex. NeuroImage, 39(2), 647–660.
   https://doi.org/10.1016/j.neuroimage.2007.09.034

-  Amano, K., Wandell, B. A., & Dumoulin, S. O. (2009). Visual field
   maps, population receptive field sizes, and visual field coverage in
   the human MT+ complex. Journal of Neurophysiology, 102(5), 2704–18.
   https://doi.org/10.1152/jn.00102.2009

-  van Dijk, J. A., de Haas, B., Moutsiana, C., & Schwarzkopf, D. S.
   (2016). Intersession reliability of population receptive field
   estimates. NeuroImage, 143, 293–303.
   https://doi.org/10.1016/j.neuroimage.2016.09.013

License
-------

The project is licensed under `GNU General Public License Version
3 <http://www.gnu.org/licenses/gpl.html>`__.
