""""""""""""
SNS-Toolbox
""""""""""""

Documentation is available on `ReadTheDocs <https://sns-toolbox.readthedocs.io/en/latest/index.html>`_:

.. image:: https://readthedocs.org/projects/sns-toolbox/badge/?version=latest
    :target: https://sns-toolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Installation Instructions
=========================

Install using pip:
::
    pip install sns-toolbox

Tutorial 6 uses the python package for OpenCV, please see its `installation instructions <https://pypi.org/project/opencv-python/>`_.

The package can also be built locally. To do so, clone the repository, navigate to the root directory, and run the following command:
::
    pip install -e

So far this has been tested using Python 3.8, which is also the newest version that pytorch (torch) is fully compatible with.

Pytorch may need to be installed separately from the other packages in pip, since which configuration of torch you install is dependent on your personal system configuration. For instructions on installing torch, please see their `documentation <https://pytorch.org/get-started/locally/>`_.

Citation
========

If you use this software for an academic publication, we ask that you please cite the following paper:

Nourse, W.R.P., Szczecinski, N.S., Quinn, R.D. (2022). SNS-Toolbox: A Tool for Efficient Simulation of Synthetic Nervous Systems. In: , et al. Biomimetic and Biohybrid Systems. Living Machines 2022. Lecture Notes in Computer Science(), vol 13548. Springer, Cham. https://doi.org/10.1007/978-3-031-20470-8_4
