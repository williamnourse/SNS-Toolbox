# SNS-Toolbox

Please see the documentation at https://sns-toolbox.readthedocs.io/en/latest/index.html

Better documentation coming soon. For now, please see the 'tutorials' folder. For network design, the python package 'graphviz' is required. For full simulation options the packages 'numpy', 'matplotlib', and 'torch' are required. Numpy will execute on the cpu, torch can be used on the gpu. I recommend installing all of the supporting packages in a virtual environment.

So far this has been tested using Python 3.8, which is also the newest version that pytorch (torch) is fully compatible with. If you are using 'pip' to manage your python installation, you can install this package and its requirements using the command 'pip install -e .' . This will allow you to call the sns_toolbox package and submodules from other directories without having to alter the path. Otherwise you will need to keep custom code in this local set of directories, or fiddle with the path. If you don't use pip, install the other packages with whatever package manager you prefer.

Pytorch must be installed separately from the other packages in pip, since which configuration of torch you install is dependent on your personal system configuration. For instructions on installing torch, please see https://pytorch.org/get-started/locally/

Tutorial 6 uses the python package for OpenCV, please see the installation instructions located at https://pypi.org/project/opencv-python/