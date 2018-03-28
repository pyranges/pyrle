
# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize

# example_module = Extension('convolve', sources=['convolve.c'])

setup(name='rle', ext_modules=cythonize("rle.pyx"), py_modules=["rle"])
