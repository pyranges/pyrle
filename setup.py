
# setup.py

from distutils.core import setup, Extension
from Cython.Build import cythonize

# example_module = Extension('convolve', sources=['convolve.c'])

setup(name='pyrle', ext_modules=cythonize("pyrle.pyx"), py_modules=["pyrle"])
