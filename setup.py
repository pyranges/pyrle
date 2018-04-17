
# setup.py

from distutils.core import setup, Extension
from setuptools import find_packages
from Cython.Build import cythonize

# example_module = Extension('convolve', sources=['convolve.c'])

setup(name='pyrle',
      packages=find_packages(),
      ext_modules=cythonize("src/pyrle.pyx"), py_modules=["pyrle"])
