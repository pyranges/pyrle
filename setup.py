
# setup.py

from distutils.core import setup
# from distutils.extension import Extension
from setuptools import find_packages, Extension, Command
from Cython.Build import cythonize


# example_module = Extension('convolve', sources=['convolve.c'])
macros = [("CYTHON_TRACE", "1")]
macros = None

if macros:
    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

e1 = Extension("pyrle.src.rle", ["pyrle/src/rle.pyx"], define_macros = macros)
e2 = Extension("pyrle.src.coverage", ["pyrle/src/coverage.pyx"], define_macros = macros)
# ,
# e2 =
# print(type(e1))
extensions = [e1, e2]

install_requires = ["cython", "pandas", "tabulate", "numpy"]

setup(name='pyrle',
      version="0.0.1",
      packages=find_packages(),
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      author="Endre Bakken Stovner",
      author_email="endrebak85@gmail.com",
      url="https://github.com/endrebak/pyrle",
      license=["MIT"],
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Development Status :: 4 - Beta",
          "Environment :: Other Environment", "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: POSIX :: Linux",
          "Operating System :: MacOS :: MacOS X",
          "Topic :: Scientific/Engineering"
      ],
      # , "1")],
      # py_modules=["rle"],
      include_dirs=["."])


# import os
# import sys
# from setuptools import setup, find_packages
# # from Cython.Build import cythonize

# from pyranges.version import __version__
# install_requires = ["pandas", "tabulate"]
# # try:
# #     os.getenv("TRAVIS")
# #     install_requires.append("coveralls")
# # except:
# #     pass

# # if sys.version_info[0] == 2:
# #     install_requires.append("functools32")

# setup(
#     name="pyranges",
#     packages=find_packages(),

#     # scripts=["bin/featurefetch"],
#     version=__version__,
#     description="PyRanges for Python.",
#     author="Endre Bakken Stovner",
#     author_email="endrebak85@gmail.com",
#     url="http://github.com/endrebak/pyranges",
#     keywords=["Bioinformatics"],
#     license=["MIT"],
#     install_requires=install_requires,
#     classifiers=[
#         "Programming Language :: Python :: 2.7",
#         "Programming Language :: Python :: 3",
#         "Development Status :: 4 - Beta",
#         "Environment :: Other Environment", "Intended Audience :: Developers",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
#         "Operating System :: POSIX :: Linux",
#         "Operating System :: MacOS :: MacOS X",
#         "Topic :: Scientific/Engineering"
#     ],
#     long_description=("Pythonic Genomic Ranges."))
