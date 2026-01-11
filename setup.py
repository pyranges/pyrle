from setuptools import Extension, setup
import sys

def get_extensions():
    if "clean" in sys.argv:
        return []
    from Cython.Build import cythonize
    e1 = Extension("pyrle.src.rle", ["pyrle/src/rle.pyx"])
    e2 = Extension("pyrle.src.coverage", ["pyrle/src/coverage.pyx"])
    e3 = Extension("pyrle.src.getitem", ["pyrle/src/getitem.pyx"])

    extensions = [e1, e2, e3]
    return cythonize(extensions, language_level=3)

setup(ext_modules=get_extensions())
