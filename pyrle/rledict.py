# import sys
# sys.setrecursionlimit(150)

from pyrle.src.getitem import getitems

from pyrle import Rle

from numbers import Number
import pyrle.methods as m

from natsort import natsorted

import numpy as np

import logging

try:
    dummy = profile
except:
    profile = lambda x: x


def ray_initialized():
    def test_function():
        pass

    try:
        test_function = ray.remote(test_function)
    except Exception as e:
        if type(e) == NameError:
            return False

        raise e

    try:
        test_function.remote()
    except Exception as e:
        if "RayConnectionError" in str(type(e)):
            return True
        else:
            raise e


def get_multithreaded_funcs(function):

    if ray_initialized():
        get = ray.get
        function = ray.remote(function)
    else:
        get = lambda x: x
        function.remote = function

    return function, get


class PyRles():
    def __init__(self, ranges, stranded=False, value_col=None):

        # Construct PyRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges
            self.__dict__["stranded"] = True if len(list(
                ranges.keys())[0]) == 2 else False

        # Construct PyRles from ranges
        else:

            if stranded:
                grpby_keys = "Chromosome Strand".split()
            else:
                grpby_keys = "Chromosome"

            try:
                df = ranges.df
            except:
                df = ranges

            grpby = list(natsorted(df.groupby(grpby_keys)))

            m_coverage, get = get_multithreaded_funcs(m.coverage)

            _rles = {}
            kwargs = {"value_col": value_col}
            if stranded:
                for (c, s), cdf in grpby:
                    _rles[c, s] = m_coverage.remote(cdf, kwargs)
            else:
                s = None
                for k, cdf in grpby:
                    _rles[k] = m_coverage.remote(cdf, kwargs)

            _rles = {
                k: v
                for k, v in zip(_rles.keys(), get(list(_rles.values())))
            }

            self.rles = _rles

            self.__dict__["stranded"] = stranded

    def add(self, other):

        return m.binary_operation("add", self, other)

    def __add__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v + other for cs, v in self.items()})

        return m.binary_operation("add", self, other)

    def __radd__(self, other):

        return PyRles({cs: other + v for cs, v in self.items()})

    def sub(self, other):

        return m.binary_operation("sub", self, other)

    def __sub__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v - other for cs, v in self.items()})

        return m.binary_operation("sub", self, other)

    def __rsub__(self, other):

        return PyRles({cs: other - v for cs, v in self.items()})

    def mul(self, other):

        return m.binary_operation("mul", self, other)

    def __rmul__(self, other):

        return PyRles({cs: other * v for cs, v in self.items()})

    def __mul__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v * other for cs, v in self.items()})

        return m.binary_operation("mul", self, other)

    __rmul__ = __mul__

    def div(self, other):

        return m.binary_operation("div", self, other)

    def __rtruediv__(self, other):

        return PyRles({cs: other / v for cs, v in self.items()})

    def __truediv__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v / other for cs, v in self.items()})

        return m.binary_operation("div", self, other)

    def to_ranges(self):

        return m.to_ranges(self)

    @property
    def stranded(self):

        return len(self.keys()[0]) == 2

    def keys(self):

        return natsorted(list(self.rles.keys()))

    def values(self):

        return [self.rles[k] for k in natsorted(self.rles.keys())]

    def items(self):

        _items = list(self.rles.items())

        return natsorted(_items, key=lambda x: x[0])

    def add_pseudocounts(self, pseudo=0.01):

        for k, rle in self.items():

            rle.values.loc[rle.values == 0] = pseudo

    def __getitem__(self, key):

        key_is_string = isinstance(key, str)
        key_is_int = isinstance(key, int)
        import pyranges as pr

        if key_is_int:
            raise Exception("Integer indexing not allowed!")

        if key_is_string and self.stranded and key not in ["+", "-"]:
            plus = self.rles.get((key, "+"), Rle([1], [0]))
            rev = self.rles.get((key, "-"), Rle([1], [0]))

            return PyRles({(key, "+"): plus, (key, "-"): rev})

        # only return particular strand, but from all chromos
        elif key_is_string and self.stranded and key in ["+", "-"]:
            to_return = dict()
            for (c, s), rle in self.items():
                if s == key:
                    to_return[c, s] = rle

            if len(to_return) > 1:
                return PyRles(to_return)
            else:  # return just the rle
                return list(to_return.values())[0]

        elif key_is_string:

            return self.rles.get(key, Rle([1], [0]))

        elif "PyRanges" in str(type(key)): # hack to avoid isinstance(key, pr.PyRanges) so that we
                                           # do not need a dep on PyRanges in this library

            result = {}
            for k, v in key.dfs.items():

                if k not in self.rles:
                    continue

                v = v["Start End".split()].astype(np.long)
                result[k] = getitems(self.rles[k].runs, self.rles[k].values,
                                     v.Start.values, v.End.values)

            return result

        elif len(key) == 2:

            return self.rles.get(key, Rle([1], [0]))

        else:
            raise IndexError(
                "Must use chromosome, strand or (chromosome, strand) to get items from PyRles."
            )

    def __str__(self):

        keys = natsorted(self.rles.keys())
        stranded = True if len(list(keys)[0]) == 2 else False

        if not stranded:
            if len(keys) > 2:
                str_list = [
                    keys[0],
                    str(self.rles[keys[0]]), "...", keys[-1],
                    str(self.rles[keys[-1]]),
                    "Unstranded PyRles object with {} chromosomes.".format(
                        len(self.rles.keys()))
                ]
            elif len(keys) == 2:
                str_list = [
                    keys[0], "-" * len(keys[0]),
                    str(self.rles[keys[0]]), "", keys[-1], "-" * len(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "Unstranded PyRles object with {} chromosomes.".format(
                        len(self.rles.keys()))
                ]
            else:
                str_list = [
                    keys[0],
                    str(self.rles[keys[0]]),
                    "Unstranded PyRles object with {} chromosome.".format(
                        len(self.rles.keys()))
                ]

        else:
            if len(keys) > 2:
                str_list = [
                    " ".join(keys[0]),
                    str(self.rles[keys[0]]), "...", " ".join(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "PyRles object with {} chromosomes/strand pairs.".format(
                        len(self.rles.keys()))
                ]
            elif len(keys) == 2:
                str_list = [
                    " ".join(keys[0]), "-" * len(keys[0]),
                    str(self.rles[keys[0]]), "", " ".join(keys[-1]),
                    "-" * len(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "PyRles object with {} chromosomes/strand pairs.".format(
                        len(self.rles.keys()))
                ]
            else:
                str_list = [
                    " ".join(keys[0]),
                    str(self.rles[keys[0]]),
                    "PyRles object with {} chromosome/strand pairs.".format(
                        len(self.rles.keys()))
                ]

        outstr = "\n".join(str_list)

        return outstr

    def __eq__(self, other):

        if not self.rles.keys() == other.rles.keys():
            return False

        for c in self.rles.keys():

            if self.rles[c] != other.rles[c]:
                return False

        return True

    def __repr__(self):

        return str(self)

    def numbers_only(self):

        return PyRles({k: v.numbers_only() for k, v in self.items()})

    def defragment(self, numbers_only=False):

        if not numbers_only:
            d = {k: v.defragment() for k, v in self.items()}
        else:
            d = {k: v.numbers_only().defragment() for k, v in self.items()}

        return PyRles(d)


if __name__ == "__main__":

    # Must turn on macros in setup.py for line tracing to work
    "kernprof -l pyrle/rledict.py && python -m line_profiler coverage.py.lprof"

    from time import time
    import datetime

    import pandas as pd

    test_file = "/mnt/scratch/endrebak/genomes/chip/UCSD.Aorta.Input.STL002.bed.gz"

    nrows = None
    df = pd.read_table(
        test_file,
        sep="\t",
        usecols=[0, 1, 2, 5],
        header=None,
        names="Chromosome Start End Strand".split(),
        nrows=nrows)

    print("Done reading")
    start = time()

    result = PyRles(df, stranded=True)

    end = time()
    total = end - start

    total_dt = datetime.datetime.fromtimestamp(total)

    minutes_seconds = total_dt.strftime('%M\t%S\n')

    print(result)
    print(minutes_seconds)
