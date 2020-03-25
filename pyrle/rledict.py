"""Data structure for collection of genomic Rles."""

from pyrle.src.getitem import getitems

from pyrle import Rle

from numbers import Number
import pyrle.methods as m

from natsort import natsorted

import numpy as np

import logging


def get_multithreaded_funcs(function, nb_cpu):

    if nb_cpu > 1:
        import ray
        get = ray.get
        function = ray.remote(function)
    else:
        get = lambda x: x
        function.remote = function

    return function, get


class PyRles():

    """Data structure to represent and manipulate a genomic collection of Rles.

    An Rle contains two vectors, one with runs (int) and one with values
    (double).

    Operations between Rles act as if it was a regular vector.

    There are three ways to build an Rle: from a vector of runs or a vector of
    values, or a vector of values.

    Parameters
    ----------
    runs : array-like

        Run lengths.

    values : array-like

        Run values.

    See Also
    --------
    pyrle.rledict.PyRles : genomic collection of Rles

    Examples
    --------

    >>> r = Rle([1, 2, 1, 5], [0, 2.1, 3, 4])
    >>> r
    +--------+-----+-----+-----+-----+
    | Runs   | 1   | 2   | 1   | 5   |
    |--------+-----+-----+-----+-----|
    | Values | 0.0 | 2.1 | 3.0 | 4.0 |
    +--------+-----+-----+-----+-----+
    Rle of length 9 containing 4 elements (avg. length 2.25)

    >>> r2 = Rle([1, 1, 1, 0, 0, 2, 2, 3, 4, 2])
    >>> r2
    +--------+-----+-----+-----+-----+-----+-----+
    | Runs   | 3   | 2   | 2   | 1   | 1   | 1   |
    |--------+-----+-----+-----+-----+-----+-----|
    | Values | 1.0 | 0.0 | 2.0 | 3.0 | 4.0 | 2.0 |
    +--------+-----+-----+-----+-----+-----+-----+
    Rle of length 10 containing 6 elements (avg. length 1.667)

    When one Rle is longer than the other, the shorter is extended with zeros:

    >>> r - r2
    +--------+------+-----+-----+-----+-----+-----+-----+------+
    | Runs   | 1    | 2   | 1   | 1   | 2   | 1   | 1   | 1    |
    |--------+------+-----+-----+-----+-----+-----+-----+------|
    | Values | -1.0 | 1.1 | 3.0 | 4.0 | 2.0 | 1.0 | 0.0 | -2.0 |
    +--------+------+-----+-----+-----+-----+-----+-----+------+
    Rle of length 10 containing 8 elements (avg. length 1.25)

    Scalar operations work with Rles:

    >>> r * 5
    +--------+-----+------+------+------+
    | Runs   | 1   | 2    | 1    | 5    |
    |--------+-----+------+------+------|
    | Values | 0.0 | 10.5 | 15.0 | 20.0 |
    +--------+-----+------+------+------+
    Rle of length 9 containing 4 elements (avg. length 2.25)

    """

    def __init__(self, ranges=None, stranded=False, value_col=None, nb_cpu=1):

        # Construct PyRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges
            self.__dict__["stranded"] = True if len(list(
                ranges.keys())[0]) == 2 else False
        elif ranges is None:
            self.rles = {}

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

            if nb_cpu > 1:
                import ray
                with m.suppress_stdout_stderr():
                    ray.init(num_cpus=nb_cpu)

            m_coverage, get = get_multithreaded_funcs(m.coverage, nb_cpu)

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

            if nb_cpu > 1:
                ray.shutdown()

            self.rles = _rles

            self.__dict__["stranded"] = stranded

    def __iter__(self):

        return iter(self.rles.items())


    def apply_values(self, f, defragment=True):

        new_rles = {}

        for k, r in self:

            new_rle = r.copy()
            new_rle.values = f(new_rle.values).astype(np.double)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return PyRles(new_rles)

    def apply_runs(self, f, defragment=True):

        new_rles = {}

        for k, r in self:

            new_rle = r.copy()
            new_rle.runs = f(new_rle.runs).astype(np.int)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return PyRles(new_rles)


    def apply(self, f, defragment=True):

        new_rles = {}

        for k, r in self:

            new_rle = r.copy()
            # new_rle.runs = f(new_rle.runs).astype(np.int)
            new_rle = f(new_rle)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return PyRles(new_rles)


    def add(self, other, nb_cpu=1):

        return m.binary_operation("add", self, other, nb_cpu)

    def __add__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v + other for cs, v in self.items()})

        return m.binary_operation("add", self, other)

    def __radd__(self, other):

        return PyRles({cs: other + v for cs, v in self.items()})

    def sub(self, other, nb_cpu=1):

        return m.binary_operation("sub", self, other, nb_cpu)

    def __sub__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v - other for cs, v in self.items()})

        return m.binary_operation("sub", self, other)

    def __rsub__(self, other):

        return PyRles({cs: other - v for cs, v in self.items()})

    def mul(self, other, nb_cpu=1):

        return m.binary_operation("mul", self, other, nb_cpu)

    def __rmul__(self, other):

        return PyRles({cs: other * v for cs, v in self.items()})

    def __mul__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v * other for cs, v in self.items()})

        return m.binary_operation("mul", self, other)

    __rmul__ = __mul__

    def div(self, other, nb_cpu=1):

        return m.binary_operation("div", self, other, nb_cpu)

    def __rtruediv__(self, other):

        return PyRles({cs: other / v for cs, v in self.items()})

    def __truediv__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v / other for cs, v in self.items()})

        return m.binary_operation("div", self, other)

    def to_ranges(self, dtype=np.int32):

        assert dtype in [np.int32, np.int64]

        return m.to_ranges(self).apply(lambda df: df.astype({"Start": dtype, "End": dtype}))

    def __len__(self):
        return len(self.rles)



    @property
    def stranded(self):

        if len(self) == 0:
            return True

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

    def copy(self):
        d = {}
        for k, r in self:
            d[k] = r.copy()

        return PyRles(d)

    def shift(self, distance):
        return self.apply(lambda r: r.shift(distance))

    def __setitem__(self, key, item):

        self.rles[key] = item

    def __getitem__(self, key):

        key_is_string = isinstance(key, str)
        key_is_int = isinstance(key, int)

        if key_is_int:
            raise Exception("Integer indexing not allowed!")

        if key_is_string and self.stranded and key not in ["+", "-"]:
            plus = self.rles.get((key, "+"), Rle())
            rev = self.rles.get((key, "-"), Rle())

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

            return self.rles.get(key, Rle())

        elif "PyRanges" in str(type(key)): # hack to avoid isinstance(key, pr.PyRanges) so that we
                                           # do not need a dep on PyRanges in this library

            import pyranges as pr
            import pandas as pd

            from pyrle.rle import find_runs

            if not len(key):
                return pd.DataFrame(columns="Chromosome Start End ID Run Value".split())

            result = {}
            for k, v in key.dfs.items():

                if k not in self.rles:
                    continue


                v = v["Start End".split()].astype(np.long)
                ids, starts, ends, runs, values = getitems(self.rles[k].runs, self.rles[k].values,
                                                           v.Start.values, v.End.values)

                df = pd.DataFrame({"Start": starts, "End": ends, "ID": ids, "Run": runs, "Value": values})

                if isinstance(k, tuple):
                    df.insert(0, "Chromosome", k[0])
                    df.insert(df.shape[1], "Strand", k[1])
                else:
                    df.insert(0, "Chromosome", k)

                result[k] = df

            return pr.PyRanges(result)

        elif len(key) == 2:

            return self.rles.get(key, Rle([1], [0]))

        else:
            raise IndexError(
                "Must use chromosome, strand or (chromosome, strand) to get items from PyRles."
            )


    @property
    def chromosomes(self):

        cs = []

        for k in self.rles:
            if isinstance(k, tuple):
                cs.append(k[0])
            else:
                cs.append(k)

        return natsorted(set(cs))


    def __str__(self):

        if not self.rles:
            return "Empty PyRles."

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

    def to_table(self):

        import pandas as pd
        dfs = []
        for k, r in self.rles.items():
            df = pd.DataFrame(data={"Runs": r.runs, "Values": r.values})
            if self.stranded:
                df.insert(0, "Chromosome", k[0])
                df.insert(1, "Strand", k[1])
            else:
                df.insert(0, "Chromosome", k)

            dfs.append(df)

        return pd.concat(dfs)

    def to_csv(self, f, sep="\t"):

        self.to_table().to_csv(f, sep=sep, index=False)

    def make_strands_same_length(self, fill_value=0):

        self = self.copy()

        if not self.stranded:
            return self

        for c in self.chromosomes:
            p = self[c]["+"]
            n = self[c]["-"]
            pl = p.length
            nl = n.length
            diff = abs(pl - nl)

            if pl > nl:
                if n.values[-1] == fill_value:
                    n.runs[-1] += diff
                else:
                    n.runs = np.r_[n.runs, diff]
                    n.values = np.r_[n.values, fill_value]
            elif pl < nl:
                if p.values[-1] == fill_value:
                    p.runs[-1] += diff
                else:
                    p.runs = np.r_[p.runs, diff]
                    p.values = np.r_[p.values, fill_value]

        return self


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
