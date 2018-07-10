from pyrle import Rle

from numbers import Number

from pyrle import methods as m

from natsort import natsorted

try:
    dummy = profile
except:
    profile = lambda x: x


class PyRles():

    def __init__(self, ranges, stranded=False, value_col=None, nb_cpu=1):

        # Construct PyRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges
            self.__dict__["stranded"] = True if len(list(ranges.keys())[0]) == 2 else False

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

            _rles = {}
            for key, cdf in grpby:
                _rles[key] = m.coverage(cdf, value_col=value_col)

            self.rles = _rles

            self.__dict__["stranded"] = stranded


    def add(self, other, nb_cpu=1):

        return m.binary_operation("add", self, other, nb_cpu=nb_cpu)

    def __add__(self, other, nb_cpu=1):

        if isinstance(other, Number):
            return PyRles({cs: v + other for cs, v in self.items()})

        return m.binary_operation("add", self, other)

    def __radd__(self, other):

        return PyRles({cs: other + v for cs, v in self.items()})

    def sub(self, other, nb_cpu=1):

        return m.binary_operation("sub", self, other, nb_cpu=nb_cpu)

    def __sub__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v - other for cs, v in self.items()})

        return m.binary_operation("sub", self, other)

    def __rsub__(self, other):

        return PyRles({cs: other - v for cs, v in self.items()})

    def mul(self, other, nb_cpu=1):

        return m.binary_operation("mul", self, other, nb_cpu=nb_cpu)

    def __rmul__(self, other):

        return PyRles({cs: other * v for cs, v in self.items()})

    def __mul__(self, other):

        if isinstance(other, Number):
            return PyRles({cs: v * other for cs, v in self.items()})

        return m.binary_operation("mul", self, other)

    __rmul__ = __mul__

    def div(self, other, nb_cpu=1):

        return m.binary_operation("div", self, other, nb_cpu=nb_cpu)

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

        return natsorted(list(self.rles.values()))

    def items(self):

        return natsorted(list(self.rles.items()))

    def add_pseudocounts(self, pseudo=0.01):

        for k, rle in self.items():

            rle.values.loc[rle.values == 0] = pseudo

    def __getitem__(self, key):

        key_is_string = isinstance(key, str)

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

            return PyRles(to_return)

        elif key_is_string:

            return self.rles[key]

        elif len(key) == 2:

            return self.rles[key]

        else:
            raise IndexError("Must use chromosome, strand or (chromosome, strand) to get items from PyRles.")


    def __str__(self):

        keys = natsorted(self.rles.keys())
        stranded = True if len(list(keys)[0]) == 2 else False

        if not stranded:
            if len(keys) > 2:
                str_list = [keys[0],
                    str(self.rles[keys[0]]),
                    "...",
                    keys[-1],
                    str(self.rles[keys[-1]]),
                    "Unstranded PyRles object with {} chromosomes.".format(len(self.rles.keys()))]
            elif len(keys) == 2:
                str_list = [keys[0],
                            "-" * len(keys[0]),
                            str(self.rles[keys[0]]),
                            "",
                            keys[-1],
                            "-" * len(keys[-1]),
                            str(self.rles[keys[-1]]),
                            "Unstranded PyRles object with {} chromosomes.".format(len(self.rles.keys()))]
            else:
                str_list = [keys[0],
                            str(self.rles[keys[0]]),
                            "Unstranded PyRles object with {} chromosome.".format(len(self.rles.keys()))]

        else:
            if len(keys) > 2:
                str_list = [" ".join(keys[0]),
                    str(self.rles[keys[0]]),
                    "...",
                    " ".join(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "PyRles object with {} chromosomes/strand pairs.".format(len(self.rles.keys()))]
            elif len(keys) == 2:
                str_list = [" ".join(keys[0]),
                            "-" * len(keys[0]),
                            str(self.rles[keys[0]]),
                            "",
                            " ".join(keys[-1]),
                            "-" * len(keys[-1]),
                            str(self.rles[keys[-1]]),
                            "PyRles object with {} chromosomes/strand pairs.".format(len(self.rles.keys()))]
            else:
                str_list = [" ".join(keys[0]),
                            str(self.rles[keys[0]]),
                            "PyRles object with {} chromosome/strand pairs.".format(len(self.rles.keys()))]

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


if __name__ == "__main__":

    # Must turn on macros in setup.py for line tracing to work
    "kernprof -l pyrle/rledict.py && python -m line_profiler coverage.py.lprof"

    from time import time
    import datetime

    import pandas as pd

    test_file = "/mnt/scratch/endrebak/genomes/chip/UCSD.Aorta.Input.STL002.bed.gz"

    nrows = None
    df = pd.read_table(test_file, sep="\t", usecols=[0, 1, 2, 5], header=None,
                       names="Chromosome Start End Strand".split(), nrows=nrows)


    print("Done reading")
    start = time()

    result = PyRles(df, stranded=True)

    end = time()
    total = end - start

    total_dt = datetime.datetime.fromtimestamp(total)

    minutes_seconds = total_dt.strftime('%M\t%S\n')

    print(result)
    print(minutes_seconds)
