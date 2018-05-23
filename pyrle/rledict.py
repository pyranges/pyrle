
from pyrle import Rle

from joblib import Parallel, delayed

from pyrle import methods as m

from natsort import natsorted

try:
    dummy = profile
except:
    profile = lambda x: x


class PyRles():

    # @profile
    def __init__(self, ranges, n_jobs=1, stranded=False, value_col=None):

        # Construct PyRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges
            self.__dict__["stranded"] = True if len(list(ranges.keys())[0]) == 2 else False
        # Construct PyRles from ranges
        elif not stranded:

            try:
                df = ranges.df
            except:
                df = ranges

            grpby = list(df.groupby("Chromosome"))

            if n_jobs > 1:
                _rles = Parallel(n_jobs=n_jobs)(delayed(m.coverage)(cdf, value_col=value_col) for _, cdf in grpby)
            else:
                _rles = []
                for _, cdf in grpby:
                    cv = m.coverage(cdf, value_col=value_col)
                    _rles.append(cv)

            self.rles = {c: r for c, r in zip([c for c, _ in grpby], _rles)}
            self.__dict__["stranded"] = False

        else:

            try:
                df = ranges.df
            except:
                df = ranges

            cs = df["Chromosome Strand".split()].drop_duplicates()
            cs = list(zip(cs.Chromosome.tolist(), cs.Strand.tolist()))

            grpby = df.groupby("Chromosome Strand".split())

            if n_jobs > 1:
                _rles = Parallel(n_jobs=n_jobs)(delayed(m.coverage)(csdf, value_col=value_col) for cs, csdf in grpby)
            else:
                _rles = []
                for cs, csdf in grpby:
                    _rles.append(m.coverage(csdf, value_col=value_col))

            self.rles = {cs: r for cs, r in zip([cs for cs, _ in grpby], _rles)}
            self.__dict__["stranded"] = True

    def add(self, other, n_jobs=1):

        return m.binary_operation("add", self, other, n_jobs=n_jobs)

    def __add__(self, other, n_jobs=1):

        return m.binary_operation("add", self, other)

    def sub(self, other, n_jobs=1):

        return m.binary_operation("sub", self, other, n_jobs=n_jobs)

    def __sub__(self, other):

        return m.binary_operation("sub", self, other)

    def mul(self, other, n_jobs=1):

        return m.binary_operation("mul", self, other, n_jobs=n_jobs)

    def __mul__(self, other):

        return m.binary_operation("mul", self, other)

    __rmul__ = __mul__

    def div(self, other, n_jobs=1):

        return m.binary_operation("div", self, other, n_jobs=n_jobs)

    def __div__(self, other):

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

        elif key_is_string and self.stranded and key in ["+", "-"]:
            to_return = dict()
            for (c, s), rle in self.items():
                if s == key:
                    to_return[c, s] = rle

            return PyRles(to_return)

        elif key_is_string:

            return self.rles[key]

        elif len(key) == 2:

            return PyRles({key: self.rles[key]})

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

    df = pd.read_table(test_file, sep="\t", usecols=[0, 1, 2, 5], header=None,
                       names="Chromosome Start End Strand".split(), nrows=None)


    print("Done reading")
    start = time()

    result = PyRles(df, n_jobs=25)

    end = time()
    total = end - start

    total_dt = datetime.datetime.fromtimestamp(total)

    minutes_seconds = total_dt.strftime('%M\t%S\n')

    print(result)
    print(minutes_seconds)
