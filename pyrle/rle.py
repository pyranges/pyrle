from pyrle.src.rle import sub_rles, add_rles, mul_rles, div_rles_zeroes, div_rles_nonzeroes
from pyrle.src.coverage import _remove_dupes

import pandas as pd
import numpy as np

from tabulate import tabulate

class Rle:

    def __init__(self, runs, values):
        assert len(runs) == len(values)

        runs = np.array(runs, dtype=np.int)
        values = np.array(values, dtype=np.double)
        s = pd.Series(values, dtype=np.double)

        if (s.shift() == s).any():
            runs, values = _remove_dupes(runs, values, len(values))

        self.runs = runs
        self.values = values


    def __add__(self, other):

        runs, values = add_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)


    def __sub__(self, other):
        runs, values = sub_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    def __mul__(self, other):
        runs, values = mul_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if (other.values == 0).any():
            runs, values = div_rles_zeroes(self.runs, self.values, other.runs, other.values)
        else:
            runs, values = div_rles_nonzeroes(self.runs, self.values, other.runs, other.values)

        return Rle(runs, values)

    def __eq__(self, other):
        runs_equal = np.equal(self.runs, other.runs).all()
        values_equal = np.allclose(self.values, other.values)
        return runs_equal and values_equal

    def __str__(self):

        if len(self.runs) > 10:
            runs = [str(i) for i in self.runs[:5]] + \
                [" ... "] + [str(i) for i in self.runs[-5:]]
            values = ["{0:.3f}".format(i) for i in self.values[:5]] + \
                    [" ... "] + ["{0:.3f}".format(i) for i in self.values[-5:]]
        else:
            runs = [str(i) for i in self.runs]
            values = ["{0:.3f}".format(i) for i in self.values]

        df = pd.Series(values).to_frame().T

        df.columns = list(runs)
        df.index = ["Values"]
        df.index.name = "Runs"

        outstr = tabulate(df, tablefmt='psql', showindex=True, headers="keys")
        length = np.sum(self.runs)
        elements = len(self.runs)
        info = "\nRle of length {} containing {} elements".format(str(length), str(elements))

        return outstr + info


    def __repr__(self):

        return str(self)
