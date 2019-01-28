from pyrle.src.rle import sub_rles, add_rles, mul_rles, div_rles_zeroes, div_rles_nonzeroes
from pyrle.src.coverage import _remove_dupes
from pyrle.src.getitem import getitem, getlocs, getitems

import pandas as pd
import numpy as np

from tabulate import tabulate

from numbers import Number


def make_rles_equal_length(func):

    def extension(self, other, **kwargs):

        if not isinstance(other, Number):
            ls = np.sum(self.runs)
            lo = np.sum(other.runs)

            if ls > lo:
                new_runs = np.append(other.runs, ls - lo)
                new_values = np.append(other.values, 0)
                other = Rle(new_runs, new_values)
            elif lo > ls:
                new_runs = np.append(self.runs, lo - ls)
                new_values = np.append(self.values, 0)
                self = Rle(new_runs, new_values)

            return func(self, other)
        else:
            return func(self, other)

    return extension



class Rle:

    def __init__(self, runs, values):
        assert len(runs) == len(values)

        runs = np.copy(runs)
        values = np.copy(values)

        runs = np.array(runs, dtype=np.int)
        values = np.array(values, dtype=np.double)
        s = pd.Series(values, dtype=np.double)

        zero_length_runs = runs == 0
        if np.any(zero_length_runs):
            runs = runs[~zero_length_runs]
            values = values[~zero_length_runs]

        if (np.isclose(s.shift(), s, equal_nan=True)).any() and len(s) > 1:
            runs, values = _remove_dupes(runs, values, len(values))

        self.runs = np.copy(runs)
        self.values = np.copy(values)

    def to_csv(self, **kwargs):

        if not kwargs.get("path_or_buf"):
            print(pd.DataFrame(data={"Runs": self.runs, "Values": self.values})["Runs Values".split()].to_csv(**kwargs))
        else:
            pd.DataFrame(data={"Runs": self.runs, "Values": self.values})["Runs Values".split()].to_csv(**kwargs)

    def __len__(self):
        return len(self.runs)

    def __radd__(self, other):

        return Rle(self.runs, self.values + other)

    def __rmul__(self, other):

        return Rle(self.runs, self.values * other)

    def __rsub__(self, other):

        return Rle(self.runs, other - self.values)

    def __rtruediv__(self, other):

        return Rle(self.runs, other / self.values)

    @make_rles_equal_length
    def __add__(self, other):

        if isinstance(other, Number):
            return Rle(self.runs, self.values + other)

        runs, values = add_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)


    @make_rles_equal_length
    def __sub__(self, other):

        if isinstance(other, Number):
            return Rle(self.runs, self.values - other)

        runs, values = sub_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    @make_rles_equal_length
    def __mul__(self, other):

        if isinstance(other, Number):
            return Rle(self.runs, self.values * other)

        runs, values = mul_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    __rmul__ = __mul__

    @make_rles_equal_length
    def __truediv__(self, other):

        if isinstance(other, Number):
            return Rle(self.runs, self.values / other)

        if (other.values == 0).any() or np.sum(other.runs) < np.sum(self.runs):
            runs, values = div_rles_zeroes(self.runs, self.values, other.runs, other.values)
        else:
            runs, values = div_rles_nonzeroes(self.runs, self.values, other.runs, other.values)

        return Rle(runs, values)

    def __eq__(self, other):

        if len(self.runs) != len(other.runs):
            return False

        runs_equal = np.equal(self.runs, other.runs).all()
        values_equal = np.allclose(self.values, other.values)
        return runs_equal and values_equal

    def __str__(self):

        if len(self.runs) > 10:
            runs = [str(i) for i in self.runs[:5]] + \
                [" ... "] + [str(i) for i in self.runs[-5:]]
            values = ["{}".format(i) for i in self.values[:5]] + \
                    [" ... "] + ["{}".format(i) for i in self.values[-5:]]
        else:
            runs = [str(i) for i in self.runs]
            values = ["{}".format(i) for i in self.values]

        df = pd.Series(values).to_frame().T

        df.columns = list(runs)
        df.index = ["Values"]
        df.index.name = "Runs"

        outstr = tabulate(df, tablefmt='psql', showindex=True, headers="keys", disable_numparse=True)
        length = np.sum(self.runs)
        elements = len(self.runs)
        info = "\nRle of length {} containing {} elements".format(str(length), str(elements))

        return outstr + info

    def __getitem__(self, val):

        if isinstance(val, int):
            values = getlocs(self.runs, self.values, np.array([val], dtype=np.long))
            return values[0]
        elif isinstance(val, slice):
            end = val.stop or np.sum(self.runs)
            start = val.start or 0
            runs, values = getitem(self.runs, self.values, start, end)
            return Rle(runs, values)
        elif isinstance(val, pd.DataFrame):
            val = val["Start End".split()].astype(np.long)
            values = getitems(self.runs, self.values, val.Start.values, val.End.values)
            return [Rle(r, v) for r, v in values]
        else:
            locs = np.sort(np.array(val, dtype=np.long))
            values = getlocs(self.runs, self.values, locs)
            return values


    def defragment(self):

        runs, values = _remove_dupes(self.runs, self.values, len(self))
        values[values == -0] = 0
        return Rle(runs, values)

    def numbers_only(self):

        return Rle(self.runs, np.nan_to_num(self.values)).defragment()

    def copy(self):

        return Rle(np.copy(self.runs), np.copy(self.values))

    def __repr__(self):

        return str(self)
