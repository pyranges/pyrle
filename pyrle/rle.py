from pyrle.src.rle import sub_rles, add_rles, mul_rles, div_rles_zeroes, div_rles_nonzeroes
from pyrle.src.coverage import _remove_dupes

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

        # Why shorten Rles with length zero here? Better to just drop those entries
        # with numpy?
        # i = len(runs) - 1
        # while i >= 0:
        #     if runs[i] != 0:
        #         break
        #     else:
        #         i -= 1

        # if i != len(runs) - 1:
        #     runs = runs[:i]
        #     values = values[:i]

        runs = np.array(runs, dtype=np.int)
        values = np.array(values, dtype=np.double)
        s = pd.Series(values, dtype=np.double)

        zero_length_runs = runs == 0
        if np.any(zero_length_runs):
            runs = runs[~zero_length_runs]
            values = values[~zero_length_runs]

        # shifted = s.shift()
        # if np.isclose(s.values[0], np.nan, equal_nan=True):
        #     if len(s) > 1 and not np.isclose(s.values[1], 1):
        #         shifted.values[0] = 1
        #     elif len(s) > 1:
        #         shifted.values[0] = 0


        # print("-----------" * 5)
        #print("nodup runs, values", runs, values)
        if (np.isclose(s.shift(), s, equal_nan=True)).any() and len(s) > 1:
            #print("runs, values", runs, values)
            runs, values = _remove_dupes(runs, values, len(values))
            #print("runs, values", runs, values)

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

        outstr = tabulate(df, tablefmt='psql', showindex=True, headers="keys")
        length = np.sum(self.runs)
        elements = len(self.runs)
        info = "\nRle of length {} containing {} elements".format(str(length), str(elements))

        return outstr + info


    def __repr__(self):

        return str(self)
