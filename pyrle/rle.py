from pyrle.src.rle import sub_rles, add_rles, mul_rles, div_rles_zeroes, div_rles_nonzeroes
from pyrle.src.coverage import _remove_dupes
from pyrle.src.getitem import getitem, getlocs, getitems

import pandas as pd
import numpy as np
import shutil

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

import numpy as np


def find_runs(x):
    """Find runs of consecutive items in an array.

    Author: Alistair Miles
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = np.array(x[loc_run_start], dtype=np.double)

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_lengths


class Rle:

    def __init__(self, runs=None, values=None):

        if values is not None and runs is not None:

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

        elif runs is not None:
            values = runs
            self.values, self.runs = find_runs(values)

        else:
            self.runs = np.array([], dtype=np.int)
            self.values = np.array([], dtype=np.double)


    def to_csv(self, **kwargs):

        if not kwargs.get("path_or_buf"):
            print(pd.DataFrame(data={"Runs": self.runs, "Values": self.values})["Runs Values".split()].to_csv(**kwargs))
        else:
            pd.DataFrame(data={"Runs": self.runs, "Values": self.values})["Runs Values".split()].to_csv(**kwargs)

    def __len__(self):
        return len(self.runs)

    def __neg__(self):
        self = self.copy()
        self.values = -self.values
        return self

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

    def apply_values(self, f, defragment=True):
        self = self.copy()
        self.values = f(self.values)
        if defragment:
            self = self.defragment()
        return self

    def apply_runs(self, f, defragment=True):
        self = self.copy()
        self.runs = f(self.runs)
        if defragment:
            self = self.defragment()
        return self

    def apply(self, f, defragment=True):
        self = self.copy()
        self = f(self)
        if defragment:
            self = self.defragment()
        return self


    def __str__(self):

        terminal_width = shutil.get_terminal_size().columns

        entries = min(len(self.runs), 10)
        half_entries = int(entries/2)

        start_runs, end_runs = [str(i) for i in self.runs[:half_entries]], [str(i) for i in self.runs[-half_entries:]]
        start_values, end_values = [str(i) for i in self.values[:half_entries]], [str(i) for i in self.values[-half_entries:]]

        if entries < len(self.runs):
            runs = start_runs + ["..."] + end_runs
            values = start_values + ["..."] + end_values
        else:
            runs, values = self.runs, self.values

        df = pd.Series(values).to_frame().T

        df.columns = list(runs)
        df.index = ["Values"]
        df.index.name = "Runs"

        outstr = tabulate(df, tablefmt='psql', showindex=True, headers="keys", disable_numparse=True)

        while len(outstr.split("\n", 1)[0]) > terminal_width:
            half_entries -= 1

            runs = start_runs[:half_entries] + ["..."] + end_runs[-half_entries:]
            values = start_values[:half_entries] + ["..."] + end_values[-half_entries:]

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
            intype = val.dtypes["Start"]
            val = val["Start End".split()].astype(np.long)
            ids, starts, ends, runs, values = getitems(self.runs, self.values,
                                            val.Start.values, val.End.values)

            df = pd.DataFrame({"Start": starts,
                                "End": ends,
                                "ID": ids,
                                "Run": runs,
                                "Value": values}).astype({"Start": intype, "End": intype})
            # val = val["Start End".split()].astype(np.long)
            # values = getitems(self.runs, self.values, val.Start.values, val.End.values)
            return df
        elif "PyRanges" in str(type(val)): # hack to avoid isinstance(key, pr.PyRanges) so that we
                                           # do not need a dep on PyRanges in this library
            import pyranges as pr
            val = val.drop().df
            if val.empty:
                return pd.DataFrame(columns="Chromosome Start End ID Run Value".split())

            chromosome = val.Chromosome.iloc[0]

            intype = val.dtypes["Start"]

            if "Strand" in val:
                strand = val.Strand.iloc[0]
            else:
                strand = None

            val = val["Start End".split()].astype(np.long)
            ids, starts, ends, runs, values = getitems(self.runs, self.values,
                                            val.Start.values, val.End.values)

            df = pd.DataFrame({"Chromosome": chromosome,
                               "Start": starts,
                                "End": ends,
                                "ID": ids,
                                "Run": runs,
                                "Value": values}).astype({"Start": intype, "End": intype})

            if strand:
                df.insert(3, "Strand", strand)
            # val = val["Start End".split()].astype(np.long)
            # values = getitems(self.runs, self.values, val.Start.values, val.End.values)
            return pr.PyRanges(df)
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

    @property
    def length(self):
        return np.sum(self.runs)

    # def shorten(self, value):
    #     if 
    # def lengthen

    def shift(self, dist, fill=0, fill_end=True):
        # TODO: missing remove_end for dist > 0 shifts

        self = self.copy()
        if dist > 0:
            if self.values[0] == fill:
                self.runs[0] += dist
            else:
                self.values = np.r_[fill, self.values]
                self.runs = np.r_[dist, self.runs]
        elif dist < 0:
            dist = -dist # remember dist is negative
            if dist < self.runs[0]:
                self.runs[0] -= dist 
            else:
                cs = np.cumsum(self.runs)
                ix = np.argmax(cs > dist)
                leftover = (np.sum(self.runs[:ix]) - dist)
                self = Rle(self.runs[ix:], self.values[ix:])
                self.runs[0] += leftover

                if self.runs[0] < 0:
                    self = Rle([], [])

                if fill_end:
                    if self.values[-1] == fill:
                        self.runs[-1] += dist
                    else:
                        self.values = np.r_[self.values, fill]
                        self.runs = np.r_[self.runs, dist]

        return self

    def __repr__(self):

        return str(self)
