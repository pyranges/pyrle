import pandas as pd
import numpy as np

from pyrle import Rle
from pyrle.rledict import GRles
from pyrle.src.coverage import _coverage

from natsort import natsorted

from sys import stderr

from joblib import Parallel, delayed

try:
    dummy = profile
except:
    profile = lambda x: x



def chromosomes_in_both_self_other(self, other):

    chromosomes_in_both = set(self.rles.keys()).intersection(other.rles.keys())
    chromosomes_in_self_not_other = set(self.rles.keys()) - set(other.rles.keys())
    chromosomes_in_other_not_self = set(other.rles.keys()) - set(self.rles.keys())

    if chromosomes_in_self_not_other:
        print(", ".join(natsorted(chromosomes_in_self_not_other)) + " missing from other.", file=stderr)

    if chromosomes_in_other_not_self:
        print(", ".join(natsorted(chromosomes_in_other_not_self)) + " missing from self.", file=stderr)

    return chromosomes_in_both, chromosomes_in_self_not_other, chromosomes_in_other_not_self


def __add(self, other):

    return self + other


def binary_operation(operation, self, other, n_jobs=1):

    func = {"div": __div, "mul": __mul, "add": __add, "sub": __sub}[operation]

    chromosomes_in_both, chromosomes_in_self_not_other, chromosomes_in_other_not_self = chromosomes_in_both_self_other(self, other)

    _rles = []
    for c in chromosomes_in_both:
        _rles.append(func(self.rles[c], other.rles[c]))


    rles = {c: r for c, r in zip(chromosomes_in_both, _rles)}

    for c in chromosomes_in_self_not_other:
        rles[c] = self.rles[c]

    for c in chromosomes_in_other_not_self:
        rles[c] = other.rles[c]

    return GRles(rles)


def __sub(self, other):

    return self - other

def __div(self, other):

    return self / other

def __mul(self, other):

    return self * other



def coverage(ranges, value_col=None):

    try:
        df = ranges.df
    except:
        df = ranges

    if value_col:
        values = df[value_col].astype(np.float64).values
    else:
        values = np.ones(len(df))

    # ndf = df["Start End Score".split()].sort_values("Start End".split())
    # else ValueError: buffer source array is read-only
    new_starts = df.Start.copy().values
    new_ends = df.End.copy().values

    runs, values = _coverage(new_starts, new_ends, values,
                             len(df))

    return Rle(runs, values)




def to_ranges(grles):

    from pyranges import PyRanges

    dfs = []
    if grles.stranded:

        for (chromosome, strand), rle in grles.items():
            starts, ends, values = _to_ranges(rle)
            df = pd.concat([pd.Series(r) for r in [starts, ends, values]], axis=1)
            df.columns = "Start End Score".split()
            df.insert(0, "Chromosome", chromosome)
            df.insert(df.shape[1], "Strand", strand)
            dfs.append(df)
    else:

        for chromosome, rle in grles.items():
            starts, ends, values = _to_ranges(rle)
            df = pd.concat([pd.Series(r) for r in [starts, ends, values]], axis=1)
            df.columns = "Start End Score".split()
            df.insert(0, "Chromosome", chromosome)
            dfs.append(df)

    return PyRanges(pd.concat(dfs))


def _to_ranges(rle):

    runs = pd.Series(rle.runs)
    starts = pd.Series([0] + list(runs)).cumsum()

    ends = starts + runs

    values = pd.Series(rle.values)

    start_idx = values[values.shift(-1) != values].index
    end_idx = values[values.shift(1) != values].index

    starts = starts.loc[start_idx]
    ends = ends.loc[end_idx]
    values = values[start_idx].reset_index(drop=True)

    return starts.astype(int).reset_index(drop=True), ends.astype(int).reset_index(drop=True), values
