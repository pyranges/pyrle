import pandas as pd
import numpy as np

from pyrle import Rle
from pyrle.rledict import GRles

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


# @profile
# def coverage(ranges, value_col=None):

#     try:
#         df = ranges.df
#     except:
#         df = ranges
#         df = df.reset_index(drop=True)

#     if value_col:
#         starts = df[["Start"] + [value_col]]
#         ends = df[["End"] + [value_col]]
#         # spurious warning
#         pd.options.mode.chained_assignment = None
#         ends.loc[:, value_col] = ends[value_col] * - 1
#         pd.options.mode.chained_assignment = "warn"
#         columns = ["Position"] + [value_col]
#     else:
#         starts = pd.concat([df.Start, pd.Series(np.ones(len(df)))], axis=1)
#         ends = pd.concat([df.End, -1 * pd.Series(np.ones(len(df)))], axis=1)
#         columns = "Position Value".split()
#         value_col = "Value"

#     starts.columns, ends.columns = columns, columns
#     runs = pd.concat([starts, ends], ignore_index=True)
#     print("\n")
#     print("Before sort value\n", runs.head())
#     runs = runs.sort_values("Position")
#     print("Before groupby sum\n", runs.head())
#     values = runs.groupby("Position").sum()
#     print("Before reset index value col: values", values.head())
#     values = values.reset_index()[value_col]
#     print("Before drop duplicates\n", runs.head())
#     runs = runs.drop_duplicates("Position")
#     print("After drop duplicates", runs.head())
#     first_value = values.iloc[0] if starts.Position.min() == 0 else 0

#     run_lengths = (runs.Position - runs.Position.shift().fillna(0))

#     # print("before " * 3, values)
#     values = values.cumsum()
#     # print("middle " * 3, values)
#     values = values.shift()
#     # print("after " * 3, values)
#     values[0] = first_value

#     # the hack that sets the first value might lead to two consecutive equal values; if so, fix
#     if len(values) > 1 and first_value == values[1]:
#         run_lengths[1] += run_lengths[0]
#         values = values[1:]
#         run_lengths = run_lengths[1:]

#     return Rle(run_lengths, values)



@profile
def coverage(ranges, value_col=None):

    try:
        df = ranges.df
    except:
        df = ranges
        # df = df.reset_index(drop=True)

    if value_col:
        values = df[value_col].tolist()
    else:
        values = [1] * len(df)

    starts = df.Start.tolist()
    ends = df.End.tolist()

    d = {}

    for start, value in zip(starts, values):
        if start in d:
            d[start] = d[start] + value
        else:
            d[start] = value

    for end, value in zip(ends, values):
        if end in d:
            d[end] = d[end] - value
        else:
            d[end] = -value

    if 0 not in d:
        d[0] = 0

    sorted_items = sorted(d.items())
    runs = pd.Series([i[0] for i in sorted_items])
    values = pd.Series([i[1] for i in sorted_items])

    first_value = values[0]
    values = values.cumsum().shift()
    values[0] = first_value

    runs = (runs - runs.shift().fillna(0))

    if len(values) > 1 and first_value == values[1]:
        runs[1] += runs[0]
        values = values[1:]
        runs = runs[1:]

    return Rle(runs, values)




def to_ranges(grles):

    from pyranges import GRanges

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

    return GRanges(pd.concat(dfs))


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


# def to_ranges(grles):

#     if grles.stranded:

#         for (chromosome, strand), grle in grles.items():
