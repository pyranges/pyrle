import pandas as pd
import numpy as np

from pyrle import Rle
from pyrle.rledict import GRles

from natsort import natsorted

from sys import stderr

from joblib import Parallel, delayed

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


def _add(self, other, n_jobs=1):

    chromosomes_in_both, chromosomes_in_self_not_other, chromosomes_in_other_not_self = chromosomes_in_both_self_other(self, other)

    _rles = Parallel(n_jobs=n_jobs)(delayed(__add)(self.rles[c], other.rles[c]) for c in chromosomes_in_both)

    rles = {c: r for c, r in zip(chromosomes_in_both, _rles)}

    for c in chromosomes_in_self_not_other:
        rles[c] = self.rles[c]

    for c in chromosomes_in_other_not_self:
        rles[c] = other.rles[c]

    return GRles(rles)


def _sub(self, other, njobs):
    pass


def coverage(ranges, value_col=None):

    try:
        df = ranges.df
    except:
        df = ranges

    if value_col:
        starts = df[["Start"] + [value_col]]
        ends = df[["End"] + [value_col]]
        # spurious warning
        pd.options.mode.chained_assignment = None
        ends.loc[:, value_col] = ends.loc[:, value_col] * - 1
        pd.options.mode.chained_assignment = "warn"
        columns = ["Position"] + [value_col]
    else:
        starts = pd.concat([df.Start, pd.Series(np.ones(len(df)))], axis=1)
        ends = pd.concat([df.End, -1 * pd.Series(np.ones(len(df)))], axis=1)
        columns = "Position Value".split()
        value_col = "Value"

    starts.columns, ends.columns = columns, columns
    runs = pd.concat([starts, ends], ignore_index=True).sort_values("Position")
    values = runs.groupby("Position").sum().reset_index()[value_col]
    runs = runs.drop_duplicates("Position")
    first_value = values.iloc[0] if starts.Position.min() == 0 else 0
    run_lengths = (runs.Position - runs.Position.shift().fillna(0))

    values = values.cumsum().shift()
    values[0] = first_value

    return Rle(run_lengths, values)
