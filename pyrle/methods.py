import os
from collections import defaultdict

import numpy as np
import pandas as pd
from natsort import natsorted  # type: ignore

from pyrle import Rle  # type: ignore
from pyrle import rledict as rd  # type: ignore
from pyrle.src.coverage import _coverage  # type: ignore


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def _merge_rles(rle):
    new_dict = {}
    dd = defaultdict(list)
    for chromosome, strand in rle.rles.keys():
        dd[chromosome].append(strand)

    for c, s in dd.items():
        if len(s) == 1:
            new_dict[c] = rle.rles[c, s[0]]
        else:
            new_dict[c] = rle.rles[c, "+"] + rle.rles[c, "-"]

    return new_dict


def ensure_both_or_none_stranded(self, other):
    # means other not stranded
    if self.stranded:
        self.rles = _merge_rles(self)
    else:
        other.rles = _merge_rles(other)

    return self, other


def chromosomes_in_both_self_other(self, other):
    chromosomes_in_both = natsorted(set(self.rles.keys()).intersection(other.rles.keys()))
    chromosomes_in_self_not_other = natsorted(set(self.rles.keys()) - set(other.rles.keys()))
    chromosomes_in_other_not_self = natsorted(set(other.rles.keys()) - set(self.rles.keys()))

    return (
        chromosomes_in_both,
        chromosomes_in_self_not_other,
        chromosomes_in_other_not_self,
    )


def binary_operation(operation, self, other, nb_cpu=1):
    func = {"div": __div, "mul": __mul, "add": __add, "sub": __sub}[operation]
    func, get = rd.get_multithreaded_funcs(func, nb_cpu)

    if nb_cpu > 1:
        import ray  # type: ignore

        with suppress_stdout_stderr():
            ray.init(num_cpus=nb_cpu)

    if self.stranded != other.stranded:
        self, other = ensure_both_or_none_stranded(self, other)

    (
        chromosomes_in_both,
        chromosomes_in_self_not_other,
        chromosomes_in_other_not_self,
    ) = chromosomes_in_both_self_other(self, other)

    both_results = []
    for c in chromosomes_in_both:
        both_results.append(func.remote(self.rles[c], other.rles[c]))

    self_results = []
    for c in chromosomes_in_self_not_other:
        _other = Rle([np.sum(self.rles[c].runs)], [0])
        self_results.append(func.remote(self.rles[c], _other))

    other_results = []
    for c in chromosomes_in_other_not_self:
        _self = Rle([np.sum(other.rles[c].runs)], [0])
        other_results.append(func.remote(_self, other.rles[c]))

    rles = {
        k: v
        for k, v in zip(
            chromosomes_in_both + chromosomes_in_self_not_other + chromosomes_in_other_not_self,
            get(both_results + self_results + other_results),
        )
    }
    return rd.RleDict(rles)


def __add(self, other):
    return self + other


def __sub(self, other):
    return self - other


def __div(self, other):
    return self / other


def __mul(self, other):
    return self * other


def coverage(df, **kwargs):
    value_col = kwargs.get("value_col", None)

    if value_col:
        values = df[value_col].astype(np.float64).values
    else:
        values = np.ones(len(df))

    starts_df = pd.DataFrame({"Position": df.Start, "Value": values})["Position Value".split()]
    ends_df = pd.DataFrame({"Position": df.End, "Value": -1 * values})["Position Value".split()]
    _df = pd.concat([starts_df, ends_df], ignore_index=True)
    _df = _df.sort_values("Position", kind="mergesort")

    _df.Position = _df.Position.astype(np.int64)

    runs, values = _coverage(_df.Position.values, _df.Value.values)

    return Rle(runs, values)


def to_ranges_df_strand(rle, k):
    chromosome, strand = k
    starts, ends, values = _to_ranges(rle)
    df = pd.concat([pd.Series(r) for r in [starts, ends, values]], axis=1)
    df.columns = "Start End Score".split()
    df.insert(0, "Chromosome", chromosome)
    df.insert(df.shape[1], "Strand", strand)
    df = df[df.Score != 0]

    return df


def to_ranges_df_no_strand(rle, k):
    starts, ends, values = _to_ranges(rle)
    df = pd.concat([pd.Series(r) for r in [starts, ends, values]], axis=1)
    df.columns = "Start End Score".split()
    df.insert(0, "Chromosome", k)
    df = df[df.Score != 0]

    return df


def to_ranges(grles, nb_cpu=1):
    from pyranges import PyRanges  # type: ignore

    func = to_ranges_df_strand if grles.stranded else to_ranges_df_no_strand

    if nb_cpu > 1:
        import ray  # type: ignore

        ray.init(num_cpus=nb_cpu)
        func = ray.remote(func)
        get = ray.get
    else:
        func.remote = func

        def get(x):
            return x

    dfs, keys = [], []
    for k, v in grles.items():
        result = func.remote(v, k)
        dfs.append(result)
        keys.append(k)

    dfs = {k: v for (k, v) in zip(keys, get(dfs))}

    if nb_cpu > 1:
        ray.shutdown()

    return PyRanges(dfs)


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

    return (
        starts.astype(int).reset_index(drop=True),
        ends.astype(int).reset_index(drop=True),
        values,
    )
