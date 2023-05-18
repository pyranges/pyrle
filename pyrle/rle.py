"""Data structure for run length encoding representation and arithmetic."""

import shutil
from numbers import Number

import numpy as np
import pandas as pd
from tabulate import tabulate

from pyrle.src.coverage import _remove_dupes  # type: ignore
from pyrle.src.getitem import getitem, getitems, getlocs  # type: ignore
from pyrle.src.rle import add_rles, div_rles_nonzeroes, div_rles_zeroes, mul_rles, sub_rles  # type: ignore

__all__ = ["Rle"]


def _make_rles_equal_length(self, other, value=0):
    if not isinstance(other, Number):
        ls = np.sum(self.runs)
        lo = np.sum(other.runs)

        if ls > lo:
            new_runs = np.append(other.runs, ls - lo)
            new_values = np.append(other.values, value)
            other = Rle(new_runs, new_values)
        elif lo > ls:
            new_runs = np.append(self.runs, lo - ls)
            new_values = np.append(self.values, value)
            self = Rle(new_runs, new_values)

    return self, other


def find_runs(x):
    """Find runs of consecutive items in an array.

    Author: Alistair Miles
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
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

    """Data structure to represent and manipulate Run Length Encodings.

    An Rle contains two vectors, one with runs (int) and one with values
    (double).

    Operations between Rles act as if it was a regular vector.

    There are three ways to build an Rle: from a vector of runs or a vector of
    values, or a vector of values.

    Parameters
    ----------
    runs : array-like

        Run lengths.

    values : array-like

        Run values.

    See Also
    --------
    pyrle.rledict.RleDict : genomic collection of Rles

    Examples
    --------

    >>> r = Rle([1, 2, 1, 5], [0, 2.1, 3, 4])
    >>> r
    +--------+-----+-----+-----+-----+
    | Runs   | 1   | 2   | 1   | 5   |
    |--------+-----+-----+-----+-----|
    | Values | 0.0 | 2.1 | 3.0 | 4.0 |
    +--------+-----+-----+-----+-----+
    Rle of length 9 containing 4 elements (avg. length 2.25)

    >>> r2 = Rle([1, 1, 1, 0, 0, 2, 2, 3, 4, 2])
    >>> r2
    +--------+-----+-----+-----+-----+-----+-----+
    | Runs   | 3   | 2   | 2   | 1   | 1   | 1   |
    |--------+-----+-----+-----+-----+-----+-----|
    | Values | 1.0 | 0.0 | 2.0 | 3.0 | 4.0 | 2.0 |
    +--------+-----+-----+-----+-----+-----+-----+
    Rle of length 10 containing 6 elements (avg. length 1.667)

    When one Rle is longer than the other, the shorter is extended with zeros:

    >>> r - r2
    +--------+------+-----+-----+-----+-----+-----+-----+------+
    | Runs   | 1    | 2   | 1   | 1   | 2   | 1   | 1   | 1    |
    |--------+------+-----+-----+-----+-----+-----+-----+------|
    | Values | -1.0 | 1.1 | 3.0 | 4.0 | 2.0 | 1.0 | 0.0 | -2.0 |
    +--------+------+-----+-----+-----+-----+-----+-----+------+
    Rle of length 10 containing 8 elements (avg. length 1.25)

    Scalar operations work with Rles:

    >>> r * 5
    +--------+-----+------+------+------+
    | Runs   | 1   | 2    | 1    | 5    |
    |--------+-----+------+------+------|
    | Values | 0.0 | 10.5 | 15.0 | 20.0 |
    +--------+-----+------+------+------+
    Rle of length 9 containing 4 elements (avg. length 2.25)

    """

    runs = None
    values = None

    def __init__(self, runs=None, values=None):
        if values is not None and runs is not None:
            assert len(runs) == len(values)

            runs = np.copy(runs)
            values = np.copy(values)

            runs = np.array(runs, dtype=np.int_)
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
            self.runs = np.array([], dtype=np.int_)
            self.values = np.array([], dtype=np.double)

    def __add__(self, other):
        """Add number or Rle to Rle.

        The shortest Rle is extended with zeros.

        Examples
        --------
        >>> r1 = Rle([1, 2], [0, 1])
        >>> r2 = Rle([2, 2], [2, 3])
        >>> r1 + r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 1   | 1   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 2.0 | 3.0 | 4.0 | 3.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 4 containing 4 elements (avg. length 1.0)

        >>> r1 * 10
        +--------+-----+------+
        | Runs   | 1   | 2    |
        |--------+-----+------|
        | Values | 0.0 | 10.0 |
        +--------+-----+------+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        if isinstance(other, Number):
            return Rle(self.runs, self.values + other)
        else:
            self, other = _make_rles_equal_length(self, other)

        runs, values = add_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    def __array_ufunc__(self, *args, **kwargs):
        """Apply unary numpy-function to the values.

        Notes
        -----

        Function must produce a vector of length equal to self.

        Examples
        --------

        >>> r = Rle([1, 2, 3, 4], [1, 4, 9, 16])
        >>> r
        +--------+-----+-----+-----+------+
        | Runs   | 1   | 2   | 3   | 4    |
        |--------+-----+-----+-----+------|
        | Values | 1.0 | 4.0 | 9.0 | 16.0 |
        +--------+-----+-----+-----+------+
        Rle of length 10 containing 4 elements (avg. length 2.5)

        >>> np.sqrt(r)
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 2   | 3   | 4   |
        |--------+-----+-----+-----+-----|
        | Values | 1.0 | 2.0 | 3.0 | 4.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 10 containing 4 elements (avg. length 2.5)

        >>> np.log10(np.sqrt(r))
        +--------+-----+--------------------+---------------------+--------------------+
        | Runs   | 1   | 2                  | 3                   | 4                  |
        |--------+-----+--------------------+---------------------+--------------------|
        | Values | 0.0 | 0.3010299956639812 | 0.47712125471966244 | 0.6020599913279624 |
        +--------+-----+--------------------+---------------------+--------------------+
        Rle of length 10 containing 4 elements (avg. length 2.5)
        """

        self = self.copy()

        func, call, gr = args

        self.values = getattr(func, call)(self.values, **kwargs)

        return self

    def __eq__(self, other):
        """Return where Rle equal.

        Examples
        --------
        >>> r = Rle([1, 2, 1], [1, 2, 3])
        >>> r2 = Rle([1, 1, 1], [1, 2, 1])
        >>> r == r2
        +--------+-----+-----+
        | Runs   | 2   | 2   |
        |--------+-----+-----|
        | Values | 1.0 | 0.0 |
        +--------+-----+-----+
        Rle of length 4 containing 2 elements (avg. length 2.0)

        >>> r == 3
        +--------+-----+-----+
        | Runs   | 3   | 1   |
        |--------+-----+-----|
        | Values | 0.0 | 1.0 |
        +--------+-----+-----+
        Rle of length 4 containing 2 elements (avg. length 2.0)
        """

        self, other = _make_rles_equal_length(self, other, np.nan)

        r = self - other
        r.values = np.where(r.values == 0, 1.0, 0.0)
        return r.defragment()

    def __getitem__(self, val):
        if isinstance(val, int):
            values = getlocs(self.runs, self.values, np.array([val], dtype=np.int_))
            return values[0]
        elif isinstance(val, slice):
            end = val.stop or np.sum(self.runs)
            start = val.start or 0
            runs, values = getitem(self.runs, self.values, start, end)
            return Rle(runs, values)
        elif isinstance(val, pd.DataFrame):
            intype = val.dtypes["Start"]
            val = val["Start End".split()].astype(np.int_)
            ids, starts, ends, runs, values = getitems(self.runs, self.values, val.Start.values, val.End.values)

            df = pd.DataFrame({"Start": starts, "End": ends, "ID": ids, "Run": runs, "Value": values}).astype(
                {"Start": intype, "End": intype}
            )
            # val = val["Start End".split()].astype(np.int)
            # values = getitems(self.runs, self.values, val.Start.values, val.End.values)
            return df
        elif "PyRanges" in str(type(val)):  # hack to avoid isinstance(key, pr.PyRanges) so that we
            # do not need a dep on PyRanges in this library
            import pyranges as pr  # type: ignore

            val = val.drop().df
            if val.empty:
                return pd.DataFrame(columns="Chromosome Start End ID Run Value".split())

            chromosome = val.Chromosome.iloc[0]

            intype = val.dtypes["Start"]

            if "Strand" in val:
                strand = val.Strand.iloc[0]
            else:
                strand = None

            val = val["Start End".split()].astype(np.int_)
            ids, starts, ends, runs, values = getitems(self.runs, self.values, val.Start.values, val.End.values)

            df = pd.DataFrame(
                {
                    "Chromosome": chromosome,
                    "Start": starts,
                    "End": ends,
                    "ID": ids,
                    "Run": runs,
                    "Value": values,
                }
            ).astype({"Start": intype, "End": intype})

            if strand:
                df.insert(3, "Strand", strand)

            return pr.PyRanges(df)

        else:
            locs = np.sort(np.array(val, dtype=np.int_))
            values = getlocs(self.runs, self.values, locs)
            return values

    def __ge__(self, other):
        """Check if greater or equal to other.

        Examples
        --------

        >>> r = Rle([1, 2, 3], [0, 2, 1])
        >>> r2 = Rle([2, 1, 2], [2, 1, 2])
        >>> r >= r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 2   | 2   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 0.0 | 1.0 | 0.0 | 1.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 6 containing 4 elements (avg. length 1.5)

        >>> r >= 1
        +--------+-----+-----+
        | Runs   | 1   | 5   |
        |--------+-----+-----|
        | Values | 0.0 | 1.0 |
        +--------+-----+-----+
        Rle of length 6 containing 2 elements (avg. length 3.0)
        """

        r = self - other
        r.values = np.where(r.values >= 0, 1.0, 0.0)
        return r.defragment()

    def __gt__(self, other):
        """Check if greater than other.

        Examples
        --------

        >>> r = Rle([1, 2, 3], [0, 5, 1])
        >>> r2 = Rle([2, 1, 2], [2, 3, 9])
        >>> r > r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 2   | 2   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 0.0 | 1.0 | 0.0 | 1.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 6 containing 4 elements (avg. length 1.5)

        >>> r > 2
        +--------+-----+-----+-----+
        | Runs   | 1   | 2   | 3   |
        |--------+-----+-----+-----|
        | Values | 0.0 | 1.0 | 0.0 |
        +--------+-----+-----+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)
        """

        r = self - other
        r.values = np.where(r.values > 0, 1.0, 0.0)
        return r.defragment()

    def __le__(self, other):
        """Check if less than or equal to other.
        Examples
        --------

        >>> r = Rle([1, 2, 3], [0, 5, 1])
        >>> r2 = Rle([2, 1, 2], [2, 3, 9])
        >>> r <= r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 2   | 2   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 1.0 | 0.0 | 1.0 | 0.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 6 containing 4 elements (avg. length 1.5)

        >>> r <= 2
        +--------+-----+-----+-----+
        | Runs   | 1   | 2   | 3   |
        |--------+-----+-----+-----|
        | Values | 1.0 | 0.0 | 1.0 |
        +--------+-----+-----+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        """

        r = self - other
        r.values = np.where(r.values <= 0, 1.0, 0.0)
        return r.defragment()

    def __len__(self):
        """Return number of runs in Rle.

        See Also
        --------
        pyrle.Rle.length : return length of Rle."""

        return len(self.runs)

    def __lt__(self, other):
        """Check if less than other.

        Examples
        --------

        >>> r = Rle([1, 2, 3], [0, 5, 1])
        >>> r2 = Rle([2, 1, 2], [2, 3, 9])
        >>> r < r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 2   | 2   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 1.0 | 0.0 | 1.0 | 0.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 6 containing 4 elements (avg. length 1.5)

        >>> r < 2
        +--------+-----+-----+-----+
        | Runs   | 1   | 2   | 3   |
        |--------+-----+-----+-----|
        | Values | 1.0 | 0.0 | 1.0 |
        +--------+-----+-----+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        """

        r = self - other
        r.values = np.where(r.values < 0, 1.0, 0.0)
        return r.defragment()

    def __mul__(self, other):
        """Subtract number or Rle from Rle.

        The shortest Rle is extended with zeros.

        Examples
        --------
        >>> r1 = Rle([1, 2], [0, 1])
        >>> r2 = Rle([2, 2], [2, 3])
        >>> r1 * r2
        +--------+-----+-----+-----+-----+
        | Runs   | 1   | 1   | 1   | 1   |
        |--------+-----+-----+-----+-----|
        | Values | 0.0 | 2.0 | 3.0 | 0.0 |
        +--------+-----+-----+-----+-----+
        Rle of length 4 containing 4 elements (avg. length 1.0)

        >>> r1 * 10
        +--------+-----+------+
        | Runs   | 1   | 2    |
        |--------+-----+------|
        | Values | 0.0 | 10.0 |
        +--------+-----+------+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        if isinstance(other, Number):
            return Rle(self.runs, self.values * other)
        else:
            self, other = _make_rles_equal_length(self, other)

        runs, values = mul_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    def __ne__(self, other):
        """Return where not equal.

        Examples
        --------
        >>> r = Rle([1, 2, 1], [1, 2, 3])
        >>> r2 = Rle([1, 1, 1], [1, 2, 1])
        >>> r != r2
        +--------+-----+-----+
        | Runs   | 2   | 2   |
        |--------+-----+-----|
        | Values | 0.0 | 1.0 |
        +--------+-----+-----+
        Rle of length 4 containing 2 elements (avg. length 2.0)
        """

        self, other = _make_rles_equal_length(self, other, np.nan)

        r = self - other
        r.values = np.where(r.values != 0, 1.0, 0.0)
        return r.defragment()

    def __neg__(self):
        """Negate values.

        Examples
        --------
        >>> r = Rle([1, 2, 3], [5, -20, 1])
        >>> r
        +--------+-----+-------+-----+
        | Runs   | 1   | 2     | 3   |
        |--------+-----+-------+-----|
        | Values | 5.0 | -20.0 | 1.0 |
        +--------+-----+-------+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        >>> -r
        +--------+------+------+------+
        | Runs   | 1    | 2    | 3    |
        |--------+------+------+------|
        | Values | -5.0 | 20.0 | -1.0 |
        +--------+------+------+------+
        Rle of length 6 containing 3 elements (avg. length 2.0)
        """

        self = self.copy()
        self.values = -self.values
        return self

    def __radd__(self, other):
        """Add scalar to Rle values.

        Examples
        --------
        >>> 5 + Rle([1, 2], [3, 4])
        +--------+-----+-----+
        | Runs   | 1   | 2   |
        |--------+-----+-----|
        | Values | 8.0 | 9.0 |
        +--------+-----+-----+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        return Rle(self.runs, self.values + other)

    def __repr__(self):
        """Return REPL string representation."""

        return str(self)

    def __rmul__(self, other):
        """Multiply scalar with Rle-values.

        Examples
        --------
        >>> 5 * Rle([1, 2], [0.5, 1])
        +--------+-----+-----+
        | Runs   | 1   | 2   |
        |--------+-----+-----|
        | Values | 2.5 | 5.0 |
        +--------+-----+-----+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        return Rle(self.runs, self.values * other)

    def __rsub__(self, other):
        """Subtract Rle-values from scalar.

        Examples
        --------
        >>> 5 - Rle([1, 2], [0.5, 1])
        +--------+-----+-----+
        | Runs   | 1   | 2   |
        |--------+-----+-----|
        | Values | 4.5 | 4.0 |
        +--------+-----+-----+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        return Rle(self.runs, other - self.values)

    def __rtruediv__(self, other):
        """Divide scalar with Rle-values.

        Examples
        --------
        >>> 5 / Rle([1, 2], [0.5, 1])
        +--------+------+-----+
        | Runs   | 1    | 2   |
        |--------+------+-----|
        | Values | 10.0 | 5.0 |
        +--------+------+-----+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        return Rle(self.runs, other / self.values)

    def __str__(self):
        """Return string representation of Rle."""

        terminal_width = shutil.get_terminal_size().columns

        entries = min(len(self.runs), 10)
        half_entries = int(entries / 2)

        start_runs, end_runs = [str(i) for i in self.runs[:half_entries]], [str(i) for i in self.runs[-half_entries:]]
        start_values, end_values = [str(i) for i in self.values[:half_entries]], [
            str(i) for i in self.values[-half_entries:]
        ]

        if entries < len(self.runs):
            runs = start_runs + ["..."] + end_runs
            values = start_values + ["..."] + end_values
        else:
            runs, values = self.runs, self.values

        df = pd.Series(values).to_frame().T

        df.columns = list(runs)
        df.index = ["Values"]
        df.index.name = "Runs"

        outstr = tabulate(df, tablefmt="psql", showindex=True, headers="keys", disable_numparse=True)

        while len(outstr.split("\n", 1)[0]) > terminal_width:
            half_entries -= 1

            runs = start_runs[:half_entries] + ["..."] + end_runs[-half_entries:]
            values = start_values[:half_entries] + ["..."] + end_values[-half_entries:]

            df = pd.Series(values).to_frame().T

            df.columns = list(runs)
            df.index = ["Values"]
            df.index.name = "Runs"

            outstr = tabulate(
                df,
                tablefmt="psql",
                showindex=True,
                headers="keys",
                disable_numparse=True,
            )

        length = np.sum(self.runs)
        elements = len(self.runs)
        info = "\nRle of length {} containing {} elements (avg. length {})".format(
            str(length), str(elements), str(np.round(length / elements, 3))
        )

        return outstr + info

    def __sub__(self, other):
        """Subtract number or Rle from Rle.

        The shortest Rle is extended with zeros.

        Examples
        --------
        >>> r1 = Rle([1, 2], [0, 1])
        >>> r2 = Rle([2, 2], [2, 3])
        >>> r1 - r2
        +--------+------+------+------+------+
        | Runs   | 1    | 1    | 1    | 1    |
        |--------+------+------+------+------|
        | Values | -2.0 | -1.0 | -2.0 | -3.0 |
        +--------+------+------+------+------+
        Rle of length 4 containing 4 elements (avg. length 1.0)

        >>> r1 - 10
        +--------+-------+------+
        | Runs   | 1     | 2    |
        |--------+-------+------|
        | Values | -10.0 | -9.0 |
        +--------+-------+------+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        if isinstance(other, Number):
            return Rle(self.runs, self.values - other)
        else:
            self, other = _make_rles_equal_length(self, other)

        runs, values = sub_rles(self.runs, self.values, other.runs, other.values)
        return Rle(runs, values)

    def __truediv__(self, other):
        """Divide Rle with number or Rle.

        The shortest Rle is extended with zeros.

        Examples
        --------
        >>> r1 = Rle([1, 2], [0, 1])
        >>> r2 = Rle([2, 2], [2, 3])
        >>> r1 / r2
        +--------+-----+-----+--------------------+-----+
        | Runs   | 1   | 1   | 1                  | 1   |
        |--------+-----+-----+--------------------+-----|
        | Values | 0.0 | 0.5 | 0.3333333333333333 | 0.0 |
        +--------+-----+-----+--------------------+-----+
        Rle of length 4 containing 4 elements (avg. length 1.0)

        >>> r1 / 10
        +--------+-----+-----+
        | Runs   | 1   | 2   |
        |--------+-----+-----|
        | Values | 0.0 | 0.1 |
        +--------+-----+-----+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        """

        if isinstance(other, Number):
            return Rle(self.runs, self.values / other)
        else:
            self, other = _make_rles_equal_length(self, other)

        if (other.values == 0).any() or np.sum(other.runs) < np.sum(self.runs):
            runs, values = div_rles_zeroes(self.runs, self.values, other.runs, other.values)
        else:
            runs, values = div_rles_nonzeroes(self.runs, self.values, other.runs, other.values)

        return Rle(runs, values)

    def apply_values(self, f, defragment=True):
        """Apply function to the values.

        Parameters
        ----------
        f : function

            Must return vector of double with same length as Rle.

        defragment : bool, default True

            Whether to merge consecutive runs of same value after application.

        See Also
        --------

        pyrle.__array_ufunc__ : apply numpy functions to pyrle.

        Examples
        --------

        >>> r = Rle([1, 3, 5], [100, 200, -300])
        >>> r.apply_values(lambda v: np.sqrt(v))
        +--------+------+--------------------+-----+
        | Runs   | 1    | 3                  | 5   |
        |--------+------+--------------------+-----|
        | Values | 10.0 | 14.142135620117188 | nan |
        +--------+------+--------------------+-----+
        Rle of length 9 containing 3 elements (avg. length 3.0)

        >>> def gt0_to_1(v):
        ...     v[v > 0] = 1
        ...     return v

        >>> r.apply_values(gt0_to_1, defragment=False)
        +--------+-----+-----+--------+
        | Runs   | 1   | 3   | 5      |
        |--------+-----+-----+--------|
        | Values | 1.0 | 1.0 | -300.0 |
        +--------+-----+-----+--------+
        Rle of length 9 containing 3 elements (avg. length 3.0)

        >>> r.apply_values(gt0_to_1, defragment=True)
        +--------+-----+--------+
        | Runs   | 4   | 5      |
        |--------+-----+--------|
        | Values | 1.0 | -300.0 |
        +--------+-----+--------+
        Rle of length 9 containing 2 elements (avg. length 4.5)
        """

        self = self.copy()
        self.values = f(self.values)
        if defragment:
            self = self.defragment()
        return self

    def apply_runs(self, f, defragment=True):
        """Apply function to the runs.

        Parameters
        ----------
        f : function

            Must return vector of ints with same length as Rle.

        defragment : bool, default True

            Whether to merge consecutive runs of same value after application.

        Examples
        --------

        >>> r = Rle([1, 3, 5], [100, 200, -300])
        >>> r.apply_runs(lambda v: (v ** 2).astype(int))
        +--------+-------+-------+--------+
        | Runs   | 1     | 9     | 25     |
        |--------+-------+-------+--------|
        | Values | 100.0 | 200.0 | -300.0 |
        +--------+-------+-------+--------+
        Rle of length 35 containing 3 elements (avg. length 11.667)
        """

        self = self.copy()
        self.runs = f(self.runs)
        if defragment:
            self = self.defragment()
        return self

    def apply(self, f, defragment=True):
        """Apply function to the Rle.

        Parameters
        ----------
        f : function

            Must return Rle.

        defragment : bool, default True

            Whether to merge consecutive runs of same value after application.

        Examples
        --------

        >>> r = Rle([1, 3, 5], [100, 200, -300])
        >>> def shuffle(rle):
        ...     np.random.seed(0)
        ...     np.random.shuffle(rle.values)
        ...     np.random.shuffle(rle.runs)
        ...     return rle

        >>> r.apply(shuffle)
        +--------+--------+-------+-------+
        | Runs   | 5      | 1     | 3     |
        |--------+--------+-------+-------|
        | Values | -300.0 | 200.0 | 100.0 |
        +--------+--------+-------+-------+
        Rle of length 9 containing 3 elements (avg. length 3.0)
        """

        self = self.copy()
        self = f(self)
        if defragment:
            self = self.defragment()
        return self

    def copy(self):
        """Return copy of Rle."""

        return Rle(np.copy(self.runs), np.copy(self.values))

    def defragment(self):
        """Merge consecutive values.

        Examples
        --------
        >>> r = Rle([1, 2, 3], [1, 0, 1])
        >>> r
        +--------+-----+-----+-----+
        | Runs   | 1   | 2   | 3   |
        |--------+-----+-----+-----|
        | Values | 1.0 | 0.0 | 1.0 |
        +--------+-----+-----+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        >>> r.values[1] = 1
        >>> r.values[2] = 2
        >>> r
        +--------+-----+-----+-----+
        | Runs   | 1   | 2   | 3   |
        |--------+-----+-----+-----|
        | Values | 1.0 | 1.0 | 2.0 |
        +--------+-----+-----+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        >>> r.defragment()
        +--------+-----+-----+
        | Runs   | 3   | 3   |
        |--------+-----+-----|
        | Values | 1.0 | 2.0 |
        +--------+-----+-----+
        Rle of length 6 containing 2 elements (avg. length 3.0)

        """

        runs, values = _remove_dupes(self.runs, self.values, len(self))
        values[values == -0] = 0
        return Rle(runs, values)

    @property
    def length(self):
        """Return sum of runs vector.

        See Also
        --------
        pyrle.Rle.__len__ : return number of runs.

        Examples
        --------

        >>> Rle([5], [0]).length
        5

        >>> gauss = Rle(np.arange(1, 101), [0, 1] * 50)
        >>> gauss
        +--------+-----+-----+-----+-----+-----+-------+------+------+------+------+-------+
        | Runs   | 1   | 2   | 3   | 4   | 5   | ...   | 96   | 97   | 98   | 99   | 100   |
        |--------+-----+-----+-----+-----+-----+-------+------+------+------+------+-------|
        | Values | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | ...   | 1.0  | 0.0  | 1.0  | 0.0  | 1.0   |
        +--------+-----+-----+-----+-----+-----+-------+------+------+------+------+-------+
        Rle of length 5050 containing 100 elements (avg. length 50.5)

        >>> gauss.length
        5050
        """
        return np.sum(self.runs)

    def mean(self):
        """Return mean of values.

        The values are multiplied with their run length.

        Examples
        --------
        >>> Rle([1, 2, 1], [1, 2, 3]).mean()
        1.5
        >>> # ((1 * 1) + (2 * 2) + (1 * 3)) / (1 + 2 + 1)
        """

        length = self.length
        _sum = np.sum(self.values)
        return _sum / length

    def numbers_only(self, nan=0.0, posinf=2147483647, neginf=-2147483648):
        """Fill inf with large values and nan with 0.

        Parameters
        ----------
        nan : double, default 0

            Value to represent nan

        posinf : double, default 2147483647

            Value to represent inf.

        neginf : double, default -2147483648

            Value to represent -inf.

        Examples
        --------
        >>> r = Rle([1, 2, 1, 2, 1], [-np.inf, 1, np.nan, 1, np.inf])
        >>> r
        +--------+------+-----+-----+-----+-----+
        | Runs   | 1    | 2   | 1   | 2   | 1   |
        |--------+------+-----+-----+-----+-----|
        | Values | -inf | 1.0 | nan | 1.0 | inf |
        +--------+------+-----+-----+-----+-----+
        Rle of length 7 containing 5 elements (avg. length 1.4)

        >>> r.numbers_only()
        +--------+---------------+-----+-----+-----+--------------+
        | Runs   | 1             | 2   | 1   | 2   | 1            |
        |--------+---------------+-----+-----+-----+--------------|
        | Values | -2147483648.0 | 1.0 | 0.0 | 1.0 | 2147483648.0 |
        +--------+---------------+-----+-----+-----+--------------+
        Rle of length 7 containing 5 elements (avg. length 1.4)
        """

        return Rle(self.runs, np.nan_to_num(self.values, nan=nan, posinf=posinf, neginf=neginf)).defragment()

    def shift(self, dist=1, preserve_length=True, fill=0):
        """Shift values.

        Parameters
        ----------
        dist : int, default 1

            Shift distance. Negative means shift left.

        preserve_length : bool, default True

            Fill end when shifting left, or truncate end when shifting right.

        fill : int, default 0

            Fill for values shifted out of bounds.

        Examples
        --------

        >>> r = Rle([3, 2, 1], [1, -1, 2])
        >>> r
        +--------+-----+------+-----+
        | Runs   | 3   | 2    | 1   |
        |--------+-----+------+-----|
        | Values | 1.0 | -1.0 | 2.0 |
        +--------+-----+------+-----+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        >>> r.shift(2, preserve_length=False, fill=np.nan)
        +--------+-----+-----+------+-----+
        | Runs   | 2   | 3   | 2    | 1   |
        |--------+-----+-----+------+-----|
        | Values | nan | 1.0 | -1.0 | 2.0 |
        +--------+-----+-----+------+-----+
        Rle of length 8 containing 4 elements (avg. length 2.0)

        >>> r.shift(2)
        +--------+-----+-----+------+
        | Runs   | 2   | 3   | 1    |
        |--------+-----+-----+------|
        | Values | 0.0 | 1.0 | -1.0 |
        +--------+-----+-----+------+
        Rle of length 6 containing 3 elements (avg. length 2.0)

        >>> r.shift(-2, fill=np.nan)
        +--------+-----+------+-----+-----+
        | Runs   | 1   | 2    | 1   | 2   |
        |--------+-----+------+-----+-----|
        | Values | 1.0 | -1.0 | 2.0 | nan |
        +--------+-----+------+-----+-----+
        Rle of length 6 containing 4 elements (avg. length 1.5)

        >>> r.shift(-4, preserve_length=False)
        +--------+------+-----+
        | Runs   | 1    | 1   |
        |--------+------+-----|
        | Values | -1.0 | 2.0 |
        +--------+------+-----+
        Rle of length 2 containing 2 elements (avg. length 1.0)
        """

        self = self.copy()
        if dist > 0:
            original_length = self.length
            if self.values[0] == fill:
                self.runs[0] += dist
            else:
                self.values = np.r_[fill, self.values]
                self.runs = np.r_[dist, self.runs]

            if preserve_length:
                self = self[:original_length]

        elif dist < 0:
            dist = -dist  # remember dist is negative
            if dist < self.runs[0]:
                self.runs[0] -= dist
            else:
                cs = np.cumsum(self.runs)
                ix = np.argmax(cs > dist)
                leftover = np.sum(self.runs[:ix]) - dist
                self = Rle(self.runs[ix:], self.values[ix:])
                self.runs[0] += leftover

                if self.runs[0] < 0:
                    self = Rle([], [])

            if preserve_length:
                if self.values[-1] == fill:
                    self.runs[-1] += dist
                else:
                    self.values = np.r_[self.values, fill]
                    self.runs = np.r_[self.runs, dist]

        return self

    def std(self):
        """Return standard deviation.

        See Also
        --------

        pyrle.Rle.mean : return mean

        Examples
        --------
        >>> Rle([1, 2, 1], [1, 2, 3]).std()
        0.8660254037844386
        """

        _sum = np.sum(self.values - self.mean()) ** 2

        return np.sqrt(_sum / (self.length - 1))

    def to_frame(self):
        """Return Rle as DataFrame.

        See Also
        --------

        pyrle.Rle.to_csv : write Rle to csv

        Examples
        --------

        >>> df = Rle([1, 5, 18], [0, 1, 0]).to_frame()
        >>> df
           Runs  Values
        0     1     0.0
        1     5     1.0
        2    18     0.0
        """

        return pd.DataFrame(data={"Runs": self.runs, "Values": self.values})["Runs Values".split()]

    def to_csv(self, **kwargs):
        """Return Rle as DataFrame.

        Parameters
        ----------
        **kwargs

            See the docs for pandas.DataFrame.to_csv

        See Also
        --------

        pyrle.Rle.to_frame : return Rle as DataFrame

        Examples
        --------

        >>> df = Rle([1, 5, 18], [0, 1, 0]).to_frame()
        >>> df
           Runs  Values
        0     1     0.0
        1     5     1.0
        2    18     0.0
        """

        self.to_frame().to_csv(**kwargs)
