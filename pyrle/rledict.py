"""Data structure for collection of genomic Rles.

It has the same methods as the Rle object, but align these on the chromosome
or chromosome and strand pairs.

See the documentation for pyrle.Rle.
"""

from numbers import Number

import numpy as np
from natsort import natsorted  # type: ignore

import pyrle.methods as m  # type: ignore
from pyrle import Rle  # type: ignore
from pyrle.src.getitem import getitems  # type: ignore

__all__ = ["RleDict"]


def get_multithreaded_funcs(function, nb_cpu):
    if nb_cpu > 1:
        import ray  # type: ignore

        get = ray.get
        function = ray.remote(function)
    else:

        def get(x):
            return x

        function.remote = function

    return function, get


class RleDict:

    """Data structure to represent and manipulate a genomic collection of Rles.

    Parameters
    ----------
    ranges : dict of Rles, DataFrame or PyRanges, default None

        Data to build RleDict from.

    stranded : bool, default False

        Whether to make separate Rles for each strand. Default False.

    value_col : str, default None

        Column to use for Rle values cols.

    nb_cpu : int, default 1

        Number of CPUs used to create the RleDict.

    See Also
    --------

    pyrle.rle.Rle : Numerical run length encoding

    Examples
    --------

    >>> r = Rle([1, 2, 1, 5], [0, 2.1, 3, 4])
    >>> r2 = Rle([1, 1, 1, 0, 0, 2, 2, 3, 4, 2])
    >>> rd = RleDict({"chr1": r, "chr2": r2})
    >>> rd
    chr1
    ----
    +--------+-----+-----+-----+-----+
    | Runs   | 1   | 2   | 1   | 5   |
    |--------+-----+-----+-----+-----|
    | Values | 0.0 | 2.1 | 3.0 | 4.0 |
    +--------+-----+-----+-----+-----+
    Rle of length 9 containing 4 elements (avg. length 2.25)
    <BLANKLINE>
    chr2
    ----
    +--------+-----+-----+-----+-----+-----+-----+
    | Runs   | 3   | 2   | 2   | 1   | 1   | 1   |
    |--------+-----+-----+-----+-----+-----+-----|
    | Values | 1.0 | 0.0 | 2.0 | 3.0 | 4.0 | 2.0 |
    +--------+-----+-----+-----+-----+-----+-----+
    Rle of length 10 containing 6 elements (avg. length 1.667)
    Unstranded RleDict object with 2 chromosomes.

    >>> import pyranges as pr
    >>> gr = pr.data.chipseq()
    >>> df = pr.data.chipseq_background().df
    >>> cs = RleDict(gr, stranded=True)
    >>> bg = RleDict(df, stranded=True)

    >>> cs
    chr1 +
    +--------+-----------+------+---------+------+-----------+-------+------+-----------+------+-----------+------+
    | Runs   | 1541598   | 25   | 57498   | 25   | 1904886   | ...   | 25   | 2952580   | 25   | 1156833   | 25   |
    |--------+-----------+------+---------+------+-----------+-------+------+-----------+------+-----------+------|
    | Values | 0.0       | 1.0  | 0.0     | 1.0  | 0.0       | ...   | 1.0  | 0.0       | 1.0  | 0.0       | 1.0  |
    +--------+-----------+------+---------+------+-----------+-------+------+-----------+------+-----------+------+
    Rle of length 247134924 containing 894 elements (avg. length 276437.275)
    ...
    chrY -
    +--------+-----------+------+----------+------+----------+-------+------+----------+------+----------+------+
    | Runs   | 7046809   | 25   | 358542   | 25   | 296582   | ...   | 25   | 143271   | 25   | 156610   | 25   |
    |--------+-----------+------+----------+------+----------+-------+------+----------+------+----------+------|
    | Values | 0.0       | 1.0  | 0.0      | 1.0  | 0.0      | ...   | 1.0  | 0.0      | 1.0  | 0.0      | 1.0  |
    +--------+-----------+------+----------+------+----------+-------+------+----------+------+----------+------+
    Rle of length 22210662 containing 32 elements (avg. length 694083.188)
    RleDict object with 48 chromosomes/strand pairs.

    >>> cs - (bg * 5)
    chr1 +
    +--------+-----------+------+----------+------+---------+-------+------+----------+------+-----------+------+
    | Runs   | 1041102   | 25   | 500471   | 25   | 57498   | ...   | 25   | 363693   | 25   | 1156833   | 25   |
    |--------+-----------+------+----------+------+---------+-------+------+----------+------+-----------+------|
    | Values | 0.0       | -5.0 | 0.0      | 1.0  | 0.0     | ...   | -5.0 | 0.0      | 1.0  | 0.0       | 1.0  |
    +--------+-----------+------+----------+------+---------+-------+------+----------+------+-----------+------+
    Rle of length 247134924 containing 1618 elements (avg. length 152740.991)
    ...
    chrY -
    +--------+-----------+------+----------+------+----------+-------+------+----------+------+------------+------+
    | Runs   | 7046809   | 25   | 358542   | 25   | 296582   | ...   | 25   | 156610   | 25   | 35191552   | 25   |
    |--------+-----------+------+----------+------+----------+-------+------+----------+------+------------+------|
    | Values | 0.0       | 1.0  | 0.0      | 1.0  | 0.0      | ...   | 1.0  | 0.0      | 1.0  | 0.0        | -5.0 |
    +--------+-----------+------+----------+------+----------+-------+------+----------+------+------------+------+
    Rle of length 57402239 containing 42 elements (avg. length 1366719.976)
    RleDict object with 50 chromosomes/strand pairs.
    """

    def __init__(self, ranges=None, stranded=False, value_col=None, nb_cpu=1):
        # Construct RleDict from dict of rles
        if isinstance(ranges, dict):
            self.rles = ranges
            self.__dict__["stranded"] = True if len(list(ranges.keys())[0]) == 2 else False
        elif ranges is None:
            self.rles = {}

        # Construct RleDict from ranges
        else:
            if stranded:
                grpby_keys = "Chromosome Strand".split()
            else:
                grpby_keys = "Chromosome"

            try:
                df = ranges.df
            except AttributeError:
                df = ranges

            grpby = list(natsorted(df.groupby(grpby_keys)))

            if nb_cpu > 1:
                import ray  # type: ignore

                with m.suppress_stdout_stderr():
                    ray.init(num_cpus=nb_cpu)

            m_coverage, get = get_multithreaded_funcs(m.coverage, nb_cpu)

            _rles = {}
            kwargs = {"value_col": value_col}
            if stranded:
                for (c, s), cdf in grpby:
                    _rles[c, s] = m_coverage.remote(cdf, **kwargs)
            else:
                s = None
                for k, cdf in grpby:
                    _rles[k] = m_coverage.remote(cdf, **kwargs)

            _rles = {k: v for k, v in zip(_rles.keys(), get(list(_rles.values())))}

            if nb_cpu > 1:
                ray.shutdown()

            self.rles = _rles

            self.__dict__["stranded"] = stranded

    def __add__(self, other):
        if isinstance(other, Number):
            return RleDict({cs: v + other for cs, v in self.items()})

        return m.binary_operation("add", self, other)

    def __eq__(self, other):
        if not self.rles.keys() == other.rles.keys():
            return False

        for c in self.rles.keys():
            if self.rles[c] != other.rles[c]:
                return False

        return True

    def __iter__(self):
        """Iterate over key and Rle.

        Examples
        --------
        >>> r = RleDict({("chr1", "+"): Rle([1, 1], [1, 2]),
        ...              ("chr1", "-"): Rle([1, 1], [3, 2.0])})
        >>> for k, v in r:
        ...     print(k)
        ...     print(v)
        ('chr1', '+')
        +--------+-----+-----+
        | Runs   | 1   | 1   |
        |--------+-----+-----|
        | Values | 1.0 | 2.0 |
        +--------+-----+-----+
        Rle of length 2 containing 2 elements (avg. length 1.0)
        ('chr1', '-')
        +--------+-----+-----+
        | Runs   | 1   | 1   |
        |--------+-----+-----|
        | Values | 3.0 | 2.0 |
        +--------+-----+-----+
        Rle of length 2 containing 2 elements (avg. length 1.0)
        """

        return iter(self.rles.items())

    def __getitem__(self, key):
        key_is_string = isinstance(key, str)
        key_is_int = isinstance(key, int)

        if key_is_int:
            raise Exception("Integer indexing not allowed!")

        if key_is_string and self.stranded and key not in ["+", "-"]:
            plus = self.rles.get((key, "+"), Rle())
            rev = self.rles.get((key, "-"), Rle())

            return RleDict({(key, "+"): plus, (key, "-"): rev})

        # only return particular strand, but from all chromos
        elif key_is_string and self.stranded and key in ["+", "-"]:
            to_return = dict()
            for (c, s), rle in self.items():
                if s == key:
                    to_return[c, s] = rle

            if len(to_return) > 1:
                return RleDict(to_return)
            else:  # return just the rle
                return list(to_return.values())[0]

        elif key_is_string:
            return self.rles.get(key, Rle())

        elif "PyRanges" in str(type(key)):  # hack to avoid isinstance(key, pr.PyRanges) so that we
            # do not need a dep on PyRanges in this library

            import pandas as pd
            import pyranges as pr  # type: ignore

            if not len(key):
                return pd.DataFrame(columns="Chromosome Start End ID Run Value".split())

            result = {}
            for k, v in key.dfs.items():
                if k not in self.rles:
                    continue

                v = v["Start End".split()].astype(np.int)
                ids, starts, ends, runs, values = getitems(
                    self.rles[k].runs, self.rles[k].values, v.Start.values, v.End.values
                )

                df = pd.DataFrame(
                    {
                        "Start": starts,
                        "End": ends,
                        "ID": ids,
                        "Run": runs,
                        "Value": values,
                    }
                )

                if isinstance(k, tuple):
                    df.insert(0, "Chromosome", k[0])
                    df.insert(df.shape[1], "Strand", k[1])
                else:
                    df.insert(0, "Chromosome", k)

                result[k] = df

            return pr.PyRanges(result)

        elif len(key) == 2:
            return self.rles.get(key, Rle([1], [0]))

        else:
            raise IndexError("Must use chromosome, strand or (chromosome, strand) to get items from RleDict.")

    def __len__(self):
        """Return number of keys in RleDict."""
        return len(self.rles)

    def __mul__(self, other):
        if isinstance(other, Number):
            return RleDict({cs: v * other for cs, v in self.items()})

        return m.binary_operation("mul", self, other)

    def __radd__(self, other):
        return RleDict({cs: other + v for cs, v in self.items()})

    def __repr__(self):
        return str(self)

    def __rsub__(self, other):
        return RleDict({cs: other - v for cs, v in self.items()})

    def __rtruediv__(self, other):
        return RleDict({cs: other / v for cs, v in self.items()})

    def __rmul__(self, other):
        return RleDict({cs: other * v for cs, v in self.items()})

    def __setitem__(self, key, item):
        self.rles[key] = item

    def __str__(self):
        if not self.rles:
            return "Empty RleDict."

        keys = natsorted(self.rles.keys())
        stranded = True if len(list(keys)[0]) == 2 else False

        if not stranded:
            if len(keys) > 2:
                str_list = [
                    keys[0],
                    str(self.rles[keys[0]]),
                    "...",
                    keys[-1],
                    str(self.rles[keys[-1]]),
                    "Unstranded RleDict object with {} chromosomes.".format(len(self.rles.keys())),
                ]
            elif len(keys) == 2:
                str_list = [
                    keys[0],
                    "-" * len(keys[0]),
                    str(self.rles[keys[0]]),
                    "",
                    keys[-1],
                    "-" * len(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "Unstranded RleDict object with {} chromosomes.".format(len(self.rles.keys())),
                ]
            else:
                str_list = [
                    keys[0],
                    str(self.rles[keys[0]]),
                    "Unstranded RleDict object with {} chromosome.".format(len(self.rles.keys())),
                ]

        else:
            if len(keys) > 2:
                str_list = [
                    " ".join(keys[0]),
                    str(self.rles[keys[0]]),
                    "...",
                    " ".join(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "RleDict object with {} chromosomes/strand pairs.".format(len(self.rles.keys())),
                ]
            elif len(keys) == 2:
                str_list = [
                    " ".join(keys[0]),
                    "-" * len(keys[0]),
                    str(self.rles[keys[0]]),
                    "",
                    " ".join(keys[-1]),
                    "-" * len(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "RleDict object with {} chromosomes/strand pairs.".format(len(self.rles.keys())),
                ]
            else:
                str_list = [
                    " ".join(keys[0]),
                    str(self.rles[keys[0]]),
                    "RleDict object with {} chromosome/strand pairs.".format(len(self.rles.keys())),
                ]

        outstr = "\n".join(str_list)

        return outstr

    def __sub__(self, other):
        if isinstance(other, Number):
            return RleDict({cs: v - other for cs, v in self.items()})

        return m.binary_operation("sub", self, other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return RleDict({cs: v / other for cs, v in self.items()})

        return m.binary_operation("div", self, other)

    def add(self, other, nb_cpu=1):
        """Add two RleDicts.

        Same as +, but add takes nb_cpu argument."""

        return m.binary_operation("add", self, other, nb_cpu)

    def add_pseudocounts(self, pseudo=0.01):
        for k, rle in self.items():
            rle.values.loc[rle.values == 0] = pseudo

    def apply(self, f, defragment=True):
        """Apply function to each Rle.

        Parameters
        ----------
        f : callable

            Takes and returns an Rle

        defragment : bool, default True

            Merge consecutive runs of equal values afterwards.

        **kwargs :

            Arguments given to f.

        See Also
        --------

        pyrle.RleDict.__array_ufunc__ : apply numpy function to RleDict

        Examples
        --------
        >>> r = RleDict({("chr1", "+"): Rle([1, 4], [1, 2]),
        ...              ("chr1", "-"): Rle([2, 1], [3, 2.0])})
        >>> def nonsense(rle):
        ...     rle.runs = rle.runs[::-1].copy()
        ...     rle.values = np.sqrt(rle.values)
        ...     return rle
        >>> r.apply(nonsense)
        chr1 +
        --
        +--------+-----+--------------------+
        | Runs   | 4   | 1                  |
        |--------+-----+--------------------|
        | Values | 1.0 | 1.4142135381698608 |
        +--------+-----+--------------------+
        Rle of length 5 containing 2 elements (avg. length 2.5)
        <BLANKLINE>
        chr1 -
        --
        +--------+--------------------+--------------------+
        | Runs   | 1                  | 2                  |
        |--------+--------------------+--------------------|
        | Values | 1.7320508075688772 | 1.4142135381698608 |
        +--------+--------------------+--------------------+
        Rle of length 3 containing 2 elements (avg. length 1.5)
        RleDict object with 2 chromosomes/strand pairs.
        """

        new_rles = {}

        for k, r in self:
            new_rle = r.copy()
            # new_rle.runs = f(new_rle.runs).astype(np.int)
            new_rle = f(new_rle)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return RleDict(new_rles)

    def apply_runs(self, f, defragment=True):
        """Apply a function to the runs of RleDict.

        Parameters
        ----------
        f : callable

            Takes the runs and returns an array of type int64 with same length as the input.

        defragment : bool, default True

            Merge consecutive runs of equal values afterwards.

        **kwargs :

            Arguments given to f.

        See Also
        --------

        pyrle.RleDict.apply_values : apply function to values of RleDict

        Examples
        --------
        >>> r = RleDict({("chr1", "+"): Rle([1, 4], [1, 2]),
        ...              ("chr1", "-"): Rle([2, 1], [3, 2.0])})
        >>> def even_times_hundred(runs):
        ...     runs[runs % 2 == 0] *= 100
        ...     return runs
        >>> r.apply_runs(even_times_hundred)
        chr1 +
        --
        +--------+-----+-------+
        | Runs   | 1   | 400   |
        |--------+-----+-------|
        | Values | 1.0 | 2.0   |
        +--------+-----+-------+
        Rle of length 401 containing 2 elements (avg. length 200.5)
        <BLANKLINE>
        chr1 -
        --
        +--------+-------+-----+
        | Runs   | 200   | 1   |
        |--------+-------+-----|
        | Values | 3.0   | 2.0 |
        +--------+-------+-----+
        Rle of length 201 containing 2 elements (avg. length 100.5)
        RleDict object with 2 chromosomes/strand pairs.
        """

        new_rles = {}

        for k, r in self:
            new_rle = r.copy()
            new_rle.runs = f(new_rle.runs).astype(np.int64)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return RleDict(new_rles)

    def apply_values(self, f, defragment=True, **kwargs):
        """Apply a function to the values of each Rle.

        Parameters
        ----------
        f : callable

            Takes the values and returns an array of type double with the same length as the input.

        defragment : bool, default True

            Merge consecutive runs of equal values afterwards.

        **kwargs :

            Arguments given to f.

        See Also
        --------

        pyrle.RleDict.__array_ufunc__ : apply numpy function to RleDict

        Examples
        --------
        >>> r = RleDict({("chr1", "+"): Rle([1, 1], [1, 2]),
        ...              ("chr1", "-"): Rle([1, 1], [3, 2.0])})
        >>> f = lambda v, **kwargs: v ** kwargs["exponent"]
        >>> r.apply_values(f, exponent=3)
        chr1 +
        --
        +--------+-----+-----+
        | Runs   | 1   | 1   |
        |--------+-----+-----|
        | Values | 1.0 | 8.0 |
        +--------+-----+-----+
        Rle of length 2 containing 2 elements (avg. length 1.0)
        <BLANKLINE>
        chr1 -
        --
        +--------+------+-----+
        | Runs   | 1    | 1   |
        |--------+------+-----|
        | Values | 27.0 | 8.0 |
        +--------+------+-----+
        Rle of length 2 containing 2 elements (avg. length 1.0)
        RleDict object with 2 chromosomes/strand pairs.
        """

        new_rles = {}

        for k, r in self:
            new_rle = r.copy()
            new_rle.values = f(new_rle.values, **kwargs).astype(np.double)

            new_rle = new_rle.defragment()

            new_rles[k] = new_rle

        return RleDict(new_rles)

    @property
    def chromosomes(self):
        cs = []

        for k in self.rles:
            if isinstance(k, tuple):
                cs.append(k[0])
            else:
                cs.append(k)

        return natsorted(set(cs))

    def copy(self):
        d = {}
        for k, r in self:
            d[k] = r.copy()

        return RleDict(d)

    def defragment(self, numbers_only=False):
        if not numbers_only:
            d = {k: v.defragment() for k, v in self.items()}
        else:
            d = {k: v.numbers_only().defragment() for k, v in self.items()}

        return RleDict(d)

    def div(self, other, nb_cpu=1):
        """Divide two RleDicts.

        Same as /, but div takes nb_cpu argument."""

        return m.binary_operation("div", self, other, nb_cpu)

    def items(self):
        _items = list(self.rles.items())

        return natsorted(_items, key=lambda x: x[0])

    def keys(self):
        return natsorted(list(self.rles.keys()))

    def make_strands_same_length(self, fill_value=0):
        self = self.copy()

        if not self.stranded:
            return self

        for c in self.chromosomes:
            p = self[c]["+"]
            n = self[c]["-"]
            pl = p.length
            nl = n.length
            diff = abs(pl - nl)

            if pl > nl:
                if n.values[-1] == fill_value:
                    n.runs[-1] += diff
                else:
                    n.runs = np.r_[n.runs, diff]
                    n.values = np.r_[n.values, fill_value]
            elif pl < nl:
                if p.values[-1] == fill_value:
                    p.runs[-1] += diff
                else:
                    p.runs = np.r_[p.runs, diff]
                    p.values = np.r_[p.values, fill_value]

        return self

    def mul(self, other, nb_cpu=1):
        """Multiply two RleDicts.

        Same as *, but mul takes nb_cpu argument."""

        return m.binary_operation("mul", self, other, nb_cpu)

    def numbers_only(self):
        return RleDict({k: v.numbers_only() for k, v in self.items()})

    def shift(self, distance):
        return self.apply(lambda r: r.shift(distance))

    def sub(self, other, nb_cpu=1):
        """Subtract two RleDicts.

        Same as -, but sub takes nb_cpu argument."""

        return m.binary_operation("sub", self, other, nb_cpu)

    @property
    def stranded(self):
        if len(self) == 0:
            return True

        return isinstance(self.keys()[0], tuple)

    def to_csv(self, f, sep="\t"):
        self.to_table().to_csv(f, sep=sep, index=False)

    def to_ranges(self, stranded=None):
        """Turn RleDict into PyRanges.

        Parameters
        ----------
        stranded : bool, default None, i.e. auto

            Whether to return stranded PyRanges.

        Example
        -------
        >>> r = RleDict({("chr1", "+"): Rle([1, 1], [1, 2]),
        ...              ("chr1", "-"): Rle([1, 1], [3, 2.0])})
        >>> r.to_ranges()
        +--------------+-----------+-----------+-------------+--------------+
        | Chromosome   |     Start |       End |       Score | Strand       |
        | (category)   |   (int64) |   (int64) |   (float64) | (category)   |
        |--------------+-----------+-----------+-------------+--------------|
        | chr1         |         0 |         1 |           1 | +            |
        | chr1         |         1 |         2 |           2 | +            |
        | chr1         |         0 |         1 |           3 | -            |
        | chr1         |         1 |         2 |           2 | -            |
        +--------------+-----------+-----------+-------------+--------------+
        Stranded PyRanges object has 4 rows and 5 columns from 1 chromosomes.
        For printing, the PyRanges was sorted on Chromosome and Strand.
        """

        dtypes = {"Chromosome": "category", "Start": np.int64, "End": np.int64}
        if self.stranded:
            dtypes["Strand"] = "category"

        return m.to_ranges(self).apply(lambda df: df.astype(dtypes))

    def to_table(self):
        import pandas as pd

        dfs = []
        for k, r in self.rles.items():
            df = pd.DataFrame(data={"Runs": r.runs, "Values": r.values})
            if self.stranded:
                df.insert(0, "Chromosome", k[0])
                df.insert(1, "Strand", k[1])
            else:
                df.insert(0, "Chromosome", k)

            dfs.append(df)

        return pd.concat(dfs)

    def values(self):
        return [self.rles[k] for k in natsorted(self.rles.keys())]


if __name__ == "__main__":
    # Must turn on macros in setup.py for line tracing to work
    "kernprof -l pyrle/rledict.py && python -m line_profiler coverage.py.lprof"

    import datetime
    from time import time

    import pandas as pd

    test_file = "/mnt/scratch/endrebak/genomes/chip/UCSD.Aorta.Input.STL002.bed.gz"

    nrows = None
    df = pd.read_table(
        test_file,
        sep="\t",
        usecols=[0, 1, 2, 5],
        header=None,
        names="Chromosome Start End Strand".split(),
        nrows=nrows,
    )

    print("Done reading")
    start = time()

    result = RleDict(df, stranded=True)

    end = time()
    total = end - start

    total_dt = datetime.datetime.fromtimestamp(total)

    minutes_seconds = total_dt.strftime("%M\t%S\n")

    print(result)
    print(minutes_seconds)
