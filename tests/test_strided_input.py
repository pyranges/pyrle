import numpy as np

from pyrle.src.coverage import _coverage
from pyrle.src.getitem import getitems


def strided(values, dtype):
    # A column of a row-major 2D array, as pandas >= 3.1 can return from Series.values.
    block = np.zeros((len(values), 3), dtype=dtype)
    block[:, 1] = values
    column = block[:, 1]
    assert not column.flags.c_contiguous
    return column


def test_getitems_accepts_strided_input():
    runs, values = np.array([2, 3, 4], dtype=np.int_), np.array([1.0, 0.0, 2.0])
    starts, ends = [0, 3], [4, 8]
    got = getitems(runs, values, strided(starts, np.int_), strided(ends, np.int_))
    want = getitems(runs, values, np.array(starts, dtype=np.int_), np.array(ends, dtype=np.int_))
    for g, w in zip(got, want):
        assert list(g) == list(w)


def test_coverage_accepts_strided_input():
    positions, values = [1, 3, 5, 8], [1.0, 1.0, -1.0, -1.0]
    got = _coverage(strided(positions, np.int_), strided(values, np.float64))
    want = _coverage(np.array(positions, dtype=np.int_), np.array(values))
    for g, w in zip(got, want):
        assert list(g) == list(w)
