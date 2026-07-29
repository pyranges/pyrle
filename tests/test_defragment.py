import numpy as np

from pyrle import Rle
from pyrle.src.coverage import _remove_dupes


def test_defragment_keeps_a_single_run_value():
    """A result that collapses to one run must keep its value.

    `_remove_dupes` returned the typed memoryviews it was handed on its
    single-element path, while every other path returned numpy arrays.
    `defragment` then evaluated `values == -0` on a memoryview, which compares
    the object rather than its elements and yields a scalar `False`; that
    indexes element 0 instead of masking, so the value was overwritten with 0.
    """
    assert list(Rle(np.array([5]), np.array([3.0])).defragment().values) == [3.0]
    assert list(Rle(np.array([1]), np.array([1.0])).defragment().values) == [1.0]
    # Two equal runs collapse to one, and the surviving value is still right.
    merged = Rle(np.array([1, 1]), np.array([2.0, 2.0])).defragment()
    assert list(merged.runs) == [2]
    assert list(merged.values) == [2.0]


def test_defragment_keeps_a_single_infinite_run():
    merged = Rle(np.array([1, 1]), np.array([np.inf, np.inf])).defragment()
    assert list(merged.runs) == [2]
    assert list(merged.values) == [np.inf]


def test_defragment_still_normalises_negative_zero():
    values = Rle(np.array([1]), np.array([-0.0])).defragment().values
    assert list(values) == [0.0]
    assert not np.signbit(values[0])


def test_defragment_still_merges_and_keeps_distinct_runs():
    merged = Rle(np.array([1, 1, 1]), np.array([5.0, 5.0, 7.0])).defragment()
    assert list(merged.runs) == [2, 1]
    assert list(merged.values) == [5.0, 7.0]

    kept = Rle(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])).defragment()
    assert list(kept.runs) == [1, 2, 3]
    assert list(kept.values) == [1.0, 2.0, 3.0]

    nans = Rle(np.array([1, 1, 1]), np.array([np.nan, np.nan, 2.0])).defragment()
    assert list(nans.runs) == [2, 1]
    assert np.isnan(nans.values[0])
    assert nans.values[1] == 2.0


def test_remove_dupes_returns_arrays_on_every_path():
    """The single-element path used to leak typed memoryviews to callers."""
    for runs, values in (([5], [3.0]), ([1, 1], [1.0, 2.0])):
        out_runs, out_values = _remove_dupes(
            np.array(runs, dtype=np.int_), np.array(values, dtype=np.double), len(values)
        )
        assert isinstance(out_runs, np.ndarray)
        assert isinstance(out_values, np.ndarray)
        # An elementwise comparison, not a scalar bool.
        assert (out_values == 0).shape == out_values.shape
