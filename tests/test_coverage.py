"""
This code is silly bad and erroneus. Still checking out ways to do it in pure python.
"""

import pytest

import pandas as pd
import numpy as np

from pyrle import Rle
from io import StringIO

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
    runs = pd.concat([starts, ends], ignore_index=True)
    values = runs.groupby("Position").sum().reset_index().drop_duplicates()[value_col]
    first_value = values.iloc[0] if starts.Position.min() == 0 else 0
    run_lengths = (runs.Position - runs.Position.shift().fillna(0))[:-1]

    values = values.cumsum().shift()
    values[0] = first_value

    return Rle(run_lengths, values)


@pytest.fixture
def supersimple_bed():

    c = """Start End
2 3"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


# def test_coverage(supersimple_bed):

#     result = coverage(supersimple_bed)

#     assert list(result.runs) == [2, 1]
#     assert list(result.values) == [0, 1]


# @pytest.fixture
# def empty_bed():

#     c = """Start End
# 2 2
# 4 4"""

#     return pd.read_table(StringIO(c), sep="\s+", header=0)


# @pytest.fixture()
# def expected_result_supersimple_bed2():

#     runs = np.array([2, 1, 1, 1], dtype=np.int)
#     values = np.array([0, 1, 0, 1], dtype=np.float)

#     return Rle(runs, values)


# def test_supersimple_bed2(empty_bed, expected_result_supersimple_bed2):

#     result = coverage(empty_bed)
#     print(result.runs)
#     print(result.values)

#     assert list(result.runs) == [2, 2]


@pytest.fixture
def simple_bed():

    c = """Start End Value
3 6 2.4
4 7 0.9
5 6 3.33"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_simple_bed():

    runs = np.array([3, 1, 1, 1, 1], dtype=np.int)
    values = np.array([0, 1, 2, 3, 1], dtype=np.float)

    return Rle(runs, values)

@pytest.fixture()
def expected_result_simple_bed_values():

    runs = np.array([3, 1, 1, 1, 1], dtype=np.int)
    values = np.array([0., 2.4, 3.3, 6.63, 0.9], dtype=np.float)

    return Rle(runs, values)


def test_simple_bed(simple_bed, expected_result_simple_bed):

    result = coverage(simple_bed)
    print(result.runs, expected_result_simple_bed.runs)
    print(result.values, expected_result_simple_bed.values)
    assert list(result.runs) == list(expected_result_simple_bed.runs)
    assert list(result.values) == list(expected_result_simple_bed.values)


def test_simple_bed_with_scores(simple_bed, expected_result_simple_bed_values):

    result = coverage(simple_bed, value_col="Value")
    print(result.runs, expected_result_simple_bed_values.runs)
    print(result.values, expected_result_simple_bed_values.values)
    assert list(result.runs) == list(expected_result_simple_bed_values.runs)
    assert np.allclose(result.values, expected_result_simple_bed_values.values)
