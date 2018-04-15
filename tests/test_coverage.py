"""
This code is silly bad and erroneus. Still checking out ways to do it in pure python.
"""

import pytest

import pandas as pd
import numpy as np

from pyrle import Rle
from io import StringIO

from collections import defaultdict, OrderedDict
from heapq import heappush, heappop, heapify


def compressed_ints_with_counts(ints, count_value=1):

    result = defaultdict(int)

    counter = count_value
    last_i = ints[0]
    for i in ints[1::1]:

        if i == last_i:
            counter = counter + count_value
        else:
            result[last_i] = counter
            counter = count_value
            last_i = i

    result[last_i] = counter

    return result


def test_compressed_ints_with_counts():

    result = compressed_ints_with_counts([0, 3, 6, 7, 7, 9, 9], -1)

    print(result)
    assert result == {0: -1, 9: -2, 3: -1, 6: -1, 7: -2}

# does not work on unsorted data
# def test_compressed_ints_with_counts2():

#     result = compressed_ints_with_counts([6, 7, 6], -1)

#     print(result)
#     assert result == {6: -2, 7: -1}



def merge_compressed_ints(compressed_starts, compressed_ends):

    merged_counts = OrderedDict()

    keys = set(compressed_starts).union(compressed_ends)
    # first count must be > 0
    last_count = -1
    for i in keys:
        count = compressed_starts[i] + compressed_ends[i]

        merged_counts[i] = count
        last_count = count

    return merged_counts


def test_merge_compressed_ints():

    s = defaultdict(int, {3: 1, 6: 2, 7: 1})
    e = defaultdict(int, {5: -1, 8: -2, 9: -1})

    result = merge_compressed_ints(s, e)

    print(sorted(result.items()))

    assert sorted(result.items()) == [(3, 1), (5, -1), (6, 2), (7, 1), (8, -2), (9, -1)]


def test_merge_compressed_ints2():

    s = defaultdict(int, {3: 1, 4: 1, 5: 1})
    e = defaultdict(int, {6: -2, 7: -1})

    result = merge_compressed_ints(s, e)

    print(sorted(result.items()))

    assert sorted(result.items()) == [(3, 1), (4, 1), (5, 1), (6, -2), (7, -1)]

def coverage(ranges):

    try:
        df = ranges.df
    except:
        df = ranges

    starts = sorted((df.Start).tolist())
    ends = sorted((df.End).tolist())

    compressed_starts = compressed_ints_with_counts(starts)
    print("compressed_starts")
    print(compressed_starts)
    compressed_ends = compressed_ints_with_counts(ends, -1)
    print("compressed_ends")
    print(compressed_ends)

    all_counts = merge_compressed_ints(compressed_starts, compressed_ends)
    print("all_counts")
    print(all_counts)

    runs, values = [], []
    counter = 0
    last_run = 0
    for r, c in all_counts.items():
        print("r, c")
        print(r, c)
        runvalue = r - last_run
        last_run = r
        print("runvalue", runvalue)
        runs.append(runvalue), values.append(counter)
        counter += c
        print("counter", c)

    return Rle(runs, values)


@pytest.fixture
def supersimple_bed():

    c = """Start End
2 3"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


def test_coverage(supersimple_bed):

    result = coverage(supersimple_bed)

    assert list(result.runs) == [2, 1]
    assert list(result.values) == [0, 1]


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

    c = """Start End
3 6
4 7
5 6"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_simple_bed():

    runs = np.array([3, 1, 1, 1, 1], dtype=np.int)
    values = np.array([0, 1, 2, 3, 1], dtype=np.float)

    return Rle(runs, values)


def test_simple_bed(simple_bed, expected_result_simple_bed):

    result = coverage(simple_bed)
    print(result.runs, expected_result_simple_bed.runs)
    print(result.values, expected_result_simple_bed.values)
    assert list(result.runs) == list(expected_result_simple_bed.runs)
    assert list(result.values) == list(expected_result_simple_bed.values)

#     assert np.allclose(result.runs, expected_result_simple_bed.runs)
#     assert np.allclose(result.values, expected_result_simple_bed.values)




# # @pytest.fixture
# # def simple_bed2():

# #     c = """Start End
# # 3 6
# # 5 7
# # 6 6"""

# #     return pd.read_table(StringIO(c), sep="\s+", header=0)


# # @pytest.fixture()
# # def expected_result_simple_bed2():

# #     runs = np.array([3, 2, 1, 1, 1], dtype=np.int)
# #     values = np.array([0, 1, 2, 3, 1], dtype=np.float)

# #     return Rle(runs, values)


# # def test_simple_bed2(simple_bed2, expected_result_simple_bed2):

# #     result = coverage(simple_bed2)
# #     print(result.runs, expected_result_simple_bed2.runs)
# #     print(result.values, expected_result_simple_bed2.values)

# #     assert np.allclose(result.runs, expected_result_simple_bed2.runs)
# #     assert np.allclose(result.values, expected_result_simple_bed2.values)
# #     # assert 0



# # @pytest.fixture
# # def simple_bed3():

# #     c = """Start End
# # 3 5
# # 5 6
# # 6 7"""

# #     return pd.read_table(StringIO(c), sep="\s+", header=0)


# # @pytest.fixture()
# # def expected_result_simple_bed3():

# #     runs = np.array([3, 2, 2, 1], dtype=np.int)
# #     values = np.array([0, 1, 2, 1], dtype=np.float)

# #     return Rle(runs, values)


# # def test_simple_bed3(simple_bed3, expected_result_simple_bed3):

# #     result = coverage(simple_bed3)
# #     print(result.runs, expected_result_simple_bed3.runs)
# #     print(result.values, expected_result_simple_bed3.values)

# #     assert np.allclose(result.runs, expected_result_simple_bed3.runs)
# #     assert np.allclose(result.values, expected_result_simple_bed3.values)
