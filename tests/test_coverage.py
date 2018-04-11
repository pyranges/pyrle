"""
This code is silly bad and erroneus. Still checking out ways to do it in pure python.
"""

import pytest

import pandas as pd
import numpy as np

from pyrle import Rle
from io import StringIO

from heapq import heappush, heappop, heapify


def coverage(ranges):

    try:
        df = ranges.df
    except:
        df = ranges

    starts = (df.Start).tolist()
    ends = (df.End).tolist()

    heapify(starts), heapify(ends)
    counter = 0
    last_counter = 0
    pos = 0

    runs = []
    values = []

    if starts[0] < ends[0]:
        pos = heappop(starts)
        runs.append(pos)
        values.append(counter)
        counter += 1
    if starts[0] == ends[0] and starts[0] != 0:
        pos = heappop(starts)
        heappop(ends)
        runs.append(pos)
        values.append(counter)

    print("starts")
    print(runs)
    print(values)
    while len(starts) > 0:

        min_start, min_end = starts[0], ends[0]
        min_pos = min(min_start, min_end)
        rl = min_pos - pos
        runs.append(rl)
        values.append(counter)

        if min_start < min_end:
            counter += 1
            next_pos = heappop(starts)
            pos = next_pos
        elif min_start > min_end:
            counter -= 1
            next_pos = heappop(ends)
            pos = next_pos
        else:
            next_pos = heappop(starts)
            heappop(ends)

        if min_start == min_end:
            counter += 1

    print(counter)
    print(runs)
    print(values)
    print("Unwinding ends")
    i = 0
    print(pos)
    while len(ends) > 0:
        print("iter", i)
        i += 1

        next_pos = heappop(ends)
        rl = next_pos - pos
        print("rl", rl)
        pos = next_pos

        print("counter", counter)
        if counter == values[-1]:
            runs.append(rl)
        values.append(counter)

        counter -= 1
        while len(ends) > 0 and ends[0] == pos:
            counter -= 1
            heappop(ends)

    print(ends)
    if counter == 1:
        runs.append(1)
        values.append(1)

    return Rle(runs, values)







def test_coverage():


    pass




@pytest.fixture
def supersimple_bed():

    c = """Start End
2 2"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_supersimple_bed():

    runs = np.array([2, 1], dtype=np.int)
    values = np.array([0, 1], dtype=np.float)

    return Rle(runs, values)



@pytest.fixture
def supersimple_bed2():

    c = """Start End
2 2
4 4"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_supersimple_bed2():

    runs = np.array([2, 1, 1, 1], dtype=np.int)
    values = np.array([0, 1, 0, 1], dtype=np.float)

    return Rle(runs, values)


def test_supersimple_bed2(supersimple_bed2, expected_result_supersimple_bed2):

    result = coverage(supersimple_bed2)
    print(result.runs, expected_result_supersimple_bed2.runs)
    print(result.values, expected_result_supersimple_bed2.values)

    assert np.allclose(result.runs, expected_result_supersimple_bed2.runs)
    assert np.allclose(result.values, expected_result_supersimple_bed2.values)


@pytest.fixture
def simple_bed():

    c = """Start End
3 6
4 7
5 6"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_simple_bed():

    runs = np.array([3, 1, 1, 2, 1], dtype=np.int)
    values = np.array([0, 1, 2, 3, 1], dtype=np.float)

    return Rle(runs, values)


def test_simple_bed(simple_bed, expected_result_simple_bed):

    result = coverage(simple_bed)
    print(result.runs, expected_result_simple_bed.runs)
    print(result.values, expected_result_simple_bed.values)
    assert 0

    assert np.allclose(result.runs, expected_result_simple_bed.runs)
    assert np.allclose(result.values, expected_result_simple_bed.values)




# @pytest.fixture
# def simple_bed2():

#     c = """Start End
# 3 6
# 5 7
# 6 6"""

#     return pd.read_table(StringIO(c), sep="\s+", header=0)


# @pytest.fixture()
# def expected_result_simple_bed2():

#     runs = np.array([3, 2, 1, 1, 1], dtype=np.int)
#     values = np.array([0, 1, 2, 3, 1], dtype=np.float)

#     return Rle(runs, values)


# def test_simple_bed2(simple_bed2, expected_result_simple_bed2):

#     result = coverage(simple_bed2)
#     print(result.runs, expected_result_simple_bed2.runs)
#     print(result.values, expected_result_simple_bed2.values)

#     assert np.allclose(result.runs, expected_result_simple_bed2.runs)
#     assert np.allclose(result.values, expected_result_simple_bed2.values)
#     # assert 0



# @pytest.fixture
# def simple_bed3():

#     c = """Start End
# 3 5
# 5 6
# 6 7"""

#     return pd.read_table(StringIO(c), sep="\s+", header=0)


# @pytest.fixture()
# def expected_result_simple_bed3():

#     runs = np.array([3, 2, 2, 1], dtype=np.int)
#     values = np.array([0, 1, 2, 1], dtype=np.float)

#     return Rle(runs, values)


# def test_simple_bed3(simple_bed3, expected_result_simple_bed3):

#     result = coverage(simple_bed3)
#     print(result.runs, expected_result_simple_bed3.runs)
#     print(result.values, expected_result_simple_bed3.values)

#     assert np.allclose(result.runs, expected_result_simple_bed3.runs)
#     assert np.allclose(result.values, expected_result_simple_bed3.values)
