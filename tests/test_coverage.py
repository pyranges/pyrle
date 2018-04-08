"""
This code is silly bad and erroneus. Still checking out ways to do it in pure python.
"""

import pytest

import pandas as pd
import numpy as np

from pyrle import Rle
from io import StringIO

from heapq import heappush, heappop, heapify

def coverage(df, is_sorted=False):

    if not is_sorted:
        df = df.sort_values(["Start", "End"])

    starts = df.Start.astype(np.int)
    ends = df.End.astype(np.int)

    cv_rle = _coverage(starts, ends)

    return cv_rle


def unwind_minq(last_start, minq, counter):

    values, runs = [], []

    last_min_val = last_start
    print(minq)
    while minq:
        min_val = heappop(minq)
        values.append(counter)
        counter -= 1
        while minq:
            if minq[0] == min_val:
                print("minq[0] == min_val " * 3)
                counter -= 1
                heappop(minq)
            else:
                break


        print("Hiya")
        print("min_val, last_min_val", min_val, last_min_val)

        next_length = min_val - last_min_val

        print("next value, last value", counter, values[-1])
        if runs:
            print("next length, last length", next_length, runs[-1])
        runs.append(next_length)

        if counter == 0:
            return runs, values

        last_min_val = min_val

    return runs, values



def _coverage(starts, ends):

    se = iter(zip(starts, ends))
    start, end = next(se)
    counter = 0

    if start != 0:
        runs = [start]
        values = [0]
        counter += 1
        minq = [end]
        heapify(minq)
        last_start = start
    else:
        runs = []
        values = []
        last_start = 0

    print("Here")
    for start, end in se:

        counter -= 1
        print("Now evaluating", start, end)
        heappush(minq, end)
        if minq:
            print("In minq")
            peek = minq[0]
            print("Peek:", peek)
            if start < peek:
                print("Start < peek")
                print(start, "<", peek)
                counter += 1
                next_length = start - last_start
                last_start = start
                values.append(counter)
            elif start > peek:
                print("Start > peek")
                print(start, ">", peek)
                min_end = heappop(minq)
                next_length = start - last_start + 1
                values.append(counter + 1)
                counter -= 1
            else:
                print("Start = peek")
                counter += 1
                next_length = start - last_start
                last_start = start
                values.append(counter)

            runs.append(next_length)
            last_start = start

        else:
            raise Exception("Not in minq")

        counter += 1

    last_start -= 1

    print("When done runs are", runs, "and values are", values, "while counter is", counter)
    print("Last start", last_start, "Heap", minq)

    last_min_val = last_start
    final_runs, final_values = unwind_minq(last_start, minq, counter)

    runs = runs + final_runs
    values = values + final_values

    return Rle(runs, values)


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

    assert np.allclose(result.runs, expected_result_simple_bed.runs)
    assert np.allclose(result.values, expected_result_simple_bed.values)




@pytest.fixture
def simple_bed2():

    c = """Start End
3 6
5 7
6 6"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_simple_bed2():

    runs = np.array([3, 2, 1, 1, 1], dtype=np.int)
    values = np.array([0, 1, 2, 3, 1], dtype=np.float)

    return Rle(runs, values)


def test_simple_bed2(simple_bed2, expected_result_simple_bed2):

    result = coverage(simple_bed2)
    print(result.runs, expected_result_simple_bed2.runs)
    print(result.values, expected_result_simple_bed2.values)

    assert np.allclose(result.runs, expected_result_simple_bed2.runs)
    assert np.allclose(result.values, expected_result_simple_bed2.values)
    # assert 0



@pytest.fixture
def simple_bed3():

    c = """Start End
3 5
5 6
6 7"""

    return pd.read_table(StringIO(c), sep="\s+", header=0)


@pytest.fixture()
def expected_result_simple_bed3():

    runs = np.array([3, 2, 2, 1], dtype=np.int)
    values = np.array([0, 1, 2, 1], dtype=np.float)

    return Rle(runs, values)


def test_simple_bed3(simple_bed3, expected_result_simple_bed3):

    result = coverage(simple_bed3)
    print(result.runs, expected_result_simple_bed3.runs)
    print(result.values, expected_result_simple_bed3.values)

    assert np.allclose(result.runs, expected_result_simple_bed3.runs)
    assert np.allclose(result.values, expected_result_simple_bed3.values)


def test_unwind_minq_simple_bed1():

    # last_start, minq, counter
    l = [6, 7, 6]
    heapify(l)
    runs, values = unwind_minq(4, l, 3)
    print(runs, values)

    assert runs == [2, 1]
    assert values == [3, 1]


def test_unwind_minq_simple_bed2():

    # last_start, minq, counter
    l = [6, 7, 6]
    heapify(l)
    runs, values = unwind_minq(5, l, 3)
    print(runs, values)

    assert runs == [1, 1]
    assert values == [3, 1]


def test_unwind_minq_simple_bed3():

    runs, values = unwind_minq(5, [6, 7], 1)

    assert runs == [1]
    assert values == [1]


# @pytest.fixture
# def simple_bed4():

#     c = """Start End
# 127471196 127472363
# 127472363 127473530
# 127473530 127474697"""

#     return pd.read_table(StringIO(c), sep="\s+", header=0)


# @pytest.fixture()
# def expected_result_simple_bed4():

#     runs = np.array([127471196, 1167, 1, 1167], dtype=np.int)
#     values = np.array([0, 1, 2, 1], dtype=np.float)

#     return Rle(runs, values)


# def test_simple_bed4(simple_bed4, expected_result_simple_bed4):

#     result = coverage(simple_bed4)
#     print(result.runs, expected_result_simple_bed4.runs)
#     print(result.values, expected_result_simple_bed4.values)
#     assert 0

#     assert np.allclose(result.runs, expected_result_simple_bed4.runs)
#     assert np.allclose(result.values, expected_result_simple_bed4.values)


# @pytest.fixture
# def medium_bed3():

#     c = """Start End
# 127471196 127472363
# 127472363 127473530
# 127473530 127474697
# 127474697 127475864
# 127475864 127477031
# 127477031 127478198
# 127478198 127479365
# 127479365 127480532
# 127480532 127481699"""

#     return pd.read_table(StringIO(c), sep="\s+", header=0)


# @pytest.fixture()
# def expected_result_medium_bed3():

#     runs = np.array([127471196, 1167, 1, 1166, 1, 1166, 1, 1166, 1, 1166, 1, 1166, 1, 1166, 1, 1166, 1, 1167], dtype=np.int)
#     values = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=np.float)

#     return Rle(runs, values)


# def test_medium_bed3(medium_bed3, expected_result_medium_bed3):

#     result = coverage(medium_bed3)
#     print(result.runs, expected_result_medium_bed3.runs)
#     print(result.values, expected_result_medium_bed3.values)
#     assert 0

    # assert np.allclose(result.runs, expected_result_medium_bed3.runs)
    # assert np.allclose(result.values, expected_result_medium_bed3.values)
