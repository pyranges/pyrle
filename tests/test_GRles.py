
import pytest

import pandas as pd
import numpy as np

from pyrle import Rle
from io import StringIO

from pyrle import GRles
from pyrle.methods import coverage



@pytest.fixture()
def expected_result_coverage():

    runs = """9739215      25  463205      25 3069430      25    9143      25  993038 25  142071      25  260968      25   71512      25   18072      25 103292 25""".split()
    runs = [int(s) for s in runs]
    values = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    return runs, values


@pytest.fixture()
def df():

    c = """chr21	9739215	9739240	U0	0	-
chr21	10202445	10202470	U0	0	+
chr21	13271900	13271925	U0	0	-
chr21	13281068	13281093	U0	0	-
chr21	14274131	14274156	U0	0	-
chr21	14416227	14416252	U0	0	+
chr21	14677220	14677245	U0	0	-
chr21	14748757	14748782	U0	0	+
chr21	14766854	14766879	U0	0	+
chr21	14870171	14870196	U0	0	-"""

    return pd.read_table(StringIO(c), header=None, names="Chromosome Start End Name Score Strand".split())


@pytest.fixture()
def grle1():

    r1 = Rle([1, 5, 10], [1, 2, 3])
    r2 = Rle([4, 7, 9], [0.01, 0.02, 0.03])

    d = {"chr1": r1, "chr2": r2}
    print(d)

    return GRles(d)


@pytest.fixture()
def grle2():

    r1 = Rle([1, 2, 3], [1, 2, 3])
    r2 = Rle([5, 4, 2], [-0.1, -0.2, -0.3])

    d = {"chr1": r1, "chr2": r2}

    return GRles(d)

@pytest.fixture
def expected_result():

    r1 = Rle([1, 2, 3, 10], [2, 4, 5, 3])
    r2 = Rle([4, 1, 4, 2, 9], [-0.09, -0.08, -0.18, -0.28, 0.03])

    d = {"chr1": r1, "chr2": r2}

    return GRles(d)


@pytest.fixture
def d1():

    r1 = Rle([1, 5, 10], [1, 2, 3])
    r2 = Rle([4, 7, 9], [0.01, 0.02, 0.03])
    print(r1)

    d = {"chr1": r1, "chr2": r2}

    return d


def test_create_GRles(d1):

    # No errors
    result = GRles(d1)



def test_add_GRles(grle1, grle2, expected_result):

    result = grle1.add(grle2)
    print(result)
    print("----")
    print(expected_result)

    assert result == expected_result
