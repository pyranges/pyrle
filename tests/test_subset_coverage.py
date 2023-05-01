import pytest

from hypothesis import given, settings, reproduce_failure, HealthCheck, seed

from tests.hypothesis_helper import (
    runlengths,
    dfs_min,
    runlengths_same_length_integers,
    _slice,
)

from itertools import product
import tempfile
import subprocess
from io import StringIO

from pyrle import Rle

import pandas as pd
import numpy as np

# using assert df equal, because we want to consider output from bedtools and
# pyranges equal even if they have different sort order
from tests.helpers import assert_df_equal

import numpy as np

from os import environ

if environ.get("TRAVIS"):
    max_examples = 100
    deadline = None
else:
    max_examples = 100
    deadline = None

rle_operation_cmd = "Rscript --vanilla tests/subset_coverage.R {} {} {} {}"


@pytest.mark.r
@given(runlengths=runlengths, interval=_slice())
@settings(
    max_examples=max_examples,
    deadline=deadline,
    suppress_health_check=HealthCheck.all(),
)
def test_subset_coverage(runlengths, interval):
    start, end = interval

    print("runlengths\n", runlengths)

    r = Rle(runlengths.Runs, runlengths.Values)

    result_pyranges = r[start:end]

    result_df = None
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir = "."
        f1 = "{}/f1.txt".format(temp_dir)
        outfile = "{}/result.txt".format(temp_dir)
        runlengths.to_csv(f1, sep="\t", index=False)

        cmd = rle_operation_cmd.format(f1, start + 1, end, outfile)  # + " 2>/dev/null"
        print(cmd)

        subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode()

        result = pd.read_csv(outfile, sep="\t")
        s4vectors_result = Rle(result.Runs, result.Values)

    print("pyranges result\n", result_pyranges)
    print("s4vectors result\n", s4vectors_result)

    assert np.allclose(result_pyranges.runs, s4vectors_result.runs, equal_nan=False)
    assert np.allclose(result_pyranges.values, s4vectors_result.values, equal_nan=True)


rle_operation_cmd = "Rscript --vanilla tests/subset_coverage.R {} {} {} {}"

# @pytest.mark.r
# @given(runlengths=runlengths, interval=_slice())
# @settings(max_examples=max_examples, deadline=deadline, timeout=unlimited, suppress_health_check=HealthCheck.all())
# def test_getloc_coverage(runlengths, interval):

#     start, end = interval
#     # Only compared against bioc with integers because float equality is hard,
#     # for both libraries, sometimes end up with slightly different runlengths
#     # when consecutive values are almost equal

#     print("runlengths\n", runlengths)

#     r = Rle(runlengths.Runs, runlengths.Values)

#     result_pyranges = r[start:end]

#     result_df = None
#     with tempfile.TemporaryDirectory() as temp_dir:
#         f1 = "{}/f1.txt".format(temp_dir)
#         outfile = "{}/result.txt".format(temp_dir)
#         runlengths.to_csv(f1, sep="\t", index=False)

#         cmd = rle_operation_cmd.format(f1, start, end, outfile) # + " 2>/dev/null"
#         print(cmd)

#         subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode()

#         result = pd.read_csv(outfile, sep="\t")
#         s4vectors_result = Rle(result.Runs, result.Values)

#     print("pyranges result\n", result_pyranges)
#     print("s4vectors result\n", s4vectors_result)

#     assert np.allclose(result_pyranges.runs, s4vectors_result.runs, equal_nan=False)
#     assert np.allclose(result_pyranges.values, s4vectors_result.values, equal_nan=True)
