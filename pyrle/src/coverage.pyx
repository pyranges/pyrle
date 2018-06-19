import numpy as np
import pandas as pd

cimport cython

from libc.math cimport isnan

# try:
#     dummy = profile
# except:
#     profile = lambda x: x
def insort(a, b, kind='mergesort'):
    # took mergesort as it seemed a tiny bit faster for my sorted large array try.
    c = np.concatenate((a, b)) # we still need to do this unfortunatly.
    c.sort(kind=kind)
    flag = np.ones(len(c), dtype=bool)
    np.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]


@cython.boundscheck(False)
@cython.wraparound(False)
def _coverage(long [::1] positions, double [::1] values):

    d = {}

    cdef int i = 0
    cdef int j = 0
    cdef int pos = -1
    cdef int oldpos = positions[0]
    cdef double value

    inlength = len(positions)

    unique = np.unique(positions)
    n_unique = len(unique)

    outlength = n_unique
    positions_arr = unique
    if 0 == positions[0]:
        first_value = values[0]
    else:
        first_value = 0

    values_arr = np.zeros(outlength)

    cdef long[::1] outposition
    cdef double[::1] outvalue

    outvalue = values_arr
    outposition = positions_arr

    while i < inlength:
        if positions[i] != oldpos:
            j += 1
            oldpos = positions[i]

        outvalue[j] += values[i]
        i += 1

    value_series = pd.Series(values_arr)
    runs = pd.Series(positions_arr)

    value_series = value_series.cumsum().shift()
    value_series[0] = first_value

    shifted = runs.shift()
    shifted[0] = 0
    runs = (runs - shifted)

    if len(value_series) > 1 and first_value == value_series[1]:
        runs[1] += runs[0]
        value_series = value_series[1:]
        runs = runs[1:]

    return runs.values, value_series.values


@cython.boundscheck(False)
@cython.wraparound(False)
def _remove_dupes(long [::1] runs, double [::1] values, int length):

    cdef long[::1] _runs
    cdef double[::1] _vals

    _runs = runs
    _vals = values

    i = 0
    cdef int counter = 0
    cdef double old_val = _vals[i]
    cdef int old_run = _runs[i]
    cdef int run
    cdef float value

    nrs_arr = np.zeros(len(runs), dtype=np.long)
    nvs_arr = np.zeros(len(runs), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    for i in range(1, len(values)):

        run = _runs[i]
        value = _vals[i]

        if isnan(value) and isnan(old_val):
            old_run += run
        elif value == old_val:
            old_run += run
        else:
            nrs[counter] = old_run
            nvs[counter] = old_val
            old_run = run
            old_val = value
            counter += 1
    # print("nrs_arr", nrs_arr)
    # print("nvs_arr", nvs_arr)

    if len(values) == 1:
        # print("len value series one")
        return runs, values

    if np.isclose(value, old_val, equal_nan=True):
        # print("value == old val")
        nrs[counter] = old_run
        nvs[counter] = old_val
        counter += 1

    # print("nrs_arr", nrs_arr)
    # print("nvs_arr", nvs_arr)
    # print("counter", counter)

    return nrs_arr[:counter], nvs_arr[:counter]
