import numpy as np
import pandas as pd

cimport cython
from libc.math cimport isnan


cdef extern from "math.h":
    float INFINITY

try:
    dummy = profile
except:
    profile = lambda x: x

def insort(a, b, kind='mergesort'):
    # took mergesort as it seemed a tiny bit faster for my sorted large array try.
    c = np.concatenate((a, b)) # we still need to do this unfortunatly.
    c.sort(kind=kind)
    flag = np.ones(len(c), dtype=bool)
    np.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
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

    if 0 == positions[0]:
        first_value = values[0]
    else:
        first_value = 0

    values_arr = np.zeros(outlength)

    cdef long[::1] outposition
    cdef double[::1] outvalue

    outvalue = values_arr
    outposition = unique

    while i < inlength:
        if positions[i] != oldpos:
            j += 1
            oldpos = positions[i]

        outvalue[j] += values[i]
        i += 1

    value_series = pd.Series(values_arr)
    runs = pd.Series(unique, dtype=np.int_)

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
@cython.initializedcheck(False)
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
    cdef int last_different = 0

    nrs_arr = np.zeros(len(runs), dtype=np.int_)
    nvs_arr = np.zeros(len(runs), dtype=np.float64)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    #print("indata runs", list(runs))
    #print("indata values", list(values))
    for i in range(1, len(values)):

        run = _runs[i]
        value = _vals[i]
        #print("run, value", run, value)

        if isnan(value) and isnan(old_val):
            old_run += run
            last_insert = 0
        elif (value == INFINITY and old_val == INFINITY) or (value == -INFINITY and old_val == -INFINITY):
            #print("elif inf")
            old_run += run
            last_insert = 0
        elif abs(value - old_val) < 1e-5:
            #print("elif abs")
            old_run += run
            last_insert = 0
        else:
            #print("else inserting", old_run, old_val)
            nrs[counter] = old_run
            nvs[counter] = old_val
            old_run = run
            old_val = value
            counter += 1
            last_insert = 1
    ##print("nrs_arr", nrs_arr)
    ##print("nvs_arr", nvs_arr)
    #print("old_val", old_val)
    #print("nvs[counter]", nvs[counter])
    #print("counter", counter)
    #print("last_insert", last_insert)

    if counter == 0:
        nvs[counter] = old_val
        nrs[counter] = old_run
        counter += 1
    elif not last_insert:
        #print("in last if " * 10)
        nvs[counter] = old_val
        nrs[counter] = old_run
        counter += 1
    else:
        nvs[counter] = value
        nrs[counter] = run
        counter += 1



    if len(values) == 1:
        ##print("len value series one")
        return runs, values

    # if np.isclose(value, old_val, equal_nan=True) and counter > 0:

    #     #print("value == old val and counter > 0")
    #     nrs[counter - 1] += old_run
    # if np.isclose(value, old_val, equal_nan=True):
    #     #print("value == old val")
    #     nrs[counter] = old_run
    #     nvs[counter] = old_val
        # counter += 1




    #print("nrs_arr", nrs_arr)
    #print("nvs_arr", nvs_arr)
    #print("counter", counter)

    return nrs_arr[:counter], nvs_arr[:counter]
