

import numpy as np
import pandas as pd

cimport cython


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


# @profile
@cython.boundscheck(False)
@cython.wraparound(False)
def _coverage(long [::1] starts, long [::1] ends, double [::1] values, int length):

    d = {}

    cdef int i = 0
    cdef int start
    cdef int end
    cdef double value

    # all_pos = insort(starts, ends)

    # # print("values", np.array(values))
    while i < length:
        start = starts[i]
        value = values[i]
        # # print("start", start)
        # # print("value", value)
        if not start in d:
            # # print("If starts!")
            d[start] = value
            # # print(d[start])
        else:
            d[start] = d[start] + value
            # # print("Else starts!")
            # # print(d[start])
        end = ends[i]
        value = values[i]
        i = i + 1
        if not end in d:
            d[end] = -value
        else:
            d[end] = d[end] - value

    if 0 not in d:
        d[0] = 0

    sorted_items = sorted(d.items())
    # # print("sorted_items", sorted_items)
    runs = pd.Series([r[0] for r in sorted_items])
    # # print("runs\n", runs)
    value_series = pd.Series([v[1] for v in sorted_items])
    # # print("value_series\n", value_series)

    first_value = value_series[0]
    value_series = value_series.cumsum().shift()
    value_series[0] = first_value

    shifted = runs.shift()
    shifted[0] = 0
    runs = (runs - shifted)

    # print("runs", runs)
    # print("value_series", value_series)

    if len(value_series) > 1 and first_value == value_series[1]:
        runs[1] += runs[0]
        value_series = value_series[1:]
        runs = runs[1:]


    cdef long[::1] _runs
    cdef double[::1] _vals

    _runs = runs.astype(long).values
    _vals = value_series.values

    i = 0
    cdef int counter = 0
    cdef double old_val = _vals[i]
    cdef int old_run = _runs[i]
    cdef int run

    nrs_arr = np.zeros(len(runs), dtype=np.long)
    nvs_arr = np.zeros(len(runs), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr


    # print("value_series", value_series)
    for i in range(1, len(value_series)):

        run = _runs[i]
        value = _vals[i]
        # print("old_val", old_val)
        # print("old_run", old_run)
        # print("run", run)
        # print("value", value)

        if value == old_val:
            # print("in equal")
            old_run += run
        else:
            # print("new")
            nrs[counter] = old_run
            nvs[counter] = old_val
            old_run = run
            old_val = value
            counter += 1

    if len(value_series) == 1:
        # print("len value series one")
        return runs.values, value_series.values

    if value == old_val:
        # print("value == old val")
        nrs[counter] = old_run
        nvs[counter] = old_val
        counter += 1

    return nrs[:counter], nvs[:counter]
