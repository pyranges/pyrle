
import numpy as np
import pandas as pd

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef getitem(const long [::1] runs, const double [::1] values, start, end):

    cdef:
        int i = 0
        int arr_length = 100
        int nfound = 0
        # int foundsum = 0
        int rsum = 0
        int r = 0
        int l = 0
        int started = 0
        cdef double[::1] vs
        cdef long[::1] rs

    values_arr = np.zeros(arr_length)
    vs = values_arr
    runs_arr = np.zeros(arr_length, dtype=np.long)
    rs = runs_arr

    for i in range(len(runs)):
        # print("i", i)
        r = runs[i]

        # print("r", r)
        rsum += r
        # print("rsum", rsum)

        if started == 0:
            # print("not started")
            if rsum > start:
                # print("rsum > start")

                if not rsum > end:
                    l = rsum - start
                    # this is always the first entry, no need to check size
                    # print("l1", l)
                    rs[nfound] = l
                    # foundsum += l
                    # print("v1", values[i])
                    vs[nfound] = values[i]
                    nfound += 1
                else:
                    return [end - start], [values[i]]

                started = 1
        else:

            if nfound >= arr_length:
                arr_length = arr_length * 2
                values_arr = np.resize(values_arr, arr_length)
                runs_arr = np.resize(runs_arr, arr_length)
                rs = runs_arr
                vs = values_arr

            if rsum < end:

                l = runs[i]
                # print("l2", l)
                # print("v2", values[i])
                rs[nfound] = l
                vs[nfound] = values[i]
                nfound += 1
            else:
                l = runs[i] - (rsum - end)
                # print("l3", l)
                rs[nfound] = l
                vs[nfound] = values[i]
                # print("v3", values[i])
                nfound += 1

                break

    return runs_arr[:nfound], values_arr[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def getlocs(const long [::1] runs, const double [::1] values, const long [::1] locs):

    cdef:
        int i = 0
        int j = 0
        int cumsum = 0
        cdef double[::1] vs
        int loc_len = len(locs)

    values_arr = np.zeros(loc_len)
    vs = values_arr

    for i in range(len(runs)):
        cumsum += runs[i]
        while locs[j] < cumsum:
            vs[j] = values[i]
            j += 1
            if j == loc_len:
                return values_arr

    return values_arr



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef _getitem(const long [::1] runs, const double [::1] values, const long [::1] run_cumsum, start, end):

    cdef:
        int i = 0
        int arr_length = 100
        int nfound = 0
        # int foundsum = 0
        int rsum = 0
        int r = 0
        int l = 0
        long search_start = np.searchsorted(run_cumsum, start)
        int started = 0
        cdef double[::1] vs
        cdef long[::1] rs

    values_arr = np.ones(arr_length) * -1
    vs = values_arr
    runs_arr = np.ones(arr_length, dtype=np.long) * -1
    rs = runs_arr


    for i in range(search_start, len(runs)):
        # print("i", i)
        r = runs[i]

        # print("r", r)
        rsum = run_cumsum[i]
        # print("rsum", rsum)

        if started == 0:
            # print("not started")
            if rsum > start:
                # print("rsum > start")

                if not rsum > end:
                    l = rsum - start
                    # this is always the first entry, no need to check size
                    # print("l1", l)
                    rs[nfound] = l
                    # foundsum += l
                    # print("v1", values[i])
                    vs[nfound] = values[i]
                    nfound += 1
                else:
                    return [end - start], [values[i]]

                started = 1
        else:

            if nfound >= arr_length:
                arr_length = arr_length * 2
                values_arr = np.resize(values_arr, arr_length)
                runs_arr = np.resize(runs_arr, arr_length)
                rs = runs_arr
                vs = values_arr

            if rsum < end:

                l = runs[i]
                # print("l2", l)
                # print("v2", values[i])
                rs[nfound] = l
                vs[nfound] = values[i]
                nfound += 1
            else:
                l = runs[i] - (rsum - end)
                # print("l3", l)
                if l == 0:
                    break

                rs[nfound] = l
                vs[nfound] = values[i]
                # print("v3", values[i])
                nfound += 1

                break

    return runs_arr[:nfound], values_arr[:nfound]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef getitems(const long [::1] runs, const double [::1] values, const long [::1] starts, const long [::1] ends):

    cdef:
        long i = 0
        long counter = 0
        long rsum = 0
        long start = 0
        long end = 0
        long old_start = -1
        long old_end = -1
        cdef long[::1] run_cumsum

    run_cumsum_arr = np.cumsum(runs)
    run_cumsum = run_cumsum_arr

    _runs, _values = [], []

    for i in range(len(starts)):

        __runs, __values = _getitem(runs, values, run_cumsum, starts[i], ends[i])

        _runs.append(__runs)
        _values.append(__values)

    ls = np.array([len(r) for r in _runs], dtype=int)
    _runs = np.concatenate(_runs)
    _values = np.concatenate(_values)

    _starts = np.repeat(starts, ls)
    _ends = np.repeat(ends, ls)

    df = pd.DataFrame({"Start": _starts, "End": _ends, "Run": _runs, "Value": _values})

    return df
