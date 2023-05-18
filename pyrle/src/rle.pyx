# cython: infer_types=True

import numpy as np

cimport cython

from numpy import nan

from libc.math cimport INFINITY, NAN, copysign


cdef float inf = INFINITY

# s/boundscheck(True/boundscheck(False
# s/boundscheck(False/boundscheck(True

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef add_rles(const long [::1] runs1, const double [::1] values1, const long [::1] runs2, const double [::1] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int nr = 0
    cdef double nv = 0
    cdef double diff = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef long r1 = runs1[x1]
    cdef long r2 = runs2[x2]
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.int_)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        nv = values1[x1] + values2[x2]
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if xn > 0 and nv == nvs[xn - 1]:
            nrs[xn - 1] += nr
        else:
            nrs[xn] = nr
            nvs[xn] = nv
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr


# s/boundscheck(True/boundscheck(False
# s/boundscheck(False/boundscheck(True

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef sub_rles(const long [::1] runs1, const double [::1] values1, const long [::1] runs2, const double [::1] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int nr = 0
    cdef double nv = 0
    cdef double diff = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef long r1 = runs1[x1]
    cdef long r2 = runs2[x2]
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.int_)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        nv = values1[x1] - values2[x2]
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if xn > 0 and nv == nvs[xn - 1]:
            nrs[xn - 1] += nr
        else:
            nrs[xn] = nr
            nvs[xn] = nv
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef div_rles_nonzeroes(const long [::1] runs1, const double [::1] values1, const long [::1] runs2, const double [::1] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int nr = 0
    cdef double nv = 0
    cdef double diff = 0
    cdef double sign = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef long r1 = runs1[x1]
    cdef long r2 = runs2[x2]
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.int_)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        nv = values1[x1] / values2[x2]
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if xn > 0 and nv == nvs[xn - 1]:
            nrs[xn - 1] += nr
        else:
            nrs[xn] = nr
            nvs[xn] = nv
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef div_rles_zeroes(const long [::1] runs1, const double [::1] values1, const long [::1] runs2, const double [::1] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int nr = 0
    cdef double nv = 0
    cdef double diff = 0
    cdef double sign = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef long r1 = runs1[x1]
    cdef long r2 = runs2[x2]
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.int_)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2

        if values2[x2] != 0:
            nv = values1[x1] / values2[x2]
        elif values1[x1] != 0:
            sign = copysign(1, values1[x1]) * copysign(1, values2[x2])
            nv = inf * sign
        else:
            nv = NAN

        if diff < 0:
            nr = r1
            r2 = r2 - r1
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if xn > 0 and nv == nvs[xn - 1]:
            nrs[xn - 1] += nr
        else:
            nrs[xn] = nr
            nvs[xn] = nv
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef mul_rles(const long [::1] runs1, const double [::1] values1, const long [::1] runs2, const double [::1] values2):
    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int nr = 0
    cdef double nv = 0
    cdef double diff = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef long r1 = runs1[x1]
    cdef long r2 = runs2[x2]
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.int_)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        nv = values1[x1] * values2[x2]
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if xn > 0 and nv == nvs[xn - 1]:
            nrs[xn - 1] += nr
        else:
            nrs[xn] = nr
            nvs[xn] = nv
            xn += 1


    return nrs_arr, nvs_arr
