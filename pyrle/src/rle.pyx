# cython: infer_types=True

import numpy as np

cimport cython




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add_rles(long [::1] runs1, double [::1] values1, long [::1] runs2, double [::1] values2):

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
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.long)
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

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nrs[xn] = r2
            nvs[xn] += values2[x2]
            xn += 1
            x2 += 1
        # if the new value is same as the old, merge
        if 0 + values2[x2] == nv:
            nrs[xn - 1] += runs2[x2]
            x2 += 1
        # now the unwinding; add all the values missing from one Rle
        for i in range(x2, l2):
            nrs[xn] = runs2[i]
            nvs[xn] += values2[i]
            xn += 1
    # (If self had largest sum of lengths)
    elif x2 == l2 and not x1 == l1:

        if diff > 0:
            nrs[xn] = r1
            nvs[xn] += values1[x1]
            xn += 1
            x1 += 1

        if values1[x1] == nvs[xn - 1]:
            nrs[xn - 1] += runs1[x1]
            x1 += 1

        for i in range(x1, l1):
            nrs[xn] = runs1[i]
            nvs[xn] = values1[i]
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sub_rles(long [::1] runs1, double [::1] values1, long [::1] runs2, double [::1] values2):

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
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.long)
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

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nrs[xn] = r2
            nvs[xn] -= values2[x2]
            xn += 1
            x2 += 1
        # if the new value is same as the old, merge
        if 0 - values2[x2] == nv:
            nrs[xn - 1] += runs2[x2]
            x2 += 1
        # now the unwinding; add all the values missing from one Rle
        for i in range(x2, l2):
            nrs[xn] = runs2[i]
            nvs[xn] -= values2[i]
            xn += 1
    # (If self had largest sum of lengths)
    elif x2 == l2 and not x1 == l1:

        if diff > 0:
            nrs[xn] = r1
            nvs[xn] += values1[x1]
            xn += 1
            x1 += 1

        if values1[x1] == nvs[xn - 1]:
            nrs[xn - 1] += runs1[x1]
            x1 += 1

        for i in range(x1, l1):
            nrs[xn] = runs1[i]
            nvs[xn] = values1[i]
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef div_rles_nonzeroes(long [::1] runs1, double [::1] values1, long [::1] runs2, double [::1] values2):

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
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.long)
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

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nrs[xn] = r2
            nvs[xn] = 0 if values2[x2] else np.nan
            xn += 1
            x2 += 1

        for i in range(x2, l2):
            nv = 0 if values2[i] else np.nan

            if nv == nvs[xn -1]:
                nrs[xn - 1] += runs2[i]
            else:
                nrs[xn] = runs2[i]
                nvs[xn] = nv
                xn += 1
    # (If self had largest sum of lengths)
    elif x2 == l2 and not x1 == l1:

        if diff > 0:
            nrs[xn] = r1
            nvs[xn] = np.inf * np.sign(values1[x1])
            xn += 1
            x1 += 1

        for i in range(x1, l1):
            nv = np.inf * np.sign(values1[i]) if values1[i] else np.nan

            if nv == nvs[xn -1]:
                nrs[xn - 1] += runs1[i]
            else:
                nrs[xn] = runs1[i]
                nvs[xn] = nv
                xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef div_rles_zeroes(long [::1] runs1, double [::1] values1, long [::1] runs2, double [::1] values2):

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
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.long)
    nvs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    cdef long[::1] nrs
    cdef double[::1] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        
        if values2[x2] != 0:
             nv = values1[x1] / values2[x2]
        else:
             nv = np.inf * np.sign(values1[x1])

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

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nrs[xn] = r2
            nvs[xn] = 0 if values2[x2] else np.nan
            xn += 1
            x2 += 1

        for i in range(x2, l2):
            nv = 0 if values2[i] else np.nan

            if nv == nvs[xn -1]:
                nrs[xn - 1] += runs2[i]
            else:
                nrs[xn] = runs2[i]
                nvs[xn] = nv
                xn += 1
    # (If self had largest sum of lengths)
    elif x2 == l2 and not x1 == l1:

        if diff > 0:
            nrs[xn] = r1
            nvs[xn] = np.inf * np.sign(values1[x1])
            xn += 1
            x1 += 1

        for i in range(x1, l1):
            nv = np.inf * np.sign(values1[i]) if values1[i] else np.nan

            if nv == nvs[xn -1]:
                nrs[xn - 1] += runs1[i]
            else:
                nrs[xn] = runs1[i]
                nvs[xn] = nv
                xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mul_rles(long [::1] runs1, double [::1] values1, long [::1] runs2, double [::1] values2):

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
    nrs_arr = np.zeros(len(runs1) + len(runs2), dtype=np.long)
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

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nrs[xn] = r2
            nvs[xn] = 0
            x2 += 1


        xn += 1
        for i in range(x2, l2):
            nrs[xn - 1] += runs2[i]

    # (If self had largest sum of lengths)
    elif x2 == l2 and not x1 == l1:

        if diff > 0:
            nrs[xn] = r1
            nvs[xn] = 0
            x1 += 1

        xn += 1
        for i in range(x1, l1):
            nrs[xn - 1] += runs1[i]

        # Must use resize because initial guess for array was likely way too large
    nrs_arr.resize(xn, refcheck=False)
    nvs_arr.resize(xn, refcheck=False)

    return nrs_arr, nvs_arr