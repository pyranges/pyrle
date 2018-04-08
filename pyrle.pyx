# cython: infer_types=True

import numpy as np

cimport cython

class Rle:

    def __init__(self, runs, values):
        assert len(runs) == len(values)

        self.runs = np.array(runs, dtype=np.int)
        self.values = np.array(values, dtype=np.double)

    def __add__(self, other):

        new_rle = add_rles(self.runs, self.values, other.runs, other.values)
        return new_rle

    def __sub__(self, other):
        new_rle = sub_rles(self.runs, self.values, other.runs, other.values)
        return new_rle


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef double diff = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef double r1 = runs1[x1]
    cdef double r2 = runs2[x2]
    nrs = np.zeros(len(runs1) + len(runs2), dtype=np.int32)
    nvs = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            nv = values1[x1] + values2[x2]
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            nv = values1[x1] + values2[x2]
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            nv = values1[x1] + values2[x2]
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if nv == nvs[xn]:
            nrs[xn] += nr
            xn += 1
        else:
            if xn < len(nvs):
                nrs[xn] = nr
                nvs[xn] = nv
                xn += 1
            # if we have no space left in our old array, double the size
            else:
                nrs.resize(1, len(nrs) * 2)
                nvs.resize(1, len(nvs) * 2)
                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs.resize((len(nvs) + (l2 - x2) + 1))
            nrs.resize((len(nvs) + (l2 - x2) + 1))

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nv = r2
            nrs[xn] = nv
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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs.resize(len(nvs) + (l1 - x1) + 1)
            nrs.resize(len(nvs) + (l1 - x1) + 1)

        if diff > 0:
            nv = r1
            nrs[xn] = nv
            nvs[xn] += values1[x1]
            xn += 1
            x1 += 1

        if values1[x1] == nv:
            nrs[xn - 1] += runs1[x1]
            x1 += 1

        for i in range(x1, l1):
            nrs[xn] = runs1[i]
            nvs[xn] = values1[i]
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs.resize(xn)
    nvs.resize(xn)

    return Rle(nrs, nvs)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sub_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef double diff = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef double r1 = runs1[x1]
    cdef double r2 = runs2[x2]
    nrs = np.zeros(len(runs1) + len(runs2), dtype=np.int32)
    nvs = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            nv = values1[x1] - values2[x2]
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            nv = values1[x1] - values2[x2]
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            nv = values1[x1] - values2[x2]
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new value is the same as the old, merge the runs
        if nv == nvs[xn]:
            nrs[xn] += nr
            xn += 1
        else:
            if xn < len(nvs):
                nrs[xn] = nr
                nvs[xn] = nv
                xn += 1
            # if we have no space left in our old array, double the size
            else:
                nrs.resize(1, len(nrs) * 2)
                nvs.resize(1, len(nvs) * 2)
                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs.resize((len(nvs) + (l2 - x2) + 1))
            nrs.resize((len(nvs) + (l2 - x2) + 1))

        # Have some values left in one rl from the previous comparison
        # which must be added before we move on
        if diff < 0:
            nv = r2
            nrs[xn] = nv
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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs.resize(len(nvs) + (l1 - x1) + 1)
            nrs.resize(len(nvs) + (l1 - x1) + 1)

        if diff > 0:
            nv = r1
            nrs[xn] = nv
            nvs[xn] += values1[x1]
            xn += 1
            x1 += 1

        if values1[x1] == nv:
            nrs[xn - 1] += runs1[x1]
            x1 += 1

        for i in range(x1, l1):
            nrs[xn] = runs1[i]
            nvs[xn] = values1[i]
            xn += 1

    # Must use resize because initial guess for array was likely way too large
    nrs.resize(xn)
    nvs.resize(xn)

    return Rle(nrs, nvs)

