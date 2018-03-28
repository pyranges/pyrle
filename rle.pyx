# cython: infer_types=True
import numpy as np
cimport cython

ctypedef fused my_type:
    int
    double
    long


class Rle:

    def __init__(self, runs, values):
        assert len(runs) == len(values)

        self.runs = np.array(runs, dtype=np.int)
        self.values = np.array(values, dtype=np.double)

    def __add__(self, other):

        new_rle = add_rles(self.runs, self.values, other.runs, other.values)
        return new_rle


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

    cdef int x1 = 0
    cdef int x2 = 0
    cdef int xn = 0
    cdef int l1 = len(runs1)
    cdef int l2 = len(runs2)
    cdef float r1 = runs1[x1]
    cdef float r2 = runs2[x2]
    nrs = np.zeros(len(runs1) + len(runs2), dtype=np.int32)
    nvs = np.zeros(len(runs1) + len(runs2), dtype=np.double)

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        # print("diff is", diff)
        if diff < 0:
            # print("r1 < r2")
            nr = r1
            r2 = r2 - r1
            nv = values1[x1] + values2[x2]
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            # print("r1 > r2")
            nr = r2
            r1 = r1 - r2
            nv = values1[x1] + values2[x2]
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            # print("==")
            nr = r2
            # print("Now nr is", nr)
            nv = values1[x1] + values2[x2]
            x1 += 1
            x2 += 1
            if x1 < l1:
                r1 = runs1[x1]
            if x2 < l2:
                r2 = runs2[x2]

        # if the new values is the same as the old, merge the runs
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

        # print("--round over---")
        # print(nrs)
        # print(nvs)

        if x1 == l1 and not x2 == l2:
            # print("x1 finished!")
            # assert that the array is long enough for the rest
            # print("xn + (l2 - x2) + 1", xn + (l2 - x2) + 1)
            # print("len(nvs)", len(nvs))
            if not (xn + (l2 - x2) + 1 < len(nvs)):
                nvs.resize((len(nvs) + (l2 - x2) + 1))
                nrs.resize((len(nvs) + (l2 - x2) + 1))

            if diff < 0:
                # print(r2)
                # print(xn)
                nrs[xn] = r2
                nvs[xn] = values2[x2]
                xn += 1
                x2 += 1
                # print("After diff 0")
                # print(nrs)
                # print(nvs)

            for i in range(x2, l2):
                nrs[xn] = runs2[i]
                nvs[xn] = values2[i]
                xn += 1

        if x2 == l2 and not x1 == l1:
            # print("x2 finished!")
            # assert that the array is long enough for the rest
            if not (xn + (l1 - x1) + 1 < len(nvs)):
                nvs.resize(len(nvs) + (l1 - x1) + 1)
                nrs.resize(len(nvs) + (l1 - x1) + 1)

            if diff < 0:
                nrs[xn] = r1
                nvs[xn] = values1[x1]
                xn += 1
                x1 += 1

            for i in range(x1, l1):
                nrs[xn] = runs1[i]
                nvs[xn] = values1[i]
                xn += 1

    nrs.resize(xn)
    nvs.resize(xn)

    return Rle(nrs, nvs)
