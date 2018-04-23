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


    def __truediv__(self, other):
        if (other.values == 0).any():
            new_rle = div_rles(self.runs, self.values, other.runs, other.values)
        else:
            new_rle = div_rles_nozeroes(self.runs, self.values, other.runs, other.values)

        return new_rle

    def __eq__(self, other):
        runs_equal = np.equal(self.runs, other.runs).all()
        values_equal = np.allclose(self.values, other.values)
        return runs_equal and values_equal

    def __str__(self):

        if len(self.runs) > 10:
            runs = " ".join([str(i) for i in self.runs[:5]]) + \
                " ... " + " ".join(["{0:.3f}".format(i) for i in self.runs[-5:]])
            values = " ".join([str(i) for i in self.values[:5]]) + \
                    " ... " + " ".join(["{0:.3f}".format(i) for i in self.values[-5:]])
        else:
            runs = " ".join([str(i) for i in self.runs])
            values = " ".join(["{0:.3f}".format(i) for i in self.values])

        runs = "Runs: " + runs
        values = "Values: " + values

        outstr = "\n".join([runs, values, "Length: " + str(len(self.runs))])

        return outstr

    def __repr__(self):

        return str(self)





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

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

    cdef long[:] nrs
    cdef double[:] nvs

    nrs = nrs_arr
    nvs = nvs_arr

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
                nrs_arr.resize(1, len(nrs) * 2, refcheck=False)
                nvs_arr.resize(1, len(nvs) * 2, refcheck=False)
                nrs = nrs_arr
                nvs = nvs_arr

                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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

    return Rle(nrs_arr, nvs_arr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sub_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

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

    cdef long[:] nrs
    cdef double[:] nvs

    nrs = nrs_arr
    nvs = nvs_arr

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
                nrs_arr.resize(1, len(nrs) * 2, refcheck=False)
                nvs_arr.resize(1, len(nvs) * 2, refcheck=False)
                nrs = nrs_arr
                nvs = nvs_arr

                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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

    return Rle(nrs_arr, nvs_arr)




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef div_rles_nozeroes(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

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

    cdef long[:] nrs
    cdef double[:] nvs

    nrs = nrs_arr
    nvs = nvs_arr

    while(x1 < l1 and x2 < l2):

        diff = r1 - r2
        if diff < 0:
            nr = r1
            r2 = r2 - r1
            nv = values1[x1] / values2[x2]
            x1 += 1
            if x1 < l1:
                r1 = runs1[x1]
        elif diff > 0:
            nr = r2
            r1 = r1 - r2
            nv = values1[x1] / values2[x2]
            x2 += 1
            if x2 < l2:
                r2 = runs2[x2]
        else:
            nr = r2
            nv = values1[x1] / values2[x2]
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
                nrs_arr.resize(1, len(nrs) * 2, refcheck=False)
                nvs_arr.resize(1, len(nvs) * 2, refcheck=False)
                nrs = nrs_arr
                nvs = nvs_arr

                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr


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

    return Rle(nrs_arr, nvs_arr)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef div_rles(long [:] runs1, double [:] values1, long [:] runs2, double [:] values2):

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

    cdef long[:] nrs
    cdef double[:] nvs

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
                nrs_arr.resize(1, len(nrs) * 2, refcheck=False)
                nvs_arr.resize(1, len(nvs) * 2, refcheck=False)
                nrs = nrs_arr
                nvs = nvs_arr

                nrs[xn] = nr
                nvs[xn] = nv

    # Here we unwind the rest of the values that were not added because one Rle was longer than the other.
    # (If other had largest sum of lengths.)
    if x1 == l1 and not x2 == l2:
        if not (xn + (l2 - x2) + 1 < len(nvs)):
            nvs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs_arr.resize((len(nvs) + (l2 - x2) + 1), refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr

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
        if not (xn + (l1 - x1) + 1 < len(nvs)) and diff > 0:
            nvs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs_arr.resize(len(nvs) + (l1 - x1) + 1, refcheck=False)
            nrs = nrs_arr
            nvs = nvs_arr


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

    return Rle(nrs_arr, nvs_arr)


















