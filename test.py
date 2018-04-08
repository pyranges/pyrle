import numpy as np
from pyrle import Rle
import pandas as pd

r = pd.Series([1, 2, 3, 4], dtype=np.int16)
# v = pd.Series([-1, 2.3, 3, 4.976], dtype=np.float)
r1 = Rle(r, r)

r2 = Rle(r * 2, r * 2)

# > r2
# numeric-Rle of length 20 with 4 runs
#   Lengths: 2 4 6 8
#   Values : 2 4 6 8
# > r4
# numeric-Rle of length 20 with 5 runs
#   Lengths:  1  2  3  4 10
#   Values :  1  2  3  4  0
# > r2 + r4
# numeric-Rle of length 20 with 7 runs
#   Lengths:  1  1  1  3  4  2  8
#   Values :  3  4  6  7 10  6  8

r3 = r1 + r2
print(r3.runs)
print(r3.values)

print(r3.runs.dtype)
print(r3.values.dtype)

print(r3.runs.shape)
print(r3.values.shape)

r4 = r2 + r1
print(r4.runs)
print(r4.values)

print(r4.runs.dtype)
print(r4.values.dtype)

print(r4.runs.shape)
print(r4.values.shape)

def resize_test():
    """
    test of workign with a numpy array that needs to be re-sized.
    """
    # create an ndarray and a memview to work with it.
    cdef cnp.ndarray[double, ndim=1, mode="c"] arr
    cdef double[:] memview

    ## allocate the array:
    arr = np.zeros( (1,) )

    ## Assign the memview to it:
    memview = arr

    ## manipulate it
    memview[0] = 3.14

    ## resize the array
    arr.resize((4,), refcheck = False)

    ## re-assign the memview -- so you get the new post-resize pointer
    memview = arr

    ## now use it
    memview[1] = 5.6
    memview[2] = 7.1
    memview[3] = 4.3

    ## return the numpy array
    return arr
