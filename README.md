# pyrle

A first stab at ultrafast run length arithmetic in Python using Cython. Inspired
by the Rle class in R's S4Vectors. Not ready for use.

As opposed to S4Vectors, pyrle does not rotate the shortest vector, but rather extends the shorter Rle with zeroes. This is likely the desired behavior in almost all cases.

## Develop

```
python build_code.py; python setup.py build_ext --inplace; py.test -f tests/
```

## TODO

- Write tests that generate random vectors and compares the results from pyrle and S4Vectors.
- Unit-tests
- Test that works in multi-cpu code (pickling)
- Add more operations (multiply, divide)
- Make some repetitive code-patterns inline functions
- Function that makes pandas df into Rle, like S4Vectors coverage

## Example

```
import numpy as np
from pyrle import Rle
import pandas as pd

r = pd.Series([1, 2, 3, 4], dtype=np.int16)
# v = pd.Series([-1, 2.3, 3, 4.976], dtype=np.float)
r1 = Rle(r, r)
r2 = Rle(r * 2, r * 2)

r3 = r1 + r2
print("Runs:", r3.runs)
# Runs: [1 1 1 3 4 2 8]
print("Values:", r3.values)
# Values: [  3.   4.   6.   7.  10.   6.   8.]
# This is the same as we get in R:
# numeric-Rle of length 20 with 7 runs
#   Lengths:  1  1  1  3  4  2  8
#   Values :  3  4  6  7 10  6  8
```

## When not to use Rles

In constructed degenerate cases, the arithmetic operations are O(n^2). This is when your data is wildly heterogenous and Rles aren't applicable.
