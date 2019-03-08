# pyrle

[![Build Status](https://travis-ci.org/endrebak/pyrle.svg?branch=master)](https://travis-ci.org/endrebak/pyrle) [![hypothesis tested](graphs/hypothesis-tested-brightgreen.svg)](http://hypothesis.readthedocs.io/) [![PyPI version](https://badge.fury.io/py/pyrle.svg)](https://badge.fury.io/py/pyrle)

Run length arithmetic in Python using Cython. Inspired by the Rle class in R's
S4Vectors.

As opposed to S4Vectors, pyrle does not rotate the shortest vector, but rather extends the shorter Rle with zeroes. This is likely the desired behavior in almost all cases.
