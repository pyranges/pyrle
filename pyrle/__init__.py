from pyrle.rle import Rle
from pyrle.version import __version__
from pyrle.rledict import RleDict
from pyrle.methods import coverage

import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict

PyRles = RleDict

def from_csv(f, sep="\t"):

    """Read PyRle from CSV.

    >>>
    """

    d = {}
    df = pd.read_csv(f, sep=sep, index_col=None)
    if "Strand" in df:
        keys = "Chromosome Strand".split()
    else:
        keys = "Chromosome"

    for c, cdf in df.groupby(keys):
        d[c] = Rle(cdf.Runs, cdf.Values)

    return PyRles(d)
