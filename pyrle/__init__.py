import pandas as pd
import pkg_resources

from pyrle.rle import Rle
from pyrle.rledict import RleDict

__version__ = pkg_resources.get_distribution("pyrle").version

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
