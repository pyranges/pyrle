
from pyrle import Rle

from joblib import Parallel, delayed

from pyrle import methods

from natsort import natsorted


class GRles():

    def __init__(self, ranges, n_jobs=1):

        # Construct GRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges

        # Construct GRles from ranges
        else:

            try:
                df = ranges.df
            except:
                df = ranges

            print("in __init__: ", df)

            chromosomes = df.Chromosome.drop_duplicates()

            if n_jobs > 1:
                _rles = Parallel(n_jobs=n_jobs)(delayed(coverage)(df[df.Chromosome == c]) for c in chromosomes)
            else:
                _rles = [coverage(df[df.Chromosome == c]) for c in chromosomes]

            self.rles = {c: r for c, r in zip(chromosomes, _rle)}


    def add(self, other):


        return methods._add(self, other)


    def __str__(self):

        keys = natsorted(self.rles.keys())

        if len(keys) > 2:
            str_list = [keys[0],
                str(self.rles[keys[0]]),
                "...",
                keys[-1],
                str(self.rles[keys[-1]]),
                "GRles object with {} chromosomes.".format(len(self.rles.keys()))]
        elif len(keys) == 2:
            str_list = [keys[0],
                        "-" * len(keys[0]),
                        str(self.rles[keys[0]]),
                        "",
                        keys[-1],
                        "-" * len(keys[-1]),
                        str(self.rles[keys[-1]]),
                        "GRles object with {} chromosomes.".format(len(self.rles.keys()))]
        else:
            str_list = [keys[0],
                        str(self.rles[keys[0]]),
                        "GRles object with {} chromosome.".format(len(self.rles.keys()))]


        outstr = "\n".join(str_list)

        return outstr


    def __eq__(self, other):

        if not self.rles.keys() == other.rles.keys():
            return False

        for c in self.rles.keys():

            if self.rles[c] != other.rles[c]:
                return False

        return True


    def __repr__(self):

        return str(self)


    # def __add__(self, other):

    #     rle = _add(self, other)
    #     return rle

    # def __sub__(self, other):

    #     rle = _sub(self, other)
    #     return rle
