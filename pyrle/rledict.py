
from pyrle import Rle

from joblib import Parallel, delayed

from pyrle import methods

from natsort import natsorted


class GRles():

    def __init__(self, ranges, n_jobs=1, stranded=False):

        # Construct GRles from dict of rles
        if isinstance(ranges, dict):

            self.rles = ranges

        # Construct GRles from ranges
        elif not stranded:

            try:
                df = ranges.df
            except:
                df = ranges


            chromosomes = df.Chromosome.drop_duplicates()

            if n_jobs > 1:
                _rles = Parallel(n_jobs=n_jobs)(delayed(methods.coverage)(df[df.Chromosome == c]) for c in chromosomes)
            else:
                _rles = []
                for c in chromosomes:
                    cv = methods.coverage(df[df.Chromosome == c])
                    _rles.append(cv)

            self.rles = {c: r for c, r in zip(chromosomes, _rles)}

        else:

            try:
                df = ranges.df
            except:
                df = ranges


            cs = df["Chromosome Strand".split()].drop_duplicates()
            cs = list(zip(cs.Chromosome.tolist(), cs.Strand.tolist()))

            if n_jobs > 1:
                _rles = Parallel(n_jobs=n_jobs)(delayed(methods.coverage)(df[(df.Chromosome == c) & (df.Strand == s)]) for c, s in cs)
            else:
                _rles = []
                for c, s in cs:
                    sub_df = df[(df.Chromosome == c) & (df.Strand == s)]
                    _rles.append(methods.coverage(sub_df))

            self.rles = {c: r for c, r in zip(cs, _rles)}


    def add(self, other):

        return methods._add(self, other)


    @property
    def stranded(self):

        return len(self.keys()[0]) == 2

    def keys(self):

        return natsorted(list(self.rles.keys()))

    def values(self):

        return natsorted(list(self.rles.values()))

    def items(self):

        return natsorted(list(self.rles.items()))


    def __str__(self):

        keys = natsorted(self.rles.keys())
        stranded = True if len(list(keys)[0]) == 2 else False

        if not stranded:
            if len(keys) > 2:
                str_list = [keys[0],
                    str(self.rles[keys[0]]),
                    "...",
                    keys[-1],
                    str(self.rles[keys[-1]]),
                    "Unstranded GRles object with {} chromosomes.".format(len(self.rles.keys()))]
            elif len(keys) == 2:
                str_list = [keys[0],
                            "-" * len(keys[0]),
                            str(self.rles[keys[0]]),
                            "",
                            keys[-1],
                            "-" * len(keys[-1]),
                            str(self.rles[keys[-1]]),
                            "Unstranded GRles object with {} chromosomes.".format(len(self.rles.keys()))]
            else:
                str_list = [keys[0],
                            str(self.rles[keys[0]]),
                            "Unstranded GRles object with {} chromosome.".format(len(self.rles.keys()))]

        else:
            if len(keys) > 2:
                str_list = [" ".join(keys[0]),
                    str(self.rles[keys[0]]),
                    "...",
                    " ".join(keys[-1]),
                    str(self.rles[keys[-1]]),
                    "GRles object with {} chromosomes/strand pairs.".format(len(self.rles.keys()))]
            elif len(keys) == 2:
                str_list = [" ".join(keys[0]),
                            "-" * len(keys[0]),
                            str(self.rles[keys[0]]),
                            "",
                            " ".join(keys[-1]),
                            "-" * len(keys[-1]),
                            str(self.rles[keys[-1]]),
                            "GRles object with {} chromosomes/strand pairs.".format(len(self.rles.keys()))]
            else:
                str_list = [" ".join(keys[0]),
                            str(self.rles[keys[0]]),
                            "GRles object with {} chromosome/strand pairs.".format(len(self.rles.keys()))]

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
