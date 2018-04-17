from src.pyrle import Rle

import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict


def coverage(ranges, value_col=None):

    try:
        df = ranges.df
    except:
        df = ranges

    if value_col:
        starts = df[["Start"] + [value_col]]
        ends = df[["End"] + [value_col]]
        # spurious warning
        pd.options.mode.chained_assignment = None
        ends.loc[:, value_col] = ends.loc[:, value_col] * - 1
        pd.options.mode.chained_assignment = "warn"
        columns = ["Position"] + [value_col]
    else:
        starts = pd.concat([df.Start, pd.Series(np.ones(len(df)))], axis=1)
        ends = pd.concat([df.End, -1 * pd.Series(np.ones(len(df)))], axis=1)
        columns = "Position Value".split()
        value_col = "Value"

    starts.columns, ends.columns = columns, columns
    runs = pd.concat([starts, ends], ignore_index=True).sort_values("Position")
    values = runs.groupby("Position").sum().reset_index()[value_col]
    runs = runs.drop_duplicates("Position")
    first_value = values.iloc[0] if starts.Position.min() == 0 else 0
    run_lengths = (runs.Position - runs.Position.shift().fillna(0))

    values = values.cumsum().shift()
    values[0] = first_value

    # print(run_lengths)
    # print(len(run_lengths), len(values))

    return Rle(run_lengths, values)
