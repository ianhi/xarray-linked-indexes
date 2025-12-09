import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


__all__ = ["interval_dataset"]


def interval_dataset() -> xr.Dataset:
    T = np.arange(0, 5000, 5)
    N = T.shape[0]
    breaks = [0, N * 2 / 10, N * 5 / 10, N * 7 / 10, N * 10 / 10 - 1]
    breaks = [T[int(b)] for b in list(breaks)]
    word_intervals = pd.IntervalIndex.from_breaks(breaks, closed="left")
    data = np.random.rand(10, N, 3)
    data[:] = 0

    # debugging colormap for easy viz
    data[:] = plt.cm.prism(np.arange(N) / N)[:, :3]

    ds = xr.Dataset(
        {
            "data": (
                ("C", "time", "rgb"),
                data,
            )
        },
        coords={
            "time": T,
            # must add enforcement that word and intervals are the same length
            "intervals": ("word", word_intervals),
            "word": ("word", ["green", "red", "blue", "red"]),
        },
    )

    return ds
