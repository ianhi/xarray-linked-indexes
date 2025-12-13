import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


__all__ = ["interval_dataset", "multi_interval_dataset"]


def interval_dataset(interval_dim="word") -> xr.Dataset:
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
            "intervals": (interval_dim, word_intervals),
            "word": (interval_dim, ["green", "red", "blue", "red"]),
        },
    )

    return ds


def multi_interval_dataset() -> xr.Dataset:
    """
    Create a dataset with multiple interval dimensions over a single continuous time.

    Structure:
        Dimensions: (time: 1000, word: 3, phoneme: 6)
        Coordinates:
          * time               (time) float64
          * word_intervals     (word) interval[float64]
          * word               (word) str  # dimension coord
          * part_of_speech     (word) str  # second label for word dimension
          * phoneme_intervals  (phoneme) interval[float64]
          * phoneme            (phoneme) str  # dimension coord

    The intervals are:
        word: [0-40), [40-80), [80-120)  (3 words)
        phoneme: [0-20), [20-40), [40-60), [60-80), [80-100), [100-120)  (6 phonemes)
    """
    C = 2
    N = 1000
    times = np.linspace(0, 120, N)

    # Word intervals - 3 intervals of 40 each
    # Each word has both a label and a part of speech
    word_breaks = [0.0, 40.0, 80.0, 120.0]
    word_intervals = pd.IntervalIndex.from_breaks(word_breaks, closed="left")
    word_labels = ["red", "green", "blue"]
    word_pos = ["adjective", "adjective", "noun"]  # part of speech labels

    # Phoneme intervals - 6 intervals of 20 each
    phoneme_breaks = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    phoneme_intervals = pd.IntervalIndex.from_breaks(phoneme_breaks, closed="left")
    phoneme_labels = ["ah", "ee", "oh", "oo", "eh", "ih"]

    data = np.random.rand(C, N)

    ds = xr.Dataset(
        {"data": (("C", "time"), data)},
        coords={
            "time": times,
            "word_intervals": ("word", word_intervals),
            "word": ("word", word_labels),
            "part_of_speech": ("word", word_pos),
            "phoneme_intervals": ("phoneme", phoneme_intervals),
            "phoneme": ("phoneme", phoneme_labels),
        },
    )

    return ds
