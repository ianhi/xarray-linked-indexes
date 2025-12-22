"""Example data generators for linked_indices documentation and testing.

These functions generate realistic example data in pandas DataFrame format,
mimicking the output of annotation tools like Praat, TextGrid, etc.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    # DataFrame generators (for documentation)
    "speech_annotations",
    "multi_level_annotations",
    "mixed_event_annotations",
    "generate_audio_signal",
    # DataFrame to xarray converters
    "intervals_from_dataframe",
    "intervals_from_long_dataframe",
    # xarray Dataset generators (for testing)
    "multi_interval_dataset",
    "onset_duration_dataset",
    "trial_based_dataset",
    # NDIndex benchmark dataset generators
    "create_trial_ndindex_dataset",
    "create_diagonal_dataset",
    "create_radial_dataset",
    "create_jittered_dataset",
]


def speech_annotations() -> pd.DataFrame:
    """
    Generate example speech annotation data with onset/duration format.

    Returns a DataFrame with word annotations that have gaps between them,
    simulating real speech data where there are pauses between words.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: onset, duration, word
        - onset: start time of each word (seconds)
        - duration: length of each word (seconds)
        - word: the word label

    Examples
    --------
    >>> from linked_indices.example_data import speech_annotations
    >>> df = speech_annotations()
    >>> df
       onset  duration   word
    0    0.5       1.2  hello
    1    2.1       1.8  world
    2    4.5       2.0    how
    3    7.0       2.5    are
    """
    data = [
        [0.5, 1.2, "hello"],
        [2.1, 1.8, "world"],
        [4.5, 2.0, "how"],
        [7.0, 2.5, "are"],
    ]
    return pd.DataFrame(data, columns=["onset", "duration", "word"])  # type: ignore[arg-type]


def multi_level_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate example multi-level speech annotations (words and phonemes).

    Returns two DataFrames representing hierarchical annotations:
    words contain multiple phonemes, and both have onset/duration format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (word_df, phoneme_df) where:
        - word_df has columns: onset, duration, word, part_of_speech
        - phoneme_df has columns: onset, duration, phoneme

    Examples
    --------
    >>> from linked_indices.example_data import multi_level_annotations
    >>> words, phonemes = multi_level_annotations()
    >>> words
       onset  duration   word part_of_speech
    0    0.0       2.5  hello   interjection
    1    3.0       2.5  world           noun
    2    6.0       3.5   test           noun
    >>> phonemes.head(3)
       onset  duration phoneme
    0    0.0       0.8      hh
    1    0.8       0.9      eh
    2    1.7       0.8       l
    """
    # Word-level annotations
    word_data = [
        [0.0, 2.5, "hello", "interjection"],
        [3.0, 2.5, "world", "noun"],
        [6.0, 3.5, "test", "noun"],
    ]
    word_df = pd.DataFrame(
        word_data,
        columns=["onset", "duration", "word", "part_of_speech"],  # type: ignore[arg-type]
    )

    # Phoneme-level annotations (more fine-grained)
    phoneme_data = [
        [0.0, 0.8, "hh"],
        [0.8, 0.9, "eh"],
        [1.7, 0.8, "l"],
        [3.0, 0.9, "w"],
        [3.9, 0.8, "er"],
        [4.7, 0.8, "ld"],
        [6.0, 1.2, "t"],
        [7.2, 2.3, "st"],
    ]
    phoneme_df = pd.DataFrame(phoneme_data, columns=["onset", "duration", "phoneme"])  # type: ignore[arg-type]

    return word_df, phoneme_df


def mixed_event_annotations() -> pd.DataFrame:
    """
    Generate example annotations with multiple event types in a single DataFrame.

    Returns a "long format" DataFrame where different event types (words, phonemes,
    stimuli) are stacked with an `event_type` column to distinguish them.
    This is a common format when exporting from annotation tools.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: onset, duration, label, event_type
        - onset: start time of each event (seconds)
        - duration: length of each event (seconds)
        - label: the event label (word text, phoneme symbol, or stimulus name)
        - event_type: category of the event ("word", "phoneme", or "stimulus")

    Examples
    --------
    >>> from linked_indices.example_data import mixed_event_annotations
    >>> df = mixed_event_annotations()
    >>> df.groupby("event_type").size()
    event_type
    phoneme     5
    stimulus    2
    word        3
    dtype: int64
    """
    data = [
        # Words
        [0.0, 2.5, "hello", "word"],
        [3.0, 2.5, "world", "word"],
        [6.0, 3.5, "test", "word"],
        # Phonemes (subset for brevity)
        [0.0, 0.8, "hh", "phoneme"],
        [0.8, 0.9, "eh", "phoneme"],
        [1.7, 0.8, "l", "phoneme"],
        [3.0, 0.9, "w", "phoneme"],
        [3.9, 1.6, "orld", "phoneme"],
        # Stimuli (visual/auditory stimuli presented during the experiment)
        [0.0, 5.0, "image_A", "stimulus"],
        [5.0, 5.0, "image_B", "stimulus"],
    ]
    return pd.DataFrame(data, columns=["onset", "duration", "label", "event_type"])  # type: ignore[arg-type]


def generate_audio_signal(
    duration: float = 10.0,
    sample_rate: int = 100,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulated audio signal for examples.

    Parameters
    ----------
    duration : float
        Total duration of the signal in seconds.
    sample_rate : int
        Number of samples per second.
    seed : int | None
        Random seed for reproducibility. None for random.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (times, signal) where:
        - times: array of time points
        - signal: simulated audio waveform

    Examples
    --------
    >>> from linked_indices.example_data import generate_audio_signal
    >>> times, signal = generate_audio_signal(duration=10.0)
    >>> len(times)
    1000
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * sample_rate)
    times = np.linspace(0, duration, n_samples)

    # Create a realistic-ish audio signal with multiple frequency components
    signal = (
        0.5 * np.sin(2 * np.pi * 5 * times)
        + 0.3 * np.sin(2 * np.pi * 12 * times)
        + 0.2 * np.random.randn(n_samples)
    )

    return times, signal


def intervals_from_dataframe(
    df: pd.DataFrame,
    dim_name: str,
    onset_col: str = "onset",
    duration_col: str = "duration",
    label_col: str | None = None,
) -> "xr.Dataset":
    """
    Convert an annotation DataFrame to an xarray Dataset with interval coordinates.

    Creates coordinates suitable for use with `DimensionInterval` index.
    Uses pandas `to_xarray()` internally and renames onset/duration columns
    with dimension-prefixed names (e.g., `word_onset`, `word_duration`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with onset, duration, and label columns.
    dim_name : str
        Name of the dimension (e.g., "word", "phoneme"). This will be used
        as the dimension name and to prefix the onset/duration coordinate names.
    onset_col : str
        Name of the onset column in the DataFrame. Default: "onset"
    duration_col : str
        Name of the duration column in the DataFrame. Default: "duration"
    label_col : str | None
        Name of the label column to use as the dimension coordinate.
        If None, uses `dim_name` as the column name.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates:
        - `{dim_name}`: the label values (dimension coordinate)
        - `{dim_name}_onset`: onset values
        - `{dim_name}_duration`: duration values
        - Any other columns become additional coordinates on the dimension

    Examples
    --------
    >>> from linked_indices.example_data import speech_annotations, intervals_from_dataframe
    >>> df = speech_annotations()
    >>> ds = intervals_from_dataframe(df, "word")
    >>> list(ds.dims)
    ['word']
    >>> sorted(ds.coords)
    ['word', 'word_duration', 'word_onset']
    >>> ds.word.values
    array(['hello', 'world', 'how', 'are'], dtype=object)

    Multiple annotation levels can be merged:

    >>> from linked_indices.example_data import multi_level_annotations
    >>> words, phonemes = multi_level_annotations()
    >>> word_ds = intervals_from_dataframe(words, "word", label_col="word")
    >>> phoneme_ds = intervals_from_dataframe(phonemes, "phoneme", label_col="phoneme")
    >>> import xarray as xr
    >>> ds = xr.merge([word_ds, phoneme_ds])
    """
    if label_col is None:
        label_col = dim_name

    # Build rename mapping: onset -> {dim_name}_onset, duration -> {dim_name}_duration
    rename_map = {
        onset_col: f"{dim_name}_onset",
        duration_col: f"{dim_name}_duration",
    }

    # If label_col is different from dim_name, we need to rename it too
    if label_col != dim_name:
        rename_map[label_col] = dim_name

    # Rename columns in DataFrame before conversion
    df_renamed = df.rename(columns=rename_map)

    # Set the dimension coordinate as index and convert to xarray
    ds = df_renamed.set_index(dim_name).to_xarray()

    # Convert all data variables to coordinates
    coord_names = list(ds.data_vars)
    if coord_names:
        ds = ds.set_coords(coord_names)

    return ds


def intervals_from_long_dataframe(
    df: pd.DataFrame,
    event_type_col: str = "event_type",
    onset_col: str = "onset",
    duration_col: str = "duration",
    label_col: str = "label",
) -> "xr.Dataset":
    """
    Convert a long-format DataFrame with multiple event types to an xarray Dataset.

    This function handles DataFrames where multiple annotation types (e.g., words,
    phonemes, stimuli) are stored in a single table with an event type column.
    Each event type becomes a separate dimension with its own onset/duration coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns for onset, duration, label, and event type.
    event_type_col : str
        Name of the column containing event type categories. Default: "event_type"
    onset_col : str
        Name of the onset column. Default: "onset"
    duration_col : str
        Name of the duration column. Default: "duration"
    label_col : str
        Name of the label column. Default: "label"

    Returns
    -------
    xr.Dataset
        Dataset with one dimension per event type, each with coordinates:
        - `{event_type}`: the label values (dimension coordinate)
        - `{event_type}_onset`: onset values
        - `{event_type}_duration`: duration values

    Examples
    --------
    >>> from linked_indices.example_data import mixed_event_annotations, intervals_from_long_dataframe
    >>> df = mixed_event_annotations()
    >>> ds = intervals_from_long_dataframe(df)
    >>> list(ds.dims)
    ['word', 'phoneme', 'stimulus']
    >>> ds.word.values
    array(['hello', 'world', 'test'], dtype=object)
    >>> ds.word_onset.values
    array([0., 3., 6.])

    Notes
    -----
    This is equivalent to calling `intervals_from_dataframe` for each event type
    and merging the results::

        import xarray as xr
        datasets = []
        for event_type in df["event_type"].unique():
            subset = df[df["event_type"] == event_type]
            ds_subset = intervals_from_dataframe(
                subset, dim_name=event_type, label_col="label"
            )
            datasets.append(ds_subset)
        ds = xr.merge(datasets)
    """
    import xarray as xr

    datasets = []
    for event_type in df[event_type_col].unique():
        # Filter to this event type
        subset = df[df[event_type_col] == event_type].copy()

        # Drop the event_type column before conversion
        subset = subset.drop(columns=[event_type_col])

        # Convert using the single-event-type function
        ds_subset = intervals_from_dataframe(
            subset,
            dim_name=event_type,
            onset_col=onset_col,
            duration_col=duration_col,
            label_col=label_col,
        )
        datasets.append(ds_subset)

    return xr.merge(datasets)


# =============================================================================
# xarray Dataset generators (for testing)
# =============================================================================


def multi_interval_dataset() -> "xr.Dataset":
    """
    Create a dataset with multiple interval dimensions over a single continuous time.

    This is useful for testing DimensionInterval with pd.IntervalIndex format.

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

    Returns
    -------
    xr.Dataset
        Dataset with interval coordinates ready for indexing.
    """
    import xarray as xr

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


def onset_duration_dataset() -> "xr.Dataset":
    """
    Create a dataset using onset/duration format instead of pd.IntervalIndex.

    Uses floats and non-contiguous intervals (with gaps) for testing.

    Structure:
        Dimensions: (time: 1000, word: 3, phoneme: 5)
        Coordinates:
          * time               (time) float64
          * word_onset         (word) float64
          * word_duration      (word) float64
          * word               (word) str  # dimension coord
          * phoneme_onset      (phoneme) float64
          * phoneme_duration   (phoneme) float64
          * phoneme            (phoneme) str  # dimension coord

    The intervals (computed from onset+duration) are:
        word: [0.0, 35.5), [40.0, 75.5), [80.0, 115.5)  (3 words with gaps)
        phoneme: [0.0, 15.5), [20.0, 35.5), [40.0, 55.5), [60.0, 75.5), [80.0, 95.5)

    Note: Non-contiguous intervals have gaps between them:
        - Gap between word 0 and 1: [35.5, 40.0)
        - Gap between word 1 and 2: [75.5, 80.0)

    Returns
    -------
    xr.Dataset
        Dataset with onset/duration coordinates ready for indexing.
    """
    import xarray as xr

    C = 2
    N = 1000
    times = np.linspace(0, 120, N)

    # Word onset/duration - 3 words with float boundaries and gaps
    word_onsets = np.array([0.0, 40.0, 80.0])
    word_durations = np.array([35.5, 35.5, 35.5])  # ends at 35.5, 75.5, 115.5
    word_labels = ["hello", "world", "test"]

    # Phoneme onset/duration - 5 phonemes with float boundaries and gaps
    phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    phoneme_durations = np.array([15.5, 15.5, 15.5, 15.5, 15.5])
    phoneme_labels = ["hh", "eh", "ll", "ow", "ld"]

    data = np.random.rand(C, N)

    ds = xr.Dataset(
        {"data": (("C", "time"), data)},
        coords={
            "time": times,
            "word_onset": ("word", word_onsets),
            "word_duration": ("word", word_durations),
            "word": ("word", word_labels),
            "phoneme_onset": ("phoneme", phoneme_onsets),
            "phoneme_duration": ("phoneme", phoneme_durations),
            "phoneme": ("phoneme", phoneme_labels),
        },
    )

    return ds


def trial_based_dataset(
    n_trials: int = 3,
    trial_length: float = 5.0,
    sample_rate: int = 100,
    trial_labels: list[str] | None = None,
    seed: int | None = 42,
    mode: str = "stacked",
) -> "xr.Dataset":
    """
    Create a dataset with trial-based data and both absolute and relative time.

    This is useful for testing NDIndex.

    Supports two modes:
    - "stacked" (default): 2D array with dimensions (trial, rel_time). Each trial
      has the same relative time coordinates but different absolute time ranges.
    - "linear": 1D array with dimension (abs_time). All trials concatenated into
      a single continuous stream indexed by absolute time, with trial as a
      1D coordinate indicating which trial each timepoint belongs to.

    By default, creates 3 trials with distinct waveforms:
    - Trial 1 ("cosine"): cosine wave
    - Trial 2 ("square"): square wave
    - Trial 3 ("sawtooth"): sawtooth wave

    Parameters
    ----------
    n_trials : int
        Number of trials. Default: 3
    trial_length : float
        Duration of each trial in seconds. Default: 5.0
    sample_rate : int
        Samples per second within each trial. Default: 100
    trial_labels : list[str] | None
        Labels for each trial. If None, uses ["cosine", "square", "sawtooth"]
        for 3 trials, or ["trial_0", "trial_1", ...] for other counts.
    seed : int | None
        Random seed for reproducibility. None for random.
    mode : str
        Either "stacked" (2D with trial × rel_time) or "linear" (1D with abs_time).
        Default: "stacked"

    Returns
    -------
    xr.Dataset
        For mode="stacked":
            Dimensions: (trial: n_trials, rel_time: trial_length * sample_rate)
            Coordinates:
              * trial     (trial) str - trial labels
              * rel_time  (rel_time) float64 - relative time within trial
                abs_time  (trial, rel_time) float64 - absolute time (2D)
                trial_onset (trial) float64 - start time of each trial
            Data variables:
                data      (trial, rel_time) float64 - simulated signal

        For mode="linear":
            Dimensions: (abs_time: n_trials * trial_length * sample_rate)
            Coordinates:
              * abs_time   (abs_time) float64 - absolute time
                rel_time   (abs_time) float64 - relative time within each trial
                trial      (abs_time) str - trial label for each timepoint
                trial_onset (abs_time) float64 - onset time of each trial
            Data variables:
                data       (abs_time) float64 - simulated signal

    Examples
    --------
    >>> from linked_indices.example_data import trial_based_dataset
    >>> ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
    >>> dict(ds.dims)
    {'trial': 3, 'rel_time': 50}
    >>> ds.abs_time.shape
    (3, 50)
    >>> float(ds.abs_time[0, 0])  # First trial starts at t=0
    0.0
    >>> float(ds.abs_time[1, 0])  # Second trial starts at t=5
    5.0

    >>> ds_linear = trial_based_dataset(mode="linear")
    >>> dict(ds_linear.dims)
    {'abs_time': 1500}
    """
    import xarray as xr
    from scipy import signal

    if mode not in ("stacked", "linear"):
        raise ValueError(f"mode must be 'stacked' or 'linear', got '{mode}'")

    if seed is not None:
        np.random.seed(seed)

    # Generate time arrays
    n_samples = int(trial_length * sample_rate)
    rel_times = np.linspace(0, trial_length, n_samples, endpoint=False)

    # Trial labels
    if trial_labels is None:
        if n_trials == 3:
            trial_labels = ["cosine", "square", "sawtooth"]
        else:
            trial_labels = [f"trial_{i}" for i in range(n_trials)]
    elif len(trial_labels) != n_trials:
        raise ValueError(
            f"trial_labels length ({len(trial_labels)}) must match n_trials ({n_trials})"
        )

    # Trial onsets (absolute time when each trial starts)
    trial_onsets = np.arange(n_trials) * trial_length

    # Generate distinct waveforms for each trial
    freq = 0.5  # Base frequency in Hz
    data_2d = np.zeros((n_trials, n_samples))

    for i in range(n_trials):
        waveform_type = i % 3  # Cycle through cosine, square, sawtooth
        if waveform_type == 0:
            # Cosine wave
            data_2d[i] = np.cos(2 * np.pi * freq * rel_times)
        elif waveform_type == 1:
            # Square wave
            data_2d[i] = signal.square(2 * np.pi * freq * rel_times)
        else:
            # Sawtooth wave
            data_2d[i] = signal.sawtooth(2 * np.pi * freq * rel_times)

    if mode == "stacked":
        # 2D mode: (trial, rel_time)
        # Absolute time is a 2D array: abs_time[trial, rel_time_idx] = trial_onset + rel_time
        abs_time_2d = trial_onsets[:, np.newaxis] + rel_times[np.newaxis, :]

        ds = xr.Dataset(
            {"data": (("trial", "rel_time"), data_2d)},
            coords={
                "trial": trial_labels,
                "rel_time": rel_times,
                "abs_time": (("trial", "rel_time"), abs_time_2d),
                "trial_onset": ("trial", trial_onsets),
            },
        )
    else:
        # Linear mode: (abs_time,)
        # Concatenate all trials into a single 1D array
        data_1d = data_2d.flatten()

        # Absolute time is continuous across all trials
        abs_time_1d = np.concatenate(
            [trial_onsets[i] + rel_times for i in range(n_trials)]
        )

        # Relative time repeats for each trial
        rel_time_1d = np.tile(rel_times, n_trials)

        # Trial label for each timepoint
        trial_1d = np.repeat(trial_labels, n_samples)

        # Trial onset for each timepoint
        trial_onset_1d = np.repeat(trial_onsets, n_samples)

        ds = xr.Dataset(
            {"data": (("abs_time",), data_1d)},
            coords={
                "abs_time": abs_time_1d,
                "rel_time": ("abs_time", rel_time_1d),
                "trial": ("abs_time", trial_1d),
                "trial_onset": ("abs_time", trial_onset_1d),
            },
        )

    return ds


# =============================================================================
# NDIndex benchmark dataset generators
# =============================================================================


def create_trial_ndindex_dataset(n_trials: int, n_times: int) -> "xr.Dataset":
    """
    Create trial-based dataset with abs_time = trial_onset + rel_time.

    This is the typical neuroscience use case: multiple trials with
    overlapping relative time but different absolute time ranges.
    Returns a dataset with NDIndex already set on abs_time.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_times : int
        Number of time points per trial.

    Returns
    -------
    xr.Dataset
        Dataset with NDIndex set on abs_time coordinate.

    Examples
    --------
    >>> from linked_indices.example_data import create_trial_ndindex_dataset
    >>> ds = create_trial_ndindex_dataset(10, 100)
    >>> result = ds.sel(abs_time=0.5, method="nearest")  # Select by absolute time
    >>> result.sizes['trial']
    1
    """
    import xarray as xr

    from linked_indices import NDIndex

    trial_onsets = np.arange(n_trials) * n_times * 0.01
    rel_time = np.linspace(0, n_times * 0.01, n_times)
    abs_time = trial_onsets[:, np.newaxis] + rel_time[np.newaxis, :]
    data = np.random.randn(n_trials, n_times)

    ds = xr.Dataset(
        {"data": (["trial", "rel_time"], data)},
        coords={
            "trial": np.arange(n_trials),
            "rel_time": rel_time,
            "abs_time": (["trial", "rel_time"], abs_time),
        },
    )
    return ds.set_xindex(["abs_time"], NDIndex)


def create_diagonal_dataset(ny: int, nx: int) -> "xr.Dataset":
    """
    Create image-like dataset with diagonal gradient coordinate.

    This is from the slicing gallery: derived[y, x] = y_offset[y] + x_coord[x]
    Similar structure to trial data but with different scale/semantics.
    Returns a dataset with NDIndex already set on the derived coordinate.

    Parameters
    ----------
    ny : int
        Number of y (row) coordinates.
    nx : int
        Number of x (column) coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with NDIndex set on derived coordinate.

    Examples
    --------
    >>> from linked_indices.example_data import create_diagonal_dataset
    >>> ds = create_diagonal_dataset(100, 100)
    >>> result = ds.sel(derived=50, method="nearest")
    >>> int(result.derived)
    50
    """
    import xarray as xr

    from linked_indices import NDIndex

    y_coord = np.arange(ny)
    x_coord = np.arange(nx)

    # Diagonal gradient: each row starts 2 units higher
    y_offset = y_coord * 2
    derived_coord = y_offset[:, np.newaxis] + x_coord[np.newaxis, :]
    data = np.random.randn(ny, nx)

    ds = xr.Dataset(
        {"data": (["y", "x"], data)},
        coords={
            "y": y_coord,
            "x": x_coord,
            "derived": (["y", "x"], derived_coord),
        },
    )
    return ds.set_xindex(["derived"], NDIndex)


def create_radial_dataset(ny: int, nx: int) -> "xr.Dataset":
    """
    Create image-like dataset with radial coordinate (non-linear 2D).

    This tests performance with non-monotonic, complex coordinate patterns.
    The radius coordinate is the distance from the center of the array.
    Returns a dataset with NDIndex already set on the radius coordinate.

    Parameters
    ----------
    ny : int
        Number of y (row) coordinates.
    nx : int
        Number of x (column) coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with NDIndex set on radius coordinate.

    Examples
    --------
    >>> from linked_indices.example_data import create_radial_dataset
    >>> ds = create_radial_dataset(100, 100)
    >>> result = ds.sel(radius=slice(10, 20))  # Select an annulus
    >>> result.sizes['y'] > 0 and result.sizes['x'] > 0
    True
    """
    import xarray as xr

    from linked_indices import NDIndex

    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    data = np.random.randn(ny, nx)

    ds = xr.Dataset(
        {"data": (["y", "x"], data)},
        coords={
            "y": np.arange(ny),
            "x": np.arange(nx),
            "radius": (["y", "x"], radius),
        },
    )
    return ds.set_xindex(["radius"], NDIndex)


def create_jittered_dataset(
    n_trials: int, n_times: int, jitter_std: float = 0.1
) -> "xr.Dataset":
    """
    Create trial dataset with per-trial timing jitter.

    More realistic: trial onsets have random variation, and sampling
    times have small per-sample jitter (like real physiological recordings).
    Returns a dataset with NDIndex already set on abs_time.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_times : int
        Number of time points per trial.
    jitter_std : float
        Standard deviation of timing jitter. Default: 0.1

    Returns
    -------
    xr.Dataset
        Dataset with NDIndex set on abs_time coordinate.

    Examples
    --------
    >>> from linked_indices.example_data import create_jittered_dataset
    >>> ds = create_jittered_dataset(10, 100, jitter_std=0.2)
    >>> result = ds.sel(abs_time=0.5, method="nearest")
    >>> result.sizes['trial']
    1
    """
    import xarray as xr

    from linked_indices import NDIndex

    np.random.seed(42)  # Reproducible

    # Trial onsets with jitter
    base_onsets = np.arange(n_trials) * n_times * 0.01
    trial_onsets = base_onsets + np.random.randn(n_trials) * jitter_std
    trial_onsets[0] = 0  # First trial starts at 0

    # Per-sample timing jitter within each trial
    base_rel_time = np.linspace(0, n_times * 0.01, n_times)
    rel_time_jitter = np.random.randn(n_trials, n_times) * (jitter_std * 0.01)

    # 2D absolute time with jitter
    abs_time = (
        trial_onsets[:, np.newaxis] + base_rel_time[np.newaxis, :] + rel_time_jitter
    )
    data = np.random.randn(n_trials, n_times)

    ds = xr.Dataset(
        {"data": (["trial", "rel_time"], data)},
        coords={
            "trial": np.arange(n_trials),
            "rel_time": base_rel_time,
            "abs_time": (["trial", "rel_time"], abs_time),
        },
    )
    return ds.set_xindex(["abs_time"], NDIndex)
