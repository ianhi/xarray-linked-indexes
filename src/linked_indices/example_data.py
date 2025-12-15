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
    "speech_annotations",
    "multi_level_annotations",
    "mixed_event_annotations",
    "generate_audio_signal",
    "intervals_from_dataframe",
    "intervals_from_long_dataframe",
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
    return pd.DataFrame(data, columns=["onset", "duration", "word"])


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
    >>> phonemes
       onset  duration phoneme
    0    0.0       0.8      hh
    1    0.8       0.9      eh
    2    1.7       0.8       l
    ...
    """
    # Word-level annotations
    word_data = [
        [0.0, 2.5, "hello", "interjection"],
        [3.0, 2.5, "world", "noun"],
        [6.0, 3.5, "test", "noun"],
    ]
    word_df = pd.DataFrame(
        word_data, columns=["onset", "duration", "word", "part_of_speech"]
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
    phoneme_df = pd.DataFrame(phoneme_data, columns=["onset", "duration", "phoneme"])

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
    return pd.DataFrame(data, columns=["onset", "duration", "label", "event_type"])


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
    >>> ds
    <xarray.Dataset>
    Dimensions:        (word: 4)
    Coordinates:
      * word           (word) object 'hello' 'world' 'how' 'are'
        word_onset     (word) float64 0.5 2.1 4.5 7.0
        word_duration  (word) float64 1.2 1.8 2.0 2.5
    Data variables:
        *empty*

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
    and merging the results:

    >>> import xarray as xr
    >>> datasets = []
    >>> for event_type in df[event_type_col].unique():
    ...     subset = df[df[event_type_col] == event_type]
    ...     ds_subset = intervals_from_dataframe(
    ...         subset, dim_name=event_type, label_col=label_col
    ...     )
    ...     datasets.append(ds_subset)
    >>> ds = xr.merge(datasets)
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
