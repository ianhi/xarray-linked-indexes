"""Tests for the example_data module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linked_indices import DimensionInterval
from linked_indices.example_data import (
    generate_audio_signal,
    intervals_from_dataframe,
    intervals_from_long_dataframe,
    mixed_event_annotations,
    multi_level_annotations,
    speech_annotations,
)


class TestSpeechAnnotations:
    """Tests for speech_annotations function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = speech_annotations()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have onset, duration, and word columns."""
        df = speech_annotations()
        assert "onset" in df.columns
        assert "duration" in df.columns
        assert "word" in df.columns

    def test_has_gaps_between_words(self):
        """Intervals should have gaps (non-contiguous)."""
        df = speech_annotations()
        # Check that end of one word != start of next
        ends = df["onset"] + df["duration"]
        starts = df["onset"].iloc[1:]
        # At least one gap should exist
        assert any(ends.iloc[:-1].values < starts.values)


class TestMultiLevelAnnotations:
    """Tests for multi_level_annotations function."""

    def test_returns_tuple_of_dataframes(self):
        """Should return two DataFrames."""
        result = multi_level_annotations()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_word_df_has_required_columns(self):
        """Word DataFrame should have expected columns."""
        word_df, _ = multi_level_annotations()
        assert "onset" in word_df.columns
        assert "duration" in word_df.columns
        assert "word" in word_df.columns
        assert "part_of_speech" in word_df.columns

    def test_phoneme_df_has_required_columns(self):
        """Phoneme DataFrame should have expected columns."""
        _, phoneme_df = multi_level_annotations()
        assert "onset" in phoneme_df.columns
        assert "duration" in phoneme_df.columns
        assert "phoneme" in phoneme_df.columns


class TestMixedEventAnnotations:
    """Tests for mixed_event_annotations function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = mixed_event_annotations()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have onset, duration, label, and event_type columns."""
        df = mixed_event_annotations()
        assert "onset" in df.columns
        assert "duration" in df.columns
        assert "label" in df.columns
        assert "event_type" in df.columns

    def test_has_multiple_event_types(self):
        """Should contain multiple event types."""
        df = mixed_event_annotations()
        event_types = df["event_type"].unique()
        assert len(event_types) >= 2
        assert "word" in event_types
        assert "phoneme" in event_types
        assert "stimulus" in event_types

    def test_each_event_type_has_entries(self):
        """Each event type should have at least one entry."""
        df = mixed_event_annotations()
        for event_type in df["event_type"].unique():
            count = (df["event_type"] == event_type).sum()
            assert count > 0, f"Event type {event_type} has no entries"


class TestGenerateAudioSignal:
    """Tests for generate_audio_signal function."""

    def test_returns_tuple_of_arrays(self):
        """Should return tuple of (times, signal) arrays."""
        times, signal = generate_audio_signal()
        assert isinstance(times, np.ndarray)
        assert isinstance(signal, np.ndarray)

    def test_arrays_same_length(self):
        """Times and signal arrays should have same length."""
        times, signal = generate_audio_signal()
        assert len(times) == len(signal)

    def test_duration_parameter(self):
        """Duration parameter should control signal length."""
        times, _ = generate_audio_signal(duration=5.0, sample_rate=100)
        assert len(times) == 500  # 5.0 * 100

    def test_sample_rate_parameter(self):
        """Sample rate should control samples per second."""
        times, _ = generate_audio_signal(duration=10.0, sample_rate=50)
        assert len(times) == 500  # 10.0 * 50

    def test_reproducibility_with_seed(self):
        """Same seed should produce same signal."""
        _, signal1 = generate_audio_signal(seed=42)
        _, signal2 = generate_audio_signal(seed=42)
        np.testing.assert_array_equal(signal1, signal2)

    def test_different_seeds_produce_different_signals(self):
        """Different seeds should produce different signals."""
        _, signal1 = generate_audio_signal(seed=42)
        _, signal2 = generate_audio_signal(seed=123)
        assert not np.array_equal(signal1, signal2)


class TestIntervalsFromDataframe:
    """Tests for intervals_from_dataframe function."""

    def test_returns_dataset(self):
        """Should return an xarray Dataset."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert isinstance(ds, xr.Dataset)

    def test_creates_dimension(self):
        """Should create a dimension with the specified name."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert "word" in ds.dims

    def test_creates_onset_coordinate(self):
        """Should create {dim}_onset coordinate."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert "word_onset" in ds.coords
        np.testing.assert_array_equal(ds.word_onset.values, df["onset"].values)

    def test_creates_duration_coordinate(self):
        """Should create {dim}_duration coordinate."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert "word_duration" in ds.coords
        np.testing.assert_array_equal(ds.word_duration.values, df["duration"].values)

    def test_label_as_dimension_coord(self):
        """Label column should become dimension coordinate."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert "word" in ds.coords
        np.testing.assert_array_equal(ds.word.values, df["word"].values)

    def test_custom_column_names(self):
        """Should work with custom column names."""
        df = pd.DataFrame(
            {
                "start": [0.0, 1.0, 2.0],
                "length": [0.5, 0.5, 0.5],
                "name": ["a", "b", "c"],
            }
        )
        ds = intervals_from_dataframe(
            df,
            "item",
            onset_col="start",
            duration_col="length",
            label_col="name",
        )
        assert "item" in ds.dims
        assert "item_onset" in ds.coords
        assert "item_duration" in ds.coords
        np.testing.assert_array_equal(ds.item.values, ["a", "b", "c"])

    def test_extra_columns_become_coords(self):
        """Extra columns should become additional coordinates."""
        word_df, _ = multi_level_annotations()
        ds = intervals_from_dataframe(word_df, "word", label_col="word")
        assert "part_of_speech" in ds.coords

    def test_no_data_variables(self):
        """Result should have no data variables (all coords)."""
        df = speech_annotations()
        ds = intervals_from_dataframe(df, "word")
        assert len(ds.data_vars) == 0


class TestIntervalsFromLongDataframe:
    """Tests for intervals_from_long_dataframe function."""

    def test_returns_dataset(self):
        """Should return an xarray Dataset."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)
        assert isinstance(ds, xr.Dataset)

    def test_creates_dimension_per_event_type(self):
        """Should create one dimension per event type."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)
        assert "word" in ds.dims
        assert "phoneme" in ds.dims
        assert "stimulus" in ds.dims

    def test_correct_dimension_sizes(self):
        """Each dimension should have correct size."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)
        # Count events per type in original DataFrame
        word_count = (df["event_type"] == "word").sum()
        phoneme_count = (df["event_type"] == "phoneme").sum()
        stimulus_count = (df["event_type"] == "stimulus").sum()
        assert ds.sizes["word"] == word_count
        assert ds.sizes["phoneme"] == phoneme_count
        assert ds.sizes["stimulus"] == stimulus_count

    def test_creates_onset_coords_per_event_type(self):
        """Should create {event_type}_onset coords."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)
        assert "word_onset" in ds.coords
        assert "phoneme_onset" in ds.coords
        assert "stimulus_onset" in ds.coords

    def test_creates_duration_coords_per_event_type(self):
        """Should create {event_type}_duration coords."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)
        assert "word_duration" in ds.coords
        assert "phoneme_duration" in ds.coords
        assert "stimulus_duration" in ds.coords

    def test_label_values_correct(self):
        """Label values should match original DataFrame."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)

        word_labels = df[df["event_type"] == "word"]["label"].values
        np.testing.assert_array_equal(ds.word.values, word_labels)

        phoneme_labels = df[df["event_type"] == "phoneme"]["label"].values
        np.testing.assert_array_equal(ds.phoneme.values, phoneme_labels)

    def test_onset_values_correct(self):
        """Onset values should match original DataFrame."""
        df = mixed_event_annotations()
        ds = intervals_from_long_dataframe(df)

        word_onsets = df[df["event_type"] == "word"]["onset"].values
        np.testing.assert_array_equal(ds.word_onset.values, word_onsets)

    def test_custom_column_names(self):
        """Should work with custom column names."""
        df = pd.DataFrame(
            {
                "start": [0.0, 1.0, 0.0],
                "length": [0.5, 0.5, 1.0],
                "name": ["a", "b", "x"],
                "category": ["type1", "type1", "type2"],
            }
        )
        ds = intervals_from_long_dataframe(
            df,
            event_type_col="category",
            onset_col="start",
            duration_col="length",
            label_col="name",
        )
        assert "type1" in ds.dims
        assert "type2" in ds.dims
        assert "type1_onset" in ds.coords
        assert "type2_duration" in ds.coords

    def test_empty_event_type_handling(self):
        """Should handle DataFrame with only one event type."""
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "duration": [0.5, 0.5],
                "label": ["a", "b"],
                "event_type": ["word", "word"],
            }
        )
        ds = intervals_from_long_dataframe(df)
        assert "word" in ds.dims
        assert ds.sizes["word"] == 2


class TestIntervalsFromDataframeWithDimensionInterval:
    """Tests for using intervals_from_dataframe output with DimensionInterval."""

    @pytest.fixture
    def ds_with_index(self):
        """Create a Dataset with DimensionInterval applied."""
        df = speech_annotations()
        times, audio = generate_audio_signal(duration=10.0)

        # Convert DataFrame to interval coords
        interval_ds = intervals_from_dataframe(df, "word")

        # Create full dataset
        ds = xr.Dataset(
            {"audio": (("time",), audio)},
            coords={"time": times},
        )
        ds = xr.merge([ds, interval_ds])

        # Apply DimensionInterval
        ds = ds.drop_indexes(["time", "word"]).set_xindex(
            ["time", "word_onset", "word_duration", "word"],
            DimensionInterval,
            onset_duration_coords={"word": ("word_onset", "word_duration")},
        )
        return ds

    def test_sel_by_word_works(self, ds_with_index):
        """Should be able to select by word label."""
        result = ds_with_index.sel(word="hello")
        assert result.sizes["word"] == 1
        assert result.word.values == ["hello"]

    def test_sel_by_time_constrains_words(self, ds_with_index):
        """Selecting time should constrain word dimension."""
        # Time 3.0 overlaps with "world" (onset=2.1, duration=1.8, ends at 3.9)
        result = ds_with_index.sel(time=slice(3.0, 4.0))
        # Should only include words that overlap this time range
        assert "world" in result.word.values


class TestIntervalsFromLongDataframeWithDimensionInterval:
    """Tests for using intervals_from_long_dataframe output with DimensionInterval."""

    @pytest.fixture
    def ds_with_index(self):
        """Create a Dataset with multiple event types and DimensionInterval."""
        df = mixed_event_annotations()
        times, audio = generate_audio_signal(duration=10.0)

        # Convert long DataFrame to interval coords
        interval_ds = intervals_from_long_dataframe(df)

        # Create full dataset
        ds = xr.Dataset(
            {"audio": (("time",), audio)},
            coords={"time": times},
        )
        ds = xr.merge([ds, interval_ds])

        # Apply DimensionInterval with all event types
        ds = ds.drop_indexes(["time", "word", "phoneme", "stimulus"]).set_xindex(
            [
                "time",
                "word_onset",
                "word_duration",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
                "stimulus_onset",
                "stimulus_duration",
                "stimulus",
            ],
            DimensionInterval,
            onset_duration_coords={
                "word": ("word_onset", "word_duration"),
                "phoneme": ("phoneme_onset", "phoneme_duration"),
                "stimulus": ("stimulus_onset", "stimulus_duration"),
            },
        )
        return ds

    def test_all_dimensions_present(self, ds_with_index):
        """Should have all event type dimensions."""
        assert "word" in ds_with_index.dims
        assert "phoneme" in ds_with_index.dims
        assert "stimulus" in ds_with_index.dims
        assert "time" in ds_with_index.dims

    def test_sel_word_constrains_others(self, ds_with_index):
        """Selecting a word should constrain phoneme and stimulus."""
        result = ds_with_index.sel(word="hello")
        # "hello" is at [0.0, 2.5)
        # Phonemes in this range: "hh" [0, 0.8), "eh" [0.8, 1.7), "l" [1.7, 2.5)
        # Stimulus overlapping: "image_A" [0.0, 5.0)
        assert result.sizes["word"] == 1
        assert result.sizes["stimulus"] == 1
        # Check phonemes are constrained
        assert result.sizes["phoneme"] <= 5  # Some phonemes should be filtered

    def test_sel_stimulus_constrains_others(self, ds_with_index):
        """Selecting a stimulus should constrain words and phonemes."""
        result = ds_with_index.sel(stimulus="image_A")
        # "image_A" is at [0.0, 5.0)
        # Words in this range: "hello" [0, 2.5), "world" [3.0, 5.5) partially overlaps
        assert result.sizes["stimulus"] == 1
        # Should have multiple words
        assert result.sizes["word"] >= 1

    def test_sel_time_constrains_all(self, ds_with_index):
        """Selecting time should constrain all event types."""
        result = ds_with_index.sel(time=slice(1.0, 2.0))
        # All dimensions should be constrained to overlapping intervals
        assert result.sizes["time"] > 0
        # Check that dimensions are reasonably constrained
        assert result.sizes["word"] <= 3
        assert result.sizes["phoneme"] <= 5
        assert result.sizes["stimulus"] <= 2


class TestIterativeApproach:
    """Tests demonstrating iterative application of intervals_from_dataframe."""

    def test_manual_iteration_matches_long_function(self):
        """Manual iteration should produce same result as intervals_from_long_dataframe."""
        df = mixed_event_annotations()

        # Method 1: Use the long dataframe function
        ds_auto = intervals_from_long_dataframe(df)

        # Method 2: Manual iteration
        datasets = []
        for event_type in df["event_type"].unique():
            subset = df[df["event_type"] == event_type].drop(columns=["event_type"])
            ds_subset = intervals_from_dataframe(subset, event_type, label_col="label")
            datasets.append(ds_subset)
        ds_manual = xr.merge(datasets)

        # Should have same structure
        assert set(ds_auto.dims) == set(ds_manual.dims)
        assert set(ds_auto.coords) == set(ds_manual.coords)

        # Should have same values
        for coord in ds_auto.coords:
            np.testing.assert_array_equal(
                ds_auto[coord].values,
                ds_manual[coord].values,
            )

    def test_filter_and_apply_workflow(self):
        """Test filtering DataFrame and applying intervals_from_dataframe."""
        df = mixed_event_annotations()

        # User wants only words and phonemes, not stimuli
        word_df = df[df["event_type"] == "word"].drop(columns=["event_type"])
        phoneme_df = df[df["event_type"] == "phoneme"].drop(columns=["event_type"])

        word_ds = intervals_from_dataframe(word_df, "word", label_col="label")
        phoneme_ds = intervals_from_dataframe(phoneme_df, "phoneme", label_col="label")

        ds = xr.merge([word_ds, phoneme_ds])

        # Should NOT have stimulus dimension
        assert "word" in ds.dims
        assert "phoneme" in ds.dims
        assert "stimulus" not in ds.dims
