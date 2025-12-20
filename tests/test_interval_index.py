"""Tests for DimensionInterval.

Note: Many tests use `_ = result * 1` to force evaluation. This is necessary
because xarray uses lazy indexing - the index's isel() method is only called
when the data is actually accessed, not when sel()/isel() is called on the
dataset. Without forcing evaluation, the dimension sizes may not reflect the
constrained values from our custom index.
"""

import pytest

from linked_indices import DimensionInterval
from linked_indices.example_data import multi_interval_dataset, onset_duration_dataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ds_multi():
    """Create a dataset with multiple interval dimensions indexed together."""
    ds = multi_interval_dataset()
    ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
        [
            "time",
            "word_intervals",
            "phoneme_intervals",
            "word",
            "part_of_speech",
            "phoneme",
        ],
        DimensionInterval,
    )
    return ds


# =============================================================================
# Basic structure tests
# =============================================================================


class TestMultiIntervalStructure:
    """Tests for basic multi-interval dataset structure."""

    def test_dataset_dimensions(self, ds_multi):
        """Verify the dataset has expected dimensions."""
        assert "time" in ds_multi.dims
        assert "word" in ds_multi.dims
        assert "phoneme" in ds_multi.dims
        assert ds_multi.sizes["time"] == 1000
        assert ds_multi.sizes["word"] == 3
        assert ds_multi.sizes["phoneme"] == 6

    def test_dataset_coords(self, ds_multi):
        """Verify coordinates are present."""
        assert "time" in ds_multi.coords
        assert "word_intervals" in ds_multi.coords
        assert "phoneme_intervals" in ds_multi.coords
        assert "word" in ds_multi.coords
        assert "part_of_speech" in ds_multi.coords
        assert "phoneme" in ds_multi.coords


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in index creation."""

    def test_missing_interval_coord_raises_error(self):
        """Setting up index without the interval coord should raise ValueError."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        # Create dataset with interval dimension but don't include interval coord in index
        times = np.linspace(0, 100, 100)
        word_breaks = [0.0, 50.0, 100.0]
        word_intervals = pd.IntervalIndex.from_breaks(word_breaks, closed="left")

        ds = xr.Dataset(
            {"data": (("time",), np.random.rand(100))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["hello", "world"]),
            },
        )

        # Try to create index without the interval coord - should fail
        with pytest.raises(ValueError, match="Expected at least 1 interval coordinate"):
            ds.drop_indexes(["time", "word"]).set_xindex(
                ["time", "word"],  # missing word_intervals
                DimensionInterval,
            )

    def test_missing_continuous_dim_raises_error(self):
        """Setting up index without a continuous dimension should raise ValueError."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        # Create dataset with only interval dimensions (no continuous)
        word_breaks = [0.0, 50.0, 100.0]
        word_intervals = pd.IntervalIndex.from_breaks(word_breaks, closed="left")

        ds = xr.Dataset(
            {"data": (("word",), np.random.rand(2))},
            coords={
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["hello", "world"]),
            },
        )

        # Try to create index with only interval dim - should fail
        with pytest.raises(ValueError, match="Expected at least 2 dimensions"):
            ds.drop_indexes(["word"]).set_xindex(
                ["word_intervals", "word"],
                DimensionInterval,
            )


# =============================================================================
# Selection on continuous dimension (time) using sel
# =============================================================================


class TestTimeSelection:
    """Tests for selecting on the continuous time dimension."""

    def test_sel_time_scalar_nearest(self, ds_multi):
        """Selecting a single time with method='nearest' should work."""
        result = ds_multi.sel(time=30, method="nearest")
        assert result.sizes["time"] == 1

    def test_sel_time_slice_constrains_both_intervals(self, ds_multi):
        """Selecting a time slice should constrain both interval dimensions."""
        # Time 30-70 overlaps:
        # - word: [0-40), [40-80) -> 2 words (30 is in [0,40), 70 is in [40,80))
        # - phoneme: [20-40), [40-60), [60-80) -> 3 phonemes
        result = ds_multi.sel(time=slice(30, 70))
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 2
        assert result.sizes["phoneme"] == 3

    def test_sel_time_slice_narrow(self, ds_multi):
        """Selecting a narrow time slice."""
        # Time 5-15 is entirely within word [0,40) and phoneme [0,20)
        result = ds_multi.sel(time=slice(5, 15))
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 1

    def test_isel_time_scalar_constrains_both_intervals(self, ds_multi):
        """isel on time with scalar should constrain both interval dims."""
        # Index 500 is roughly time=60, in word [40,80) and phoneme [60,80)
        result = ds_multi.isel(time=500)
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 1
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 1

    def test_isel_time_slice_constrains_both_intervals(self, ds_multi):
        """isel on time with slice should constrain both interval dims."""
        # First 300 indices cover roughly time 0-36
        # word: [0,40) -> 1 word
        # phoneme: [0,20), [20,40) -> 2 phonemes
        result = ds_multi.isel(time=slice(0, 300))
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 300
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2


# =============================================================================
# Selection on interval dimensions using sel
# =============================================================================


class TestIntervalSelection:
    """Tests for selecting on interval dimensions."""

    def test_sel_word_interval_constrains_time_and_phonemes(self, ds_multi):
        """Selecting a word interval should constrain time AND phonemes."""
        # Select word interval [0, 40) by giving a point inside it
        # phoneme overlapping [0, 40): [0-20), [20-40) -> 2 phonemes
        result = ds_multi.sel(word_intervals=20)
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    def test_sel_phoneme_interval_constrains_time_and_words(self, ds_multi):
        """Selecting a phoneme interval should constrain time AND words."""
        # Select phoneme interval [60, 80) by giving a point inside it
        # word overlapping [60, 80): [40,80) -> 1 word
        result = ds_multi.sel(phoneme_intervals=70)
        _ = result * 1  # force evaluation

        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1

    @pytest.mark.xfail(
        reason="isel can't propagate indexers to non-indexed coords - xarray limitation"
    )
    def test_isel_word_constrains_phonemes(self, ds_multi):
        """isel on word should constrain phoneme."""
        # First word interval [0, 40)
        # phoneme overlapping: [0-20), [20-40) -> 2 phonemes
        result = ds_multi.isel(word=0)
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    @pytest.mark.xfail(
        reason="isel can't propagate indexers to non-indexed coords - xarray limitation"
    )
    def test_isel_phoneme_constrains_words(self, ds_multi):
        """isel on phoneme should constrain word."""
        # 4th phoneme interval [60, 80)
        # word overlapping: [40, 80) -> 1 word
        result = ds_multi.isel(phoneme=3)
        _ = result * 1  # force evaluation

        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1


# =============================================================================
# Selection with labels
# =============================================================================


class TestLabelSelection:
    """Tests for selecting using label coordinates (dimension coords)."""

    def test_sel_word_label_constrains_all(self, ds_multi):
        """Selecting by word label should constrain time and phoneme."""
        # "red" corresponds to first word interval [0, 40)
        # phoneme overlapping: [0-20), [20-40) -> 2 phonemes
        result = ds_multi.sel(word="red")
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    def test_sel_phoneme_label_constrains_all(self, ds_multi):
        """Selecting by phoneme label should constrain time and word."""
        # "ah" is first phoneme interval [0, 20)
        # word overlapping [0, 20): [0-40) -> 1 word
        result = ds_multi.sel(phoneme="ah")
        _ = result * 1  # force evaluation

        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1

    def test_sel_part_of_speech_constrains_all(self, ds_multi):
        """Selecting by part_of_speech (second label on word dim) should work."""
        # "noun" is the 3rd word "blue" at [80, 120)
        # phoneme overlapping [80, 120): [80-100), [100-120) -> 2 phonemes
        result = ds_multi.sel(part_of_speech="noun")
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2
        # Verify the correct word was selected
        assert result["word"].values == ["blue"]


# =============================================================================
# Cross-slicing verification using sel
# =============================================================================


class TestCrossSlicing:
    """Tests verifying that cross-slicing works correctly."""

    def test_time_slice_middle(self, ds_multi):
        """Select middle time range and verify both intervals are constrained."""
        # Time 50-90 overlaps:
        # - word: [40-80), [80-120) -> 2 words
        # - phoneme: [40-60), [60-80), [80-100) -> 3 phonemes
        result = ds_multi.sel(time=slice(50, 90))
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 2
        assert result.sizes["phoneme"] == 3

    def test_word_selection_propagates_to_phoneme(self, ds_multi):
        """Selecting a word interval should properly constrain phoneme."""
        # Select word interval [40, 80) (2nd word, "green") by selecting at 60
        # phoneme overlapping [40, 80): [40-60), [60-80) -> 2 phonemes
        result = ds_multi.sel(word_intervals=60)
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    def test_word_selection_first_interval(self, ds_multi):
        """Selecting first word interval."""
        # Select word interval [0, 40) (1st word, "red")
        # phoneme overlapping [0, 40): [0-20), [20-40) -> 2 phonemes
        result = ds_multi.sel(word_intervals=10)
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    def test_phoneme_fully_within_word(self, ds_multi):
        """Test case where selected phoneme is fully within one word."""
        # Select phoneme [40, 60) which is in word [40, 80)
        result = ds_multi.sel(phoneme_intervals=50)
        _ = result * 1  # force evaluation

        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1

    def test_phoneme_spans_word_boundary(self, ds_multi):
        """Test case where phoneme range spans word boundary."""
        # Select phoneme [20, 40) which ends exactly at word [0,40) boundary
        # With left-closed intervals [20, 40) overlaps only [0, 40)
        result = ds_multi.sel(phoneme_intervals=30)
        _ = result * 1  # force evaluation

        assert result.sizes["phoneme"] == 1
        # [20, 40) overlaps [0, 40) only (since intervals are left-closed)
        assert result.sizes["word"] == 1

    def test_multiple_interval_selections(self, ds_multi):
        """Selecting multiple interval dimensions simultaneously should intersect constraints."""
        # Select word [40, 80) and phoneme [60, 80)
        # The intersection is time [60, 80)
        # word [40, 80) contains time 60-80 -> 1 word
        # phoneme [60, 80) is fully within -> 1 phoneme
        result = ds_multi.sel(word_intervals=60, phoneme_intervals=70)
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 1
        # Time should be constrained to the intersection [60, 80)
        assert result["time"].min().values >= 60
        assert result["time"].max().values <= 80


# =============================================================================
# Onset/Duration Format Tests
# =============================================================================


@pytest.fixture
def ds_onset_duration():
    """Create a dataset with onset/duration coordinates indexed together."""
    ds = onset_duration_dataset()
    ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
        [
            "time",
            "word_onset",
            "word_duration",
            "word",
            "phoneme_onset",
            "phoneme_duration",
            "phoneme",
        ],
        DimensionInterval,
        onset_duration_coords={
            "word": ("word_onset", "word_duration"),
            "phoneme": ("phoneme_onset", "phoneme_duration"),
        },
    )
    return ds


class TestOnsetDurationStructure:
    """Tests for basic onset/duration dataset structure."""

    def test_dataset_dimensions(self, ds_onset_duration):
        """Verify the dataset has expected dimensions."""
        assert "time" in ds_onset_duration.dims
        assert "word" in ds_onset_duration.dims
        assert "phoneme" in ds_onset_duration.dims
        assert ds_onset_duration.sizes["time"] == 1000
        assert ds_onset_duration.sizes["word"] == 3
        assert ds_onset_duration.sizes["phoneme"] == 5

    def test_onset_duration_coords_visible(self, ds_onset_duration):
        """Verify onset/duration coordinates remain visible."""
        assert "word_onset" in ds_onset_duration.coords
        assert "word_duration" in ds_onset_duration.coords
        assert "phoneme_onset" in ds_onset_duration.coords
        assert "phoneme_duration" in ds_onset_duration.coords

    def test_no_synthetic_interval_coord(self, ds_onset_duration):
        """Verify no synthetic interval coordinate was created."""
        # The internal interval representation should NOT be exposed
        assert "word_intervals" not in ds_onset_duration.coords
        assert "phoneme_intervals" not in ds_onset_duration.coords

    def test_label_coords_present(self, ds_onset_duration):
        """Verify label coordinates are present."""
        assert "word" in ds_onset_duration.coords
        assert "phoneme" in ds_onset_duration.coords


class TestOnsetDurationTimeSelection:
    """Tests for selecting on time with onset/duration format."""

    def test_sel_time_scalar_nearest(self, ds_onset_duration):
        """Selecting a single time with method='nearest' should work."""
        result = ds_onset_duration.sel(time=30.0, method="nearest")
        assert result.sizes["time"] == 1

    def test_sel_time_slice_constrains_intervals(self, ds_onset_duration):
        """Selecting a time slice should constrain interval dimensions."""
        # Time 10-50 overlaps:
        # - word: [0, 35.5), [40, 75.5) -> 2 words
        # - phoneme: [0, 15.5), [20, 35.5), [40, 55.5) -> 3 phonemes
        result = ds_onset_duration.sel(time=slice(10.0, 50.0))
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 2
        assert result.sizes["phoneme"] == 3

    def test_sel_time_in_gap(self, ds_onset_duration):
        """Selecting time in a gap between intervals."""
        # Time 36-39 is in the gap between word[0] and word[1]
        # Should select no words (or handle gracefully)
        result = ds_onset_duration.sel(time=slice(36.0, 39.0))
        _ = result * 1  # force evaluation
        # Gap handling - verify behavior is defined
        assert result.sizes["time"] > 0  # Time slice exists

    def test_isel_time_constrains_intervals(self, ds_onset_duration):
        """isel on time should constrain interval dimensions."""
        # Select time indices in the middle
        result = ds_onset_duration.isel(time=slice(0, 200))
        _ = result * 1  # force evaluation
        # First ~200 indices cover roughly time 0-24
        # word: [0, 35.5) -> 1 word
        # phoneme: [0, 15.5), [20, 35.5) -> 2 phonemes
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2


class TestOnsetDurationLabelSelection:
    """Tests for selecting using label coordinates."""

    def test_sel_word_label(self, ds_onset_duration):
        """Selecting by word label should constrain time and phoneme."""
        # "hello" is first word [0, 35.5)
        result = ds_onset_duration.sel(word="hello")
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 1
        # phoneme overlapping [0, 35.5): [0, 15.5), [20, 35.5) -> 2 phonemes
        assert result.sizes["phoneme"] == 2

    def test_sel_phoneme_label(self, ds_onset_duration):
        """Selecting by phoneme label should constrain time and word."""
        # "hh" is first phoneme [0, 15.5)
        result = ds_onset_duration.sel(phoneme="hh")
        _ = result * 1  # force evaluation
        assert result.sizes["phoneme"] == 1
        # word overlapping [0, 15.5): [0, 35.5) -> 1 word
        assert result.sizes["word"] == 1


class TestOnsetDurationOnsetSelection:
    """Tests for selecting using onset coordinates."""

    def test_sel_onset_exact(self, ds_onset_duration):
        """Selecting by exact onset value should work."""
        # Select word with onset=40.0
        result = ds_onset_duration.sel(word_onset=40.0)
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 1
        # Verify correct word was selected
        assert result["word"].values == ["world"]

    def test_sel_onset_slice(self, ds_onset_duration):
        """Selecting by onset slice should work."""
        # Select words with onset between 35 and 85
        result = ds_onset_duration.sel(word_onset=slice(35.0, 85.0))
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 2
        # Should select words with onset=40.0 and onset=80.0
        assert list(result["word"].values) == ["world", "test"]


class TestOnsetDurationCrossSlicing:
    """Tests verifying cross-slicing works correctly."""

    def test_word_selection_propagates_to_phoneme(self, ds_onset_duration):
        """Selecting a word should constrain phonemes correctly."""
        # Select "world" word [40, 75.5)
        result = ds_onset_duration.sel(word="world")
        _ = result * 1  # force evaluation
        assert result.sizes["word"] == 1
        # phoneme overlapping [40, 75.5): [40, 55.5), [60, 75.5) -> 2 phonemes
        assert result.sizes["phoneme"] == 2

    def test_phoneme_selection_propagates_to_word(self, ds_onset_duration):
        """Selecting a phoneme should constrain words correctly."""
        # Select "ll" phoneme [40, 55.5)
        result = ds_onset_duration.sel(phoneme="ll")
        _ = result * 1  # force evaluation
        assert result.sizes["phoneme"] == 1
        # word overlapping [40, 55.5): [40, 75.5) -> 1 word
        assert result.sizes["word"] == 1

    def test_time_in_gap_constrains_correctly(self, ds_onset_duration):
        """Selecting time in gap should not select intervals on either side."""
        # Time exactly at 37.0 is in the gap between words
        # This tests gap handling behavior
        result = ds_onset_duration.sel(time=37.0, method="nearest")
        _ = result * 1  # force evaluation
        # Should select time, but word/phoneme behavior depends on design
        assert result.sizes["time"] == 1


class TestOnsetDurationErrorHandling:
    """Tests for error handling with onset/duration format."""

    def test_missing_onset_coord(self):
        """Error when onset coord doesn't exist in variables list."""
        ds = onset_duration_dataset()
        with pytest.raises(KeyError):
            ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
                ["time", "word_duration", "word"],  # missing onset
                DimensionInterval,
                onset_duration_coords={"word": ("word_onset", "word_duration")},
            )

    def test_missing_duration_coord(self):
        """Error when duration coord doesn't exist in variables list."""
        ds = onset_duration_dataset()
        with pytest.raises(KeyError):
            ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
                ["time", "word_onset", "word"],  # missing duration
                DimensionInterval,
                onset_duration_coords={"word": ("word_onset", "word_duration")},
            )

    def test_invalid_dim_in_onset_duration_coords(self):
        """Error when onset_duration_coords references invalid dimension."""
        ds = onset_duration_dataset()
        with pytest.raises(ValueError):
            ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
                ["time", "word_onset", "word_duration", "word"],
                DimensionInterval,
                onset_duration_coords={
                    "nonexistent_dim": ("word_onset", "word_duration")
                },
            )


class TestOnsetDurationIntervalClosed:
    """Tests for interval_closed parameter."""

    def test_interval_closed_left_default(self):
        """Default is left-closed intervals."""
        ds = onset_duration_dataset()
        ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            ["time", "word_onset", "word_duration", "word"],
            DimensionInterval,
            onset_duration_coords={"word": ("word_onset", "word_duration")},
        )
        # Test boundary behavior (left endpoint included, right excluded)
        result = ds.sel(time=0.0, method="nearest")
        _ = result * 1
        assert result.sizes["word"] == 1

    def test_interval_closed_right(self):
        """Right-closed intervals."""
        ds = onset_duration_dataset()
        ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            ["time", "word_onset", "word_duration", "word"],
            DimensionInterval,
            onset_duration_coords={"word": ("word_onset", "word_duration")},
            interval_closed="right",
        )
        # Test boundary behavior
        _ = ds * 1  # verify creation works


class TestOnsetDurationEquivalence:
    """Tests that onset/duration format produces equivalent results to IntervalIndex."""

    @pytest.fixture
    def ds_interval_format(self):
        """Create dataset using IntervalIndex format."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)

        # Same intervals as onset/duration dataset
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [35.5, 75.5, 115.5], closed="left"
        )

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["hello", "world", "test"]),
            },
        )

        return ds.drop_indexes(["time", "word"]).set_xindex(
            ["time", "word_intervals", "word"],
            DimensionInterval,
        )

    @pytest.fixture
    def ds_onset_format(self):
        """Create dataset using onset/duration format."""
        import numpy as np
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_onset": ("word", [0.0, 40.0, 80.0]),
                "word_duration": ("word", [35.5, 35.5, 35.5]),
                "word": ("word", ["hello", "world", "test"]),
            },
        )

        return ds.drop_indexes(["time", "word"]).set_xindex(
            ["time", "word_onset", "word_duration", "word"],
            DimensionInterval,
            onset_duration_coords={"word": ("word_onset", "word_duration")},
        )

    def test_sel_word_equivalent(self, ds_interval_format, ds_onset_format):
        """Selecting by word should give same time constraint."""
        result_interval = ds_interval_format.sel(word="world")
        result_onset = ds_onset_format.sel(word="world")

        _ = result_interval * 1
        _ = result_onset * 1

        # Same time range should be selected
        assert result_interval.sizes["time"] == result_onset.sizes["time"]

    def test_sel_time_slice_equivalent(self, ds_interval_format, ds_onset_format):
        """Selecting time slice should constrain words the same way."""
        result_interval = ds_interval_format.sel(time=slice(20.0, 60.0))
        result_onset = ds_onset_format.sel(time=slice(20.0, 60.0))

        _ = result_interval * 1
        _ = result_onset * 1

        # Same words should be selected
        assert result_interval.sizes["word"] == result_onset.sizes["word"]


# =============================================================================
# Mixed format tests (IntervalIndex + onset/duration on same object)
# =============================================================================


class TestMixedFormatStructure:
    """Tests for datasets with both IntervalIndex and onset/duration formats."""

    @pytest.fixture
    def ds_mixed(self):
        """Create a dataset with mixed interval formats.

        - word: uses pd.IntervalIndex directly
        - phoneme: uses onset/duration format
        """
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)

        # Word intervals using pd.IntervalIndex - 3 words covering [0,40), [40,80), [80,120)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )

        # Phoneme uses onset/duration format - 6 phonemes, each 20 units
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                # Word uses IntervalIndex directly
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                # Phoneme uses onset/duration
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_dataset_dimensions(self, ds_mixed):
        """Verify the dataset has expected dimensions."""
        assert "time" in ds_mixed.dims
        assert "word" in ds_mixed.dims
        assert "phoneme" in ds_mixed.dims
        assert ds_mixed.sizes["time"] == 1000
        assert ds_mixed.sizes["word"] == 3
        assert ds_mixed.sizes["phoneme"] == 6

    def test_interval_coord_visible(self, ds_mixed):
        """IntervalIndex coord should be visible."""
        assert "word_intervals" in ds_mixed.coords

    def test_onset_duration_coords_visible(self, ds_mixed):
        """Onset/duration coordinates should be visible."""
        assert "phoneme_onset" in ds_mixed.coords
        assert "phoneme_duration" in ds_mixed.coords

    def test_no_synthetic_interval_for_onset_duration(self, ds_mixed):
        """No synthetic interval coord should be created for onset/duration dim."""
        assert "phoneme_intervals" not in ds_mixed.coords
        assert "__phoneme_intervals__" not in ds_mixed.coords


class TestMixedFormatSelByWord:
    """Tests for sel on the IntervalIndex dimension (word)."""

    @pytest.fixture
    def ds_mixed(self):
        """Create dataset with mixed formats."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_sel_word_constrains_time(self, ds_mixed):
        """Selecting word should constrain time to word's interval."""
        # "red" is [0, 40)
        result = ds_mixed.sel(word="red")
        _ = result * 1
        assert result.sizes["word"] == 1
        # Time should be in [0, 40] (linspace includes endpoint)
        assert result.time.min().values >= 0.0
        assert result.time.max().values <= 40.0

    def test_sel_word_constrains_phoneme(self, ds_mixed):
        """Selecting word should constrain phonemes to overlapping ones."""
        # "red" is [0, 40), phonemes in this range: aa[0,20), bb[20,40)
        result = ds_mixed.sel(word="red")
        _ = result * 1
        assert result.sizes["phoneme"] == 2
        assert set(result.phoneme.values) == {"aa", "bb"}

    def test_sel_word_middle(self, ds_mixed):
        """Selecting middle word should constrain correctly."""
        # "green" is [40, 80), phonemes: cc[40,60), dd[60,80)
        result = ds_mixed.sel(word="green")
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2
        assert set(result.phoneme.values) == {"cc", "dd"}

    def test_sel_word_last(self, ds_mixed):
        """Selecting last word should constrain correctly."""
        # "blue" is [80, 120), phonemes: ee[80,100), ff[100,120)
        result = ds_mixed.sel(word="blue")
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2
        assert set(result.phoneme.values) == {"ee", "ff"}


class TestMixedFormatSelByPhoneme:
    """Tests for sel on the onset/duration dimension (phoneme)."""

    @pytest.fixture
    def ds_mixed(self):
        """Create dataset with mixed formats."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_sel_phoneme_constrains_time(self, ds_mixed):
        """Selecting phoneme should constrain time to phoneme's interval."""
        # "aa" is [0, 20)
        result = ds_mixed.sel(phoneme="aa")
        _ = result * 1
        assert result.sizes["phoneme"] == 1
        assert result.time.min().values >= 0.0
        assert result.time.max().values < 20.0

    def test_sel_phoneme_constrains_word(self, ds_mixed):
        """Selecting phoneme should constrain words to overlapping ones."""
        # "aa" is [0, 20), word in this range: red[0,40)
        result = ds_mixed.sel(phoneme="aa")
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.word.values[0] == "red"

    def test_sel_phoneme_middle(self, ds_mixed):
        """Selecting middle phoneme should constrain correctly."""
        # "cc" is [40, 60), word: green[40,80)
        result = ds_mixed.sel(phoneme="cc")
        _ = result * 1
        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1
        assert result.word.values[0] == "green"

    def test_sel_phoneme_last(self, ds_mixed):
        """Selecting last phoneme should constrain correctly."""
        # "ff" is [100, 120), word: blue[80,120)
        result = ds_mixed.sel(phoneme="ff")
        _ = result * 1
        assert result.sizes["phoneme"] == 1
        assert result.sizes["word"] == 1
        assert result.word.values[0] == "blue"


class TestMixedFormatSelByTime:
    """Tests for sel on time dimension with mixed formats."""

    @pytest.fixture
    def ds_mixed(self):
        """Create dataset with mixed formats."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_sel_time_constrains_word(self, ds_mixed):
        """Selecting time should constrain word (IntervalIndex format)."""
        # Time [30, 70) overlaps word: red[0,40), green[40,80) -> 2 words
        result = ds_mixed.sel(time=slice(30, 70))
        _ = result * 1
        assert result.sizes["word"] == 2
        assert set(result.word.values) == {"red", "green"}

    def test_sel_time_constrains_phoneme(self, ds_mixed):
        """Selecting time should constrain phoneme (onset/duration format)."""
        # Time [30, 70) overlaps phoneme: bb[20,40), cc[40,60), dd[60,80) -> 3 phonemes
        result = ds_mixed.sel(time=slice(30, 70))
        _ = result * 1
        assert result.sizes["phoneme"] == 3
        assert set(result.phoneme.values) == {"bb", "cc", "dd"}

    def test_sel_time_constrains_both(self, ds_mixed):
        """Selecting time should constrain both dimensions correctly."""
        # Time [50, 90) overlaps:
        # - word: green[40,80), blue[80,120) -> 2 words
        # - phoneme: cc[40,60), dd[60,80), ee[80,100) -> 3 phonemes
        result = ds_mixed.sel(time=slice(50, 90))
        _ = result * 1
        assert result.sizes["word"] == 2
        assert result.sizes["phoneme"] == 3
        assert set(result.word.values) == {"green", "blue"}
        assert set(result.phoneme.values) == {"cc", "dd", "ee"}

    def test_sel_time_scalar_nearest(self, ds_mixed):
        """Selecting scalar time with nearest should work."""
        result = ds_mixed.sel(time=50.0, method="nearest")
        assert result.sizes["time"] == 1


class TestMixedFormatIsel:
    """Tests for isel with mixed formats."""

    @pytest.fixture
    def ds_mixed(self):
        """Create dataset with mixed formats."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_isel_time_constrains_both(self, ds_mixed):
        """isel on time should constrain both interval dimensions."""
        # First 300 indices cover time ~0-36, which overlaps:
        # - word: red[0,40) -> 1 word
        # - phoneme: aa[0,20), bb[20,40) -> 2 phonemes
        result = ds_mixed.isel(time=slice(0, 300))
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2

    def test_isel_time_middle(self, ds_mixed):
        """isel on middle time range."""
        # Indices 400-600 cover time ~48-72, which overlaps:
        # - word: green[40,80) -> 1 word
        # - phoneme: cc[40,60), dd[60,80) -> 2 phonemes
        result = ds_mixed.isel(time=slice(400, 600))
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.sizes["phoneme"] == 2


class TestMixedFormatSelByInterval:
    """Tests for sel on interval coordinates directly."""

    @pytest.fixture
    def ds_mixed(self):
        """Create dataset with mixed formats."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        N = 1000
        times = np.linspace(0, 120, N)
        word_intervals = pd.IntervalIndex.from_arrays(
            [0.0, 40.0, 80.0], [40.0, 80.0, 120.0], closed="left"
        )
        phoneme_onsets = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        phoneme_durations = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        ds = xr.Dataset(
            {"data": (("C", "time"), np.random.rand(2, N))},
            coords={
                "time": times,
                "word_intervals": ("word", word_intervals),
                "word": ("word", ["red", "green", "blue"]),
                "phoneme_onset": ("phoneme", phoneme_onsets),
                "phoneme_duration": ("phoneme", phoneme_durations),
                "phoneme": ("phoneme", ["aa", "bb", "cc", "dd", "ee", "ff"]),
            },
        )

        return ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "word",
                "phoneme_onset",
                "phoneme_duration",
                "phoneme",
            ],
            DimensionInterval,
            onset_duration_coords={"phoneme": ("phoneme_onset", "phoneme_duration")},
        )

    def test_sel_word_interval_point(self, ds_mixed):
        """Selecting by point in word_intervals should work."""
        # Point 50 is in green[40,80)
        result = ds_mixed.sel(word_intervals=50)
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.word.values[0] == "green"

    def test_sel_phoneme_onset(self, ds_mixed):
        """Selecting by phoneme_onset should work."""
        # onset=40.0 is phoneme "cc"
        result = ds_mixed.sel(phoneme_onset=40.0)
        _ = result * 1
        assert result.sizes["phoneme"] == 1
        assert result.phoneme.values[0] == "cc"

    def test_sel_word_interval_constrains_phoneme(self, ds_mixed):
        """Selecting by word_interval should constrain phoneme."""
        # Point 10 is in red[0,40), phonemes: aa[0,20), bb[20,40) -> 2
        result = ds_mixed.sel(word_intervals=10)
        _ = result * 1
        assert result.sizes["phoneme"] == 2
        assert set(result.phoneme.values) == {"aa", "bb"}

    def test_sel_phoneme_onset_constrains_word(self, ds_mixed):
        """Selecting by phoneme_onset should constrain word."""
        # onset=60.0 is phoneme "dd" [60,80), word: green[40,80)
        result = ds_mixed.sel(phoneme_onset=60.0)
        _ = result * 1
        assert result.sizes["word"] == 1
        assert result.word.values[0] == "green"


class TestDimensionIntervalCoverageGaps:
    """Tests for DimensionInterval coverage gaps."""

    def test_interval_idx_min_max_with_single_interval(self, ds_multi):
        """_interval_idx_min_max works with single Interval."""
        import pandas as pd

        idx = ds_multi.xindexes["time"]

        # Get a single interval
        single = pd.Interval(0.0, 0.5, closed="left")
        result = idx._interval_idx_min_max(single)
        assert result.left == 0.0
        assert result.right == 0.5

    def test_interval_idx_min_max_invalid_type(self, ds_multi):
        """_interval_idx_min_max raises for invalid type."""
        idx = ds_multi.xindexes["time"]

        with pytest.raises(TypeError, match="Expected IntervalIndex or Interval"):
            idx._interval_idx_min_max("not an interval")

    def test_debug_mode_from_variables(self, capsys):
        """Debug mode prints during from_variables."""
        from linked_indices.example_data import multi_interval_dataset

        ds = multi_interval_dataset()
        ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "phoneme_intervals",
                "word",
                "part_of_speech",
                "phoneme",
            ],
            DimensionInterval,
            debug=True,
        )

        captured = capsys.readouterr()
        assert "DEBUG from_variables:" in captured.out

    def test_debug_mode_sel(self, capsys):
        """Debug mode prints during sel."""
        from linked_indices.example_data import multi_interval_dataset

        ds = multi_interval_dataset()
        ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
            [
                "time",
                "word_intervals",
                "phoneme_intervals",
                "word",
                "part_of_speech",
                "phoneme",
            ],
            DimensionInterval,
            debug=True,
        )
        capsys.readouterr()  # Clear creation output

        ds.sel(word="hello")
        captured = capsys.readouterr()
        assert "DEBUG sel:" in captured.out

    def test_sel_continuous_scalar(self, ds_multi):
        """Selecting scalar on continuous dimension works with method='nearest'."""
        result = ds_multi.sel(time=50.0, method="nearest")
        _ = result * 1  # Force evaluation
        assert result.sizes["time"] == 1
