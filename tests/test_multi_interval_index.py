"""Tests for DimensionIntervalMulti.

Note: Many tests use `_ = result * 1` to force evaluation. This is necessary
because xarray uses lazy indexing - the index's isel() method is only called
when the data is actually accessed, not when sel()/isel() is called on the
dataset. Without forcing evaluation, the dimension sizes may not reflect the
constrained values from our custom index.
"""

import pytest

from linked_indices.multi_interval_index import DimensionIntervalMulti
from linked_indices.util import multi_interval_dataset


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
        DimensionIntervalMulti,
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
                DimensionIntervalMulti,
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
                DimensionIntervalMulti,
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
