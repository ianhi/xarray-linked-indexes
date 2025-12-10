import pytest

from linked_indices.interval_index import DimensionInterval
from linked_indices.util import interval_dataset


# =============================================================================
# Simple case fixtures - continuous and intervals only (no word coord)
# =============================================================================


@pytest.fixture
def ds_simple():
    """Create a dataset with DimensionInterval index (intervals as dim, no word)."""
    ds = interval_dataset(interval_dim="intervals")
    ds = (
        ds.drop_indexes(["time", "intervals"])
        .set_xindex(["time", "intervals"], DimensionInterval)
        .drop_vars("word")
    )
    return ds


# =============================================================================
# Simple case: sel tests
# =============================================================================


class TestSimpleSel:
    """Tests for sel on simple case (continuous + intervals only)."""

    def test_sel_time_scalar(self, ds_simple):
        """Selecting a single time value should return dataset with time=1, intervals=1."""
        result = ds_simple.sel(time=10)
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 1
        assert result.sizes["intervals"] == 1
        assert result["time"].values == 10
        # The interval should contain the selected time point
        interval = result["intervals"].values[0]
        assert interval.left <= 10 < interval.right

    def test_sel_time_slice(self, ds_simple):
        """Selecting a time slice should return matching time range and intervals."""
        result = ds_simple.sel(time=slice(10, 1500))
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 299  # 10, 15, 20, ..., 1500
        assert result.sizes["intervals"] == 2  # [0, 1000) and [1000, 2500)
        assert result["time"].values[0] == 10
        assert result["time"].values[-1] == 1500

    def test_sel_intervals_scalar(self, ds_simple):
        """Selecting on intervals dimension with scalar."""
        result = ds_simple.sel(intervals=500)
        _ = result * 1  # force evaluation

        assert result.sizes["intervals"] == 1
        # Time should be constrained to the interval [0, 1000)
        assert result["time"].min().values == 0
        assert result["time"].max().values == 1000  # right bound of interval


# =============================================================================
# Simple case: isel tests
# =============================================================================


class TestSimpleIsel:
    """Tests for isel on simple case (continuous + intervals only)."""

    def test_isel_time_scalar(self, ds_simple):
        """isel on time with scalar index."""
        result = ds_simple.isel(time=3)
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 1
        assert result.sizes["intervals"] == 1
        assert result["time"].values == 15  # index 3 -> value 15 (0, 5, 10, 15)

    def test_isel_time_slice(self, ds_simple):
        """isel on time with slice."""
        result = ds_simple.isel(time=slice(3, 5))
        _ = result * 1  # force evaluation

        assert result.sizes["time"] == 2
        assert result.sizes["intervals"] == 1
        assert list(result["time"].values) == [15, 20]

    def test_isel_intervals_scalar(self, ds_simple):
        """isel on intervals with scalar index."""
        result = ds_simple.isel(intervals=3)
        _ = result * 1  # force evaluation

        assert result.sizes["intervals"] == 1
        # Should select the 4th interval [3500, 4995)
        interval = result["intervals"].values[0]
        assert interval.left == 3500
        # Time should be constrained to this interval
        assert result["time"].min().values == 3500
        assert result["time"].max().values == 4995


# =============================================================================
# Complex case fixtures - named intervals (word as dimension)
# =============================================================================


@pytest.fixture
def ds_with_word():
    """Create a dataset with word as dimension for intervals."""
    ds = interval_dataset(interval_dim="word")
    ds = ds.drop_indexes(["time"]).set_xindex(
        ["time", "intervals"],
        DimensionInterval,
    )
    return ds


# =============================================================================
# Complex case: sel tests
# =============================================================================


class TestComplexSel:
    """Tests for sel on complex case (word as dimension)."""

    def test_sel_word(self, ds_with_word):
        """Selecting by word should work and constrain time."""
        result = ds_with_word.sel(word="green")
        _ = result * 1  # force evaluation

        assert result.sizes["word"] == 1
        # Time should be constrained to the interval for "green" [0, 1000)
        assert result["time"].min().values == 0
        assert result["time"].max().values == 1000

    @pytest.mark.xfail(reason="sel on intervals doesn't slice when word is dim")
    def test_sel_intervals_with_word_dim(self, ds_with_word):
        """Selecting on intervals when word is the dimension."""
        result = ds_with_word.sel(intervals=500)
        _ = result * 1  # force evaluation
        # This currently doesn't do any slicing
        assert result.sizes["word"] == 1


# =============================================================================
# Complex case: isel tests - known limitations
# =============================================================================


class TestComplexIsel:
    """Tests for isel with word as dimension."""

    def test_isel_time_scalar_with_word_dim(self, ds_with_word):
        """isel(time=5) should work and constrain intervals."""
        result = ds_with_word.isel(time=5)

        assert result.sizes["time"] == 1
        assert result.sizes["word"] == 1
        assert result["time"].values == 25  # index 5 -> value 25

    def test_isel_time_and_word_together(self, ds_with_word):
        """isel on both time and word works."""
        result = ds_with_word.isel(time=5, word=0)

        assert result.sizes["time"] == 1
        # word coord becomes scalar, but intervals still has word dim
        assert result["word"].values == "green"

    @pytest.mark.xfail(
        reason="sel(time=5) fails - dimension already exists as scalar variable"
    )
    def test_sel_time_scalar_with_word_dim(self, ds_with_word):
        """sel(time=5) fails with word as dimension."""
        # This raises:
        # ValueError: dimension 'word' already exists as a scalar variable
        result = ds_with_word.sel(time=5)
        assert result.sizes["time"] == 1

    @pytest.mark.xfail(
        reason="word coord not sliced - isel can't convey additional dim indexers"
    )
    def test_sel_time_slice_with_word_dim(self, ds_with_word):
        """sel(time=slice(1, 20)) should work with word as dimension."""
        result = ds_with_word.sel(time=slice(1, 20))

        # Force evaluation - fails because intervals has word dim size 1, but word has size 4
        _ = result * 1

        assert result.sizes["time"] == 4  # 5, 10, 15, 20
        assert result.sizes["word"] == 1  # constrained to one interval
