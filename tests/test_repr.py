"""Tests for index repr methods (__repr__ and _repr_html_)."""

import pytest

from linked_indices import DimensionInterval, NDIndex
from linked_indices.example_data import multi_interval_dataset, trial_based_dataset


# =============================================================================
# DimensionInterval repr tests
# =============================================================================


class TestDimensionIntervalRepr:
    """Tests for DimensionInterval.__repr__."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a dataset with DimensionInterval index."""
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

    def test_repr_contains_class_name(self, ds_indexed):
        """repr should contain the class name."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        assert "DimensionInterval" in repr_str

    def test_repr_contains_continuous_dim(self, ds_indexed):
        """repr should show continuous dimension info."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        assert "Continuous: time" in repr_str
        assert "size: 1000" in repr_str
        assert "range:" in repr_str

    def test_repr_contains_interval_dims(self, ds_indexed):
        """repr should show interval dimension info."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        assert "Interval dimensions:" in repr_str
        assert "word:" in repr_str
        assert "phoneme:" in repr_str

    def test_repr_shows_interval_coords(self, ds_indexed):
        """repr should show interval coordinate names."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        assert "word_intervals" in repr_str
        assert "phoneme_intervals" in repr_str

    def test_repr_shows_labels(self, ds_indexed):
        """repr should show label coordinates."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        # Labels for word dimension
        assert "labels:" in repr_str
        assert "word" in repr_str
        assert "part_of_speech" in repr_str
        assert "phoneme" in repr_str

    def test_repr_shows_closed_property(self, ds_indexed):
        """repr should show interval closed property."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        assert "closed:" in repr_str
        assert "'left'" in repr_str

    def test_repr_shows_size_and_range(self, ds_indexed):
        """repr should show size and range for intervals."""
        index = ds_indexed.xindexes["time"]
        repr_str = repr(index)
        # Word has 3 intervals
        assert "size: 3" in repr_str
        # Phoneme has 6 intervals
        assert "size: 6" in repr_str


class TestDimensionIntervalReprOnsetDuration:
    """Tests for DimensionInterval repr with onset/duration format."""

    @pytest.fixture
    def ds_onset_duration(self):
        """Create a dataset with onset/duration coordinates."""
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

    def test_repr_shows_onset_duration_indicator(self, ds_onset_duration):
        """repr should indicate onset/duration format."""
        index = ds_onset_duration.xindexes["time"]
        repr_str = repr(index)
        assert "from onset/duration" in repr_str


class TestDimensionIntervalHtmlRepr:
    """Tests for DimensionInterval._repr_html_."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a dataset with DimensionInterval index."""
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

    def test_repr_html_returns_string(self, ds_indexed):
        """_repr_html_ should return a string."""
        index = ds_indexed.xindexes["time"]
        html = index._repr_html_()
        assert isinstance(html, str)

    def test_repr_html_contains_html_elements(self, ds_indexed):
        """_repr_html_ should contain HTML elements."""
        index = ds_indexed.xindexes["time"]
        html = index._repr_html_()
        assert "<div" in html
        assert "<table>" in html
        assert "<tr>" in html
        assert "<th>" in html
        assert "<td>" in html
        assert "</table>" in html
        assert "</div>" in html

    def test_repr_html_contains_style(self, ds_indexed):
        """_repr_html_ should contain CSS styling."""
        index = ds_indexed.xindexes["time"]
        html = index._repr_html_()
        assert "<style>" in html
        assert ".di-repr" in html

    def test_repr_html_contains_class_name(self, ds_indexed):
        """_repr_html_ should contain the class name."""
        index = ds_indexed.xindexes["time"]
        html = index._repr_html_()
        assert "DimensionInterval" in html

    def test_repr_html_contains_dimension_info(self, ds_indexed):
        """_repr_html_ should contain dimension information."""
        index = ds_indexed.xindexes["time"]
        html = index._repr_html_()
        assert "time" in html
        assert "word" in html
        assert "phoneme" in html
        assert "Continuous Dimension" in html
        assert "Interval Dimensions" in html


# =============================================================================
# NDIndex repr tests
# =============================================================================


class TestNDIndexRepr:
    """Tests for NDIndex.__repr__."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a dataset with NDIndex."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_repr_contains_class_name(self, ds_indexed):
        """repr should contain the class name."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "NDIndex" in repr_str

    def test_repr_shows_slice_method(self, ds_indexed):
        """repr should show slice_method."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "slice_method:" in repr_str
        assert "'bounding_box'" in repr_str

    def test_repr_shows_coordinates(self, ds_indexed):
        """repr should show coordinates section."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "Coordinates:" in repr_str
        assert "abs_time:" in repr_str

    def test_repr_shows_dims(self, ds_indexed):
        """repr should show coordinate dimensions."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "dims:" in repr_str
        assert "trial" in repr_str
        assert "rel_time" in repr_str

    def test_repr_shows_shape(self, ds_indexed):
        """repr should show coordinate shape."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "shape:" in repr_str
        # 3 trials × 50 rel_time points
        assert "3 × 50" in repr_str

    def test_repr_shows_range(self, ds_indexed):
        """repr should show value range."""
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "range:" in repr_str


class TestNDIndexReprTrimOuter:
    """Tests for NDIndex repr with trim_outer slice method."""

    def test_repr_shows_trim_outer(self):
        """repr should show trim_outer when configured."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        index = ds_indexed.xindexes["abs_time"]
        repr_str = repr(index)
        assert "'trim_outer'" in repr_str


class TestNDIndexReprMultipleCoords:
    """Tests for NDIndex repr with multiple coordinates."""

    @pytest.fixture
    def ds_multi(self):
        """Create dataset with multiple 2D coords."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        ds = ds.assign_coords(
            {"normalized_time": ds.abs_time / float(ds.abs_time.max())}
        )
        return ds.set_xindex(["abs_time", "normalized_time"], NDIndex)

    def test_repr_shows_multiple_coords(self, ds_multi):
        """repr should show all coordinates."""
        index = ds_multi.xindexes["abs_time"]
        repr_str = repr(index)
        assert "abs_time:" in repr_str
        assert "normalized_time:" in repr_str


class TestNDIndexHtmlRepr:
    """Tests for NDIndex._repr_html_."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a dataset with NDIndex."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_repr_html_returns_string(self, ds_indexed):
        """_repr_html_ should return a string."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert isinstance(html, str)

    def test_repr_html_contains_html_elements(self, ds_indexed):
        """_repr_html_ should contain HTML elements."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert "<div" in html
        assert "<table>" in html
        assert "<tr>" in html
        assert "<th>" in html
        assert "<td>" in html
        assert "</table>" in html
        assert "</div>" in html

    def test_repr_html_contains_style(self, ds_indexed):
        """_repr_html_ should contain CSS styling."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert "<style>" in html
        assert ".nd-repr" in html

    def test_repr_html_contains_class_name(self, ds_indexed):
        """_repr_html_ should contain the class name."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert "NDIndex" in html

    def test_repr_html_contains_coordinate_info(self, ds_indexed):
        """_repr_html_ should contain coordinate information."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert "abs_time" in html
        assert "N-D Coordinates" in html
        assert "Coordinate" in html
        assert "Dimensions" in html
        assert "Shape" in html
        assert "Range" in html

    def test_repr_html_shows_slice_method(self, ds_indexed):
        """_repr_html_ should show slice method."""
        index = ds_indexed.xindexes["abs_time"]
        html = index._repr_html_()
        assert "slice_method" in html
        assert "bounding_box" in html


# =============================================================================
# Integration tests
# =============================================================================


class TestReprIntegration:
    """Integration tests for repr display in different contexts."""

    def test_dimension_interval_repr_in_print(self):
        """DimensionInterval repr works with print()."""
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
        index = ds.xindexes["time"]
        # Should not raise
        output = str(index)
        assert len(output) > 0

    def test_nd_index_repr_in_print(self):
        """NDIndex repr works with print()."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        index = ds_indexed.xindexes["abs_time"]
        # Should not raise
        output = str(index)
        assert len(output) > 0

    def test_repr_after_slicing_dimension_interval(self):
        """DimensionInterval repr works after slicing."""
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
        # Slice the dataset
        ds_sliced = ds.sel(time=slice(20, 60))
        _ = ds_sliced * 1  # Force evaluation

        # repr should still work
        index = ds_sliced.xindexes["time"]
        repr_str = repr(index)
        assert "DimensionInterval" in repr_str

    def test_repr_after_slicing_nd_index(self):
        """NDIndex repr works after slicing."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)

        # Slice the dataset
        ds_sliced = ds_indexed.isel(trial=slice(0, 2))

        # repr should still work
        if "abs_time" in ds_sliced.xindexes:
            index = ds_sliced.xindexes["abs_time"]
            repr_str = repr(index)
            assert "NDIndex" in repr_str
