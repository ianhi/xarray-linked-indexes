"""Tests for AbsoluteRelativeIndex - trial-based data with absolute/relative time."""

import numpy as np
import pytest
import xarray as xr

from linked_indices import AbsoluteRelativeIndex
from linked_indices.example_data import trial_based_dataset


class TestTrialBasedDataset:
    """Tests for the trial_based_dataset helper function."""

    def test_returns_dataset(self):
        """Function returns an xarray Dataset."""
        ds = trial_based_dataset()
        assert isinstance(ds, xr.Dataset)

    def test_default_dimensions(self):
        """Default dataset has expected dimensions."""
        ds = trial_based_dataset()
        assert "trial" in ds.dims
        assert "rel_time" in ds.dims
        assert ds.sizes["trial"] == 3
        assert ds.sizes["rel_time"] == 500  # 5s * 100 samples/s

    def test_default_waveforms(self):
        """Default dataset has cosine, square, sawtooth waveforms."""
        ds = trial_based_dataset()
        assert list(ds.trial.values) == ["cosine", "square", "sawtooth"]
        # Check that waveforms are distinct (not random noise)
        # Cosine at t=0 should be 1.0
        assert abs(float(ds.data.sel(trial="cosine").isel(rel_time=0)) - 1.0) < 0.01
        # Square at t=0 should be 1.0
        assert abs(float(ds.data.sel(trial="square").isel(rel_time=0)) - 1.0) < 0.01
        # Sawtooth at t=0 should be -1.0
        assert (
            abs(float(ds.data.sel(trial="sawtooth").isel(rel_time=0)) - (-1.0)) < 0.01
        )

    def test_custom_parameters(self):
        """Custom parameters are respected."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=50)
        assert ds.sizes["trial"] == 3
        assert ds.sizes["rel_time"] == 250  # 5s * 50 samples/s

    def test_abs_time_shape(self):
        """abs_time is a 2D coordinate."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        assert ds.abs_time.dims == ("trial", "rel_time")
        assert ds.abs_time.shape == (3, 50)

    def test_abs_time_values(self):
        """abs_time values are correct (trial_onset + rel_time)."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        # First trial starts at 0
        assert float(ds.abs_time[0, 0]) == 0.0
        # Second trial starts at 5
        assert float(ds.abs_time[1, 0]) == 5.0
        # Third trial starts at 10
        assert float(ds.abs_time[2, 0]) == 10.0

    def test_trial_onset_values(self):
        """trial_onset coordinate has correct values."""
        ds = trial_based_dataset(n_trials=4, trial_length=2.5)
        expected_onsets = [0.0, 2.5, 5.0, 7.5]
        np.testing.assert_array_almost_equal(ds.trial_onset.values, expected_onsets)

    def test_custom_trial_labels(self):
        """Custom trial labels are used."""
        labels = ["baseline", "stimulus", "recovery"]
        ds = trial_based_dataset(n_trials=3, trial_labels=labels)
        assert list(ds.trial.values) == labels

    def test_mismatched_labels_raises(self):
        """Mismatched trial_labels length raises ValueError."""
        with pytest.raises(ValueError, match="must match n_trials"):
            trial_based_dataset(n_trials=3, trial_labels=["a", "b"])


class TestAbsoluteRelativeIndexCreation:
    """Tests for creating an AbsoluteRelativeIndex."""

    @pytest.fixture
    def ds(self):
        """Create a trial-based dataset."""
        return trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)

    def test_create_index(self, ds):
        """Index can be created from trial-based dataset."""
        # Need to drop existing indexes before setting our custom index
        ds_indexed = ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )
        assert isinstance(ds_indexed.xindexes["abs_time"], AbsoluteRelativeIndex)

    def test_index_coords_remain(self, ds):
        """All coordinates remain visible after indexing."""
        ds_indexed = ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )
        assert "abs_time" in ds_indexed.coords
        assert "trial" in ds_indexed.coords
        assert "rel_time" in ds_indexed.coords
        assert "trial_onset" in ds_indexed.coords

    def test_data_preserved(self, ds):
        """Data variable is preserved."""
        ds_indexed = ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )
        assert "data" in ds_indexed.data_vars
        assert ds_indexed.data.shape == ds.data.shape


class TestAbsoluteRelativeIndexSelAbsTime:
    """Tests for selecting by absolute time."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_sel_abs_time_scalar_in_first_trial(self, ds_indexed):
        """Selecting abs_time in first trial returns correct trial."""
        # abs_time=2.5 is in trial 0 (which spans 0-5)
        result = ds_indexed.sel(abs_time=2.5, method="nearest")
        _ = result * 1  # force evaluation
        # Should return a single point
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        # Should be trial 0
        assert result.trial.values == "trial_0"
        # rel_time should be ~2.5
        assert abs(float(result.rel_time) - 2.5) < 0.15

    def test_sel_abs_time_scalar_in_second_trial(self, ds_indexed):
        """Selecting abs_time in second trial returns correct trial."""
        # abs_time=7.5 is in trial 1 (which spans 5-10)
        result = ds_indexed.sel(abs_time=7.5, method="nearest")
        _ = result * 1
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"
        # rel_time should be ~2.5 (7.5 - 5.0 = 2.5)
        assert abs(float(result.rel_time) - 2.5) < 0.15

    def test_sel_abs_time_slice_within_trial(self, ds_indexed):
        """Selecting abs_time slice within single trial."""
        # abs_time 1-3 is entirely in trial 0
        result = ds_indexed.sel(abs_time=slice(1.0, 3.0))
        _ = result * 1
        # Should be only trial 0
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_0"
        # rel_time should span approximately 1-3
        assert float(result.rel_time.min()) >= 0.9
        assert float(result.rel_time.max()) <= 3.1

    def test_sel_abs_time_slice_spanning_trials(self, ds_indexed):
        """Selecting abs_time slice that spans multiple trials."""
        # abs_time 4-7 spans trial 0 (4-5) and trial 1 (5-7)
        result = ds_indexed.sel(abs_time=slice(4.0, 7.0))
        _ = result * 1
        # Should include 2 trials
        assert result.sizes["trial"] == 2
        assert "trial_0" in result.trial.values
        assert "trial_1" in result.trial.values


class TestAbsoluteRelativeIndexSelTrial:
    """Tests for selecting by trial."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_sel_trial_scalar(self, ds_indexed):
        """Selecting a single trial."""
        result = ds_indexed.sel(trial="trial_1")
        _ = result * 1
        assert result.sizes["trial"] == 1
        # All rel_time points should be present
        assert result.sizes["rel_time"] == 50
        # abs_time should be constrained to trial 1's range (5-10)
        abs_min = float(result.abs_time.min())
        abs_max = float(result.abs_time.max())
        assert abs_min >= 4.9
        assert abs_max <= 10.1

    def test_sel_trial_multiple(self, ds_indexed):
        """Selecting multiple trials."""
        result = ds_indexed.sel(trial=["trial_0", "trial_2"])
        _ = result * 1
        assert result.sizes["trial"] == 2
        assert "trial_0" in result.trial.values
        assert "trial_2" in result.trial.values


class TestAbsoluteRelativeIndexSelRelTime:
    """Tests for selecting by relative time."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_sel_rel_time_scalar(self, ds_indexed):
        """Selecting a single relative time across all trials."""
        result = ds_indexed.sel(rel_time=2.5, method="nearest")
        _ = result * 1
        # Should have all trials
        assert result.sizes["trial"] == 3
        # Scalar selection drops the rel_time dimension
        assert "rel_time" not in result.dims

    def test_sel_rel_time_slice(self, ds_indexed):
        """Selecting a range of relative times."""
        result = ds_indexed.sel(rel_time=slice(1.0, 3.0))
        _ = result * 1
        # Should have all trials
        assert result.sizes["trial"] == 3
        # rel_time should be constrained
        assert float(result.rel_time.min()) >= 0.9
        assert float(result.rel_time.max()) <= 3.1


class TestAbsoluteRelativeIndexIsel:
    """Tests for integer-based selection."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_isel_trial(self, ds_indexed):
        """isel on trial dimension."""
        result = ds_indexed.isel(trial=1)
        _ = result * 1
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"

    def test_isel_rel_time(self, ds_indexed):
        """isel on rel_time dimension."""
        result = ds_indexed.isel(rel_time=slice(10, 20))
        _ = result * 1
        assert result.sizes["rel_time"] == 10
        # All trials should be present
        assert result.sizes["trial"] == 3

    def test_isel_both_dims(self, ds_indexed):
        """isel on both dimensions."""
        result = ds_indexed.isel(trial=0, rel_time=slice(0, 10))
        _ = result * 1
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 10

    def test_isel_rel_time_scalar(self, ds_indexed):
        """isel on rel_time with scalar index."""
        result = ds_indexed.isel(rel_time=5)
        _ = result * 1
        # Should have all trials and one rel_time point
        assert result.sizes["trial"] == 3
        assert result.sizes["rel_time"] == 1


class TestAbsoluteRelativeIndexCombinedSelection:
    """Tests for combined selections on multiple dimensions."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_sel_trial_and_rel_time(self, ds_indexed):
        """Select both trial and rel_time."""
        result = ds_indexed.sel(trial="trial_1", rel_time=slice(1.0, 3.0))
        _ = result * 1
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"
        # rel_time constrained
        assert float(result.rel_time.min()) >= 0.9
        assert float(result.rel_time.max()) <= 3.1

    def test_sel_abs_time_and_trial_consistent(self, ds_indexed):
        """Select abs_time within the range of a specific trial."""
        # abs_time 6-8 is in trial 1 (5-10), so specifying trial_1 should work
        result = ds_indexed.sel(abs_time=slice(6.0, 8.0), trial="trial_1")
        _ = result * 1
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"


class TestAbsoluteRelativeIndexEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
        )

    def test_sel_abs_time_outside_range(self, ds_indexed):
        """Selecting abs_time outside data range returns nearest boundary."""
        # abs_time=100 is way outside the range (0-15)
        # The current implementation finds the nearest point (last trial, last time)
        result = ds_indexed.sel(abs_time=100.0, method="nearest")
        _ = result * 1
        # Should return a single point (the boundary)
        assert result.sizes["trial"] == 1
        # Check that it selected the last trial
        assert result.trial.values == "trial_2"

    def test_sel_abs_time_at_trial_boundary(self, ds_indexed):
        """Selecting abs_time exactly at trial boundary."""
        # abs_time=5.0 is exactly at the boundary between trial 0 and 1
        result = ds_indexed.sel(abs_time=5.0, method="nearest")
        _ = result * 1
        # Should select exactly one point
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1

    def test_sel_abs_time_slice_outside_range(self, ds_indexed):
        """Selecting abs_time slice completely outside data range returns empty."""
        # abs_time 100-200 is way outside the range (0-15)
        result = ds_indexed.sel(abs_time=slice(100.0, 200.0))
        _ = result * 1
        # Should return empty slices
        assert result.sizes["trial"] == 0
        assert result.sizes["rel_time"] == 0


class TestAbsoluteRelativeIndexValidation:
    """Tests for validation and error handling during index creation."""

    def test_missing_2d_coord_raises(self):
        """Error when no 2D coordinate is provided."""
        import xarray as xr

        # Create a dataset with only 1D coordinates
        ds = xr.Dataset(
            {"data": (("trial", "rel_time"), np.random.rand(3, 10))},
            coords={
                "trial": ["a", "b", "c"],
                "rel_time": np.linspace(0, 1, 10),
            },
        )
        with pytest.raises(ValueError, match="requires exactly one 2D coordinate"):
            ds.drop_indexes(["trial", "rel_time"]).set_xindex(
                ["trial", "rel_time"],
                AbsoluteRelativeIndex,
            )

    def test_missing_trial_dim_coord_raises(self):
        """Error when trial dimension coordinate is missing."""
        import xarray as xr

        # Create a dataset with 2D abs_time but missing trial coord
        abs_time = np.arange(30).reshape(3, 10).astype(float)
        ds = xr.Dataset(
            {"data": (("trial", "rel_time"), np.random.rand(3, 10))},
            coords={
                "abs_time": (("trial", "rel_time"), abs_time),
                "rel_time": np.linspace(0, 1, 10),
                # trial coord missing!
            },
        )
        with pytest.raises(ValueError, match="Missing coordinate for trial dimension"):
            ds.drop_indexes(["rel_time"]).set_xindex(
                ["abs_time", "rel_time"],
                AbsoluteRelativeIndex,
            )

    def test_missing_rel_time_dim_coord_raises(self):
        """Error when rel_time dimension coordinate is missing."""
        import xarray as xr

        # Create a dataset with 2D abs_time but missing rel_time coord
        abs_time = np.arange(30).reshape(3, 10).astype(float)
        ds = xr.Dataset(
            {"data": (("trial", "rel_time"), np.random.rand(3, 10))},
            coords={
                "abs_time": (("trial", "rel_time"), abs_time),
                "trial": ["a", "b", "c"],
                # rel_time coord missing!
            },
        )
        with pytest.raises(
            ValueError, match="Missing coordinate for rel_time dimension"
        ):
            ds.drop_indexes(["trial"]).set_xindex(
                ["abs_time", "trial"],
                AbsoluteRelativeIndex,
            )


class TestAbsoluteRelativeIndexDebugMode:
    """Tests for debug mode output."""

    def test_debug_mode_creation(self, capsys):
        """Debug mode prints info during index creation."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "DEBUG from_variables:" in captured.out
        assert "abs_time_name=abs_time" in captured.out

    def test_debug_mode_sel(self, capsys):
        """Debug mode prints info during sel."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
            debug=True,
        )
        # Clear creation output
        capsys.readouterr()

        ds_indexed.sel(abs_time=1.0, method="nearest")
        captured = capsys.readouterr()
        assert "DEBUG sel:" in captured.out
        assert "DEBUG sel result:" in captured.out

    def test_debug_mode_isel(self, capsys):
        """Debug mode prints info during isel."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.drop_indexes(["trial", "rel_time"]).set_xindex(
            ["abs_time", "trial", "rel_time"],
            AbsoluteRelativeIndex,
            debug=True,
        )
        # Clear creation output
        capsys.readouterr()

        ds_indexed.isel(trial=0)
        captured = capsys.readouterr()
        assert "DEBUG isel:" in captured.out


class TestTrialBasedDatasetModeLinear:
    """Tests for trial_based_dataset with mode='linear'."""

    def test_linear_mode_returns_dataset(self):
        """Linear mode returns an xarray Dataset."""
        ds = trial_based_dataset(mode="linear")
        assert isinstance(ds, xr.Dataset)

    def test_linear_mode_dimensions(self):
        """Linear mode has abs_time as the single dimension."""
        ds = trial_based_dataset(mode="linear")
        assert "abs_time" in ds.dims
        # trial and rel_time should be coords, not dims
        assert "trial" not in ds.dims
        assert "rel_time" not in ds.dims
        # Check total samples: 3 trials * 5s * 100 Hz = 1500
        assert ds.sizes["abs_time"] == 1500

    def test_linear_mode_coords(self):
        """Linear mode has trial and rel_time as 1D coordinates."""
        ds = trial_based_dataset(mode="linear")
        assert "trial" in ds.coords
        assert "rel_time" in ds.coords
        assert "trial_onset" in ds.coords
        # Coords should be 1D with abs_time dimension
        assert ds.coords["trial"].dims == ("abs_time",)
        assert ds.coords["rel_time"].dims == ("abs_time",)

    def test_linear_mode_abs_time_values(self):
        """Linear mode abs_time values are contiguous."""
        ds = trial_based_dataset(
            n_trials=3, trial_length=5.0, sample_rate=10, mode="linear"
        )
        abs_time = ds.coords["abs_time"].values
        # Should go from 0 to near 15 (3 trials * 5s)
        assert abs_time[0] == 0.0
        assert abs_time[-1] < 15.0
        # Should be monotonically increasing
        assert np.all(np.diff(abs_time) > 0)

    def test_linear_mode_trial_coord_values(self):
        """Linear mode trial coord correctly identifies trial for each sample."""
        ds = trial_based_dataset(
            n_trials=3, trial_length=2.0, sample_rate=10, mode="linear"
        )
        # 3 trials * 2s * 10 Hz = 60 samples
        # Each trial has 20 samples
        trial_values = ds.coords["trial"].values
        assert len(trial_values) == 60
        # First 20 should be trial 0
        assert all(t == "cosine" for t in trial_values[:20])
        # Next 20 should be trial 1
        assert all(t == "square" for t in trial_values[20:40])
        # Last 20 should be trial 2
        assert all(t == "sawtooth" for t in trial_values[40:])

    def test_linear_mode_rel_time_coord_values(self):
        """Linear mode rel_time coord resets for each trial."""
        ds = trial_based_dataset(
            n_trials=3, trial_length=2.0, sample_rate=10, mode="linear"
        )
        rel_time = ds.coords["rel_time"].values
        # rel_time should reset at each trial boundary
        # First trial: 0.0, 0.1, ..., 1.9
        assert rel_time[0] == 0.0
        assert rel_time[19] < 2.0
        # Second trial starts again at 0.0
        assert rel_time[20] == 0.0

    def test_linear_mode_data_shape(self):
        """Linear mode data has 1D shape."""
        ds = trial_based_dataset(
            n_trials=3, trial_length=5.0, sample_rate=10, mode="linear"
        )
        assert ds.data.dims == ("abs_time",)
        assert ds.data.shape == (150,)  # 3 * 5 * 10

    def test_linear_mode_waveforms(self):
        """Linear mode preserves waveform patterns."""
        ds = trial_based_dataset(mode="linear")
        # Check that the waveforms are present in the correct order
        # First sample (t=0 of cosine) should be ~1.0
        assert abs(float(ds.data[0]) - 1.0) < 0.01
        # First sample of second trial (square at t=0) should be ~1.0
        n_per_trial = 500  # 5s * 100 Hz
        assert abs(float(ds.data[n_per_trial]) - 1.0) < 0.01
        # First sample of third trial (sawtooth at t=0) should be ~-1.0
        assert abs(float(ds.data[2 * n_per_trial]) - (-1.0)) < 0.01

    def test_linear_mode_custom_labels(self):
        """Linear mode respects custom trial labels."""
        labels = ["trial_A", "trial_B", "trial_C"]
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=2.0,
            sample_rate=10,
            trial_labels=labels,
            mode="linear",
        )
        unique_trials = set(ds.coords["trial"].values)
        assert unique_trials == set(labels)

    def test_invalid_mode_raises(self):
        """Invalid mode parameter raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'stacked' or 'linear'"):
            trial_based_dataset(mode="invalid")


class TestTrialBasedDatasetModeStacked:
    """Tests for trial_based_dataset with mode='stacked' (the default)."""

    def test_stacked_mode_is_default(self):
        """Stacked mode is the default."""
        ds_default = trial_based_dataset()
        ds_stacked = trial_based_dataset(mode="stacked")
        # Both should have same structure
        assert ds_default.dims == ds_stacked.dims
        assert ds_default.sizes == ds_stacked.sizes

    def test_stacked_mode_dimensions(self):
        """Stacked mode has trial and rel_time dimensions."""
        ds = trial_based_dataset(mode="stacked")
        assert "trial" in ds.dims
        assert "rel_time" in ds.dims
        assert "abs_time" not in ds.dims

    def test_stacked_mode_abs_time_is_2d(self):
        """Stacked mode has abs_time as 2D coordinate."""
        ds = trial_based_dataset(mode="stacked")
        assert ds.coords["abs_time"].dims == ("trial", "rel_time")
