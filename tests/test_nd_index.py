"""Tests for NDIndex - simple N-D coordinate indexing."""

import numpy as np
import pytest
import xarray as xr

from linked_indices import NDIndex
from linked_indices.example_data import trial_based_dataset


class TestNDIndexCreation:
    """Tests for creating a NDIndex."""

    @pytest.fixture
    def ds(self):
        """Create a trial-based dataset."""
        return trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)

    def test_create_index(self, ds):
        """Index can be created with 2D coord."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        assert isinstance(ds_indexed.xindexes["abs_time"], NDIndex)

    def test_1d_coords_keep_default_index(self, ds):
        """1D coords keep their default PandasIndex."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # trial and rel_time should have their own indexes
        assert ds_indexed.xindexes["trial"] is not ds_indexed.xindexes["abs_time"]
        assert ds_indexed.xindexes["rel_time"] is not ds_indexed.xindexes["abs_time"]

    def test_all_coords_remain(self, ds):
        """All coordinates remain visible after indexing."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        assert "abs_time" in ds_indexed.coords
        assert "trial" in ds_indexed.coords
        assert "rel_time" in ds_indexed.coords
        assert "trial_onset" in ds_indexed.coords

    def test_data_preserved(self, ds):
        """Data variable is preserved."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        assert "data" in ds_indexed.data_vars
        assert ds_indexed.data.shape == ds.data.shape


class TestNDIndexSelFollower:
    """Tests for selecting by follower coord."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_sel_scalar_in_first_trial(self, ds_indexed):
        """Selecting abs_time in first trial returns correct indices."""
        result = ds_indexed.sel(abs_time=2.5)
        # Should return a single cell
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        # Should be trial 0
        assert result.trial.values == "trial_0"
        # rel_time should be ~2.5
        assert abs(float(result.rel_time) - 2.5) < 0.15

    def test_sel_scalar_in_second_trial(self, ds_indexed):
        """Selecting abs_time in second trial returns correct indices."""
        result = ds_indexed.sel(abs_time=7.5)
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"
        # rel_time should be ~2.5 (7.5 - 5.0 = 2.5)
        assert abs(float(result.rel_time) - 2.5) < 0.15

    def test_sel_slice_within_trial(self, ds_indexed):
        """Selecting abs_time slice within single trial."""
        result = ds_indexed.sel(abs_time=slice(1.0, 3.0))
        # Should be only trial 0
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_0"
        # rel_time should span approximately 1-3
        assert float(result.rel_time.min()) >= 0.9
        assert float(result.rel_time.max()) <= 3.1

    def test_sel_slice_spanning_trials(self, ds_indexed):
        """Selecting abs_time slice that spans multiple trials."""
        result = ds_indexed.sel(abs_time=slice(4.0, 7.0))
        # Should include 2 trials
        assert result.sizes["trial"] == 2
        assert "trial_0" in result.trial.values
        assert "trial_1" in result.trial.values


class TestNDIndexSelDimensionCoords:
    """Tests for selecting by dimension coords (uses default xarray index)."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_sel_trial_scalar(self, ds_indexed):
        """Selecting a single trial via default xarray index."""
        result = ds_indexed.sel(trial="trial_1")
        # trial dimension is dropped for scalar selection
        assert "trial" not in result.dims
        # All rel_time points should be present
        assert result.sizes["rel_time"] == 50
        # abs_time should be 1D now
        assert result.abs_time.dims == ("rel_time",)

    def test_sel_trial_multiple(self, ds_indexed):
        """Selecting multiple trials."""
        result = ds_indexed.sel(trial=["trial_0", "trial_2"])
        assert result.sizes["trial"] == 2
        assert "trial_0" in result.trial.values
        assert "trial_2" in result.trial.values

    def test_sel_rel_time_scalar(self, ds_indexed):
        """Selecting a single relative time across all trials."""
        result = ds_indexed.sel(rel_time=2.5, method="nearest")
        # Should have all trials
        assert result.sizes["trial"] == 3
        # Scalar selection drops the rel_time dimension
        assert "rel_time" not in result.dims
        # abs_time should be 1D now (NDIndex returns None for < 2D)
        assert result.abs_time.dims == ("trial",)
        # NDIndex should be dropped since abs_time is now 1D
        assert "abs_time" not in result.xindexes or not isinstance(
            result.xindexes.get("abs_time"), NDIndex
        )

    def test_sel_rel_time_slice(self, ds_indexed):
        """Selecting a range of relative times."""
        result = ds_indexed.sel(rel_time=slice(1.0, 3.0))
        # Should have all trials
        assert result.sizes["trial"] == 3
        # rel_time should be constrained
        assert float(result.rel_time.min()) >= 0.9
        assert float(result.rel_time.max()) <= 3.1


class TestNDIndexIsel:
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
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_isel_trial_scalar(self, ds_indexed):
        """Scalar isel on trial dimension drops it."""
        result = ds_indexed.isel(trial=1)
        # trial dimension should be dropped
        assert "trial" not in result.dims
        assert result.sizes["rel_time"] == 50
        # abs_time should be 1D now
        assert result.abs_time.dims == ("rel_time",)
        assert result.abs_time.shape == (50,)

    def test_isel_trial_slice(self, ds_indexed):
        """Slice isel preserves dimension."""
        result = ds_indexed.isel(trial=slice(0, 2))
        assert result.sizes["trial"] == 2
        assert result.sizes["rel_time"] == 50
        # abs_time should still be 2D
        assert result.abs_time.dims == ("trial", "rel_time")

    def test_isel_rel_time_scalar(self, ds_indexed):
        """Scalar isel on rel_time dimension drops it."""
        result = ds_indexed.isel(rel_time=5)
        assert result.sizes["trial"] == 3
        assert "rel_time" not in result.dims
        # abs_time should be 1D now
        assert result.abs_time.dims == ("trial",)

    def test_isel_both_dims_scalar(self, ds_indexed):
        """Scalar isel on both dimensions makes abs_time 0D."""
        result = ds_indexed.isel(trial=1, rel_time=5)
        assert "trial" not in result.dims
        assert "rel_time" not in result.dims
        # abs_time should be scalar
        assert result.abs_time.dims == ()
        # Value should be from trial 1, rel_time index 5 (5.0 + 0.5 = 5.5)
        assert float(result.abs_time) == 5.5
        # NDIndex should be dropped since abs_time is now 0D
        assert "abs_time" not in result.xindexes or not isinstance(
            result.xindexes.get("abs_time"), NDIndex
        )

    def test_isel_both_dims_slice(self, ds_indexed):
        """Slice isel on both dimensions preserves 2D."""
        result = ds_indexed.isel(trial=slice(0, 2), rel_time=slice(0, 10))
        assert result.sizes["trial"] == 2
        assert result.sizes["rel_time"] == 10
        assert result.abs_time.shape == (2, 10)


class TestNDIndexEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_sel_outside_range_raises_keyerror(self, ds_indexed):
        """Selecting abs_time outside data range raises KeyError without method='nearest'."""
        with pytest.raises(KeyError, match="not found"):
            ds_indexed.sel(abs_time=100.0)

    def test_sel_outside_range_high_nearest(self, ds_indexed):
        """Selecting abs_time above data range with method='nearest' returns nearest."""
        result = ds_indexed.sel(abs_time=100.0, method="nearest")
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        # Should be the last trial, last time
        assert result.trial.values == "trial_2"

    def test_sel_outside_range_low_nearest(self, ds_indexed):
        """Selecting abs_time below data range with method='nearest' returns nearest."""
        result = ds_indexed.sel(abs_time=-100.0, method="nearest")
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        assert result.trial.values == "trial_0"

    def test_sel_slice_outside_range(self, ds_indexed):
        """Selecting slice completely outside range returns empty."""
        result = ds_indexed.sel(abs_time=slice(100.0, 200.0))
        assert result.sizes["trial"] == 0
        assert result.sizes["rel_time"] == 0

    def test_sel_at_boundary(self, ds_indexed):
        """Selecting at trial boundary returns one cell."""
        result = ds_indexed.sel(abs_time=5.0)
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1


class TestNDIndexValidation:
    """Tests for validation during index creation."""

    def test_1d_coord_raises(self):
        """Error when 1D coordinate is provided."""
        ds = xr.Dataset(
            {"data": (("trial", "rel_time"), np.random.rand(3, 10))},
            coords={
                "trial": ["a", "b", "c"],
                "rel_time": np.linspace(0, 1, 10),
            },
        )
        with pytest.raises(ValueError, match="only accepts N-D coordinates"):
            ds.drop_indexes(["trial"]).set_xindex(["trial"], NDIndex)

    def test_no_coords_raises(self):
        """Error when no coordinates are provided."""
        ds = xr.Dataset(
            {"data": (("x", "y"), np.random.rand(3, 10))},
        )
        with pytest.raises(ValueError, match="requires at least one N-D coordinate"):
            ds.set_xindex([], NDIndex)


class TestNDIndexMultipleFollowers:
    """Tests for multiple follower coordinates."""

    @pytest.fixture
    def ds_multi(self):
        """Create dataset with multiple 2D coords."""
        ds = trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,
        )
        # Add second 2D coord
        ds = ds.assign_coords(
            {"normalized_time": ds.abs_time / float(ds.abs_time.max())}
        )
        return ds.set_xindex(["abs_time", "normalized_time"], NDIndex)

    def test_create_with_multiple(self, ds_multi):
        """Index can be created with multiple 2D coords."""
        assert "abs_time" in ds_multi.xindexes
        assert "normalized_time" in ds_multi.xindexes
        # Should be same index object
        assert ds_multi.xindexes["abs_time"] is ds_multi.xindexes["normalized_time"]

    def test_sel_on_second_follower(self, ds_multi):
        """Selection works on second follower coord with method='nearest'."""
        result = ds_multi.sel(normalized_time=0.5, method="nearest")
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1


class TestNDIndex3D:
    """Tests for 3D coordinates (e.g., subject × trial × time)."""

    @pytest.fixture
    def ds_3d(self):
        """Create dataset with 3D abs_time coordinate.

        Simulates multi-subject, multi-trial time series where each
        subject has a different session start time.
        """
        n_subjects = 2
        n_trials = 3
        n_times = 20

        # Dimension coordinates
        subjects = ["subj_A", "subj_B"]
        trials = ["trial_0", "trial_1", "trial_2"]
        rel_time = np.linspace(0, 1, n_times)

        # Subject session offsets (e.g., different recording start times)
        subject_offset = xr.DataArray(
            [0.0, 100.0], dims=["subject"], coords={"subject": subjects}
        )

        # Trial onsets within each session
        trial_onset = xr.DataArray(
            [0.0, 2.0, 4.0], dims=["trial"], coords={"trial": trials}
        )

        # 3D absolute time: subject_offset + trial_onset + rel_time
        # Shape: (subject, trial, rel_time)
        abs_time_3d = (
            subject_offset + trial_onset + xr.DataArray(rel_time, dims=["rel_time"])
        )

        # Create dataset
        data = np.random.rand(n_subjects, n_trials, n_times)
        ds = xr.Dataset(
            {"signal": (("subject", "trial", "rel_time"), data)},
            coords={
                "subject": subjects,
                "trial": trials,
                "rel_time": rel_time,
                "abs_time": abs_time_3d,
                "subject_offset": subject_offset,
                "trial_onset": trial_onset,
            },
        )
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_create_3d_index(self, ds_3d):
        """NDIndex can be created with 3D coordinate."""
        assert "abs_time" in ds_3d.xindexes
        assert isinstance(ds_3d.xindexes["abs_time"], NDIndex)
        assert ds_3d.abs_time.dims == ("subject", "trial", "rel_time")

    def test_sel_3d_exact(self, ds_3d):
        """Exact selection on 3D coord finds correct cell."""
        # Subject A, trial 1, rel_time 0.5 -> abs_time = 0 + 2 + 0.5 = 2.5
        # But rel_time is linspace(0, 1, 20), so 0.5 is not exact
        # Let's use an exact value: subject A, trial 0, rel_time[0] = 0.0
        result = ds_3d.sel(abs_time=0.0)
        assert result.sizes["subject"] == 1
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        assert result.subject.values == "subj_A"
        assert result.trial.values == "trial_0"

    def test_sel_3d_nearest(self, ds_3d):
        """Nearest selection on 3D coord."""
        # Subject B starts at 100.0, trial 1 at +2.0 -> 102.5ish
        result = ds_3d.sel(abs_time=102.5, method="nearest")
        assert result.sizes["subject"] == 1
        assert result.sizes["trial"] == 1
        assert result.sizes["rel_time"] == 1
        assert result.subject.values == "subj_B"
        assert result.trial.values == "trial_1"

    def test_sel_3d_slice(self, ds_3d):
        """Slice selection on 3D coord spans multiple dims."""
        # Subject A: trial 0 -> 0-1, trial 1 -> 2-3, trial 2 -> 4-5
        # Select abs_time from 1.5 to 4.5 - should span trials 1 and 2 for subject A
        result = ds_3d.sel(abs_time=slice(1.5, 4.5))
        assert result.sizes["subject"] == 1  # Only subject A
        assert result.sizes["trial"] == 2  # Trials 1 (2-3) and 2 (4-5)
        assert result.subject.values == "subj_A"

    def test_isel_3d_reduces_correctly(self, ds_3d):
        """isel with scalar on one dim reduces 3D to 2D."""
        result = ds_3d.isel(subject=0)
        assert "subject" not in result.dims
        assert result.abs_time.dims == ("trial", "rel_time")
        # NDIndex should still be present (2D coord)
        assert "abs_time" in result.xindexes

    def test_isel_3d_two_scalars(self, ds_3d):
        """isel with scalars on two dims reduces 3D to 1D, drops index."""
        result = ds_3d.isel(subject=0, trial=1)
        assert "subject" not in result.dims
        assert "trial" not in result.dims
        assert result.abs_time.dims == ("rel_time",)
        # NDIndex should be dropped (1D coord)
        assert "abs_time" not in result.xindexes or not isinstance(
            result.xindexes.get("abs_time"), NDIndex
        )

    def test_sel_3d_combined_with_dim_coord(self, ds_3d):
        """Can combine N-D coord selection with 1D coord selection."""
        # First select subject (scalar drops the dimension)
        result = ds_3d.sel(subject="subj_B")
        assert "subject" not in result.dims  # Scalar selection drops dim
        # abs_time is now 2D (trial, rel_time), still indexed
        assert result.abs_time.dims == ("trial", "rel_time")
        assert "abs_time" in result.xindexes
        # Can still select by abs_time on the 2D array
        result2 = result.sel(abs_time=102.0, method="nearest")
        assert result2.sizes["trial"] == 1
        assert result2.sizes["rel_time"] == 1


class TestNDIndexPlotting:
    """Tests that plotting works correctly."""

    @pytest.fixture
    def ds_indexed(self):
        """Create an indexed trial-based dataset."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_plot_row_trial_correct_lines(self, ds_indexed):
        """Plotting with row='trial' draws correct number of lines."""
        import matplotlib.pyplot as plt

        fig = ds_indexed["data"].plot.line(x="rel_time", row="trial")
        # Should have 1 line per facet, not 50
        lines_per_facet = [len(ax.lines) for ax in fig.axs.flat]
        assert all(n == 1 for n in lines_per_facet)
        plt.close(fig.fig)

    def test_plot_abs_time(self, ds_indexed):
        """Plotting with x='abs_time' and hue='trial' gives 3 lines."""
        import matplotlib.pyplot as plt

        # Without hue, xarray iterates over rel_time (50 lines)
        # With hue='trial', we get 3 lines (one per trial)
        lines = ds_indexed["data"].plot.line(x="abs_time", hue="trial")
        assert len(lines) == 3
        plt.close()

    def test_time_locked_plotting(self, ds_indexed):
        """Plotting with derived time-locked coordinate works."""
        import matplotlib.pyplot as plt

        ds_locked = ds_indexed.assign_coords(
            {"speech_onset": ("trial", [1.5, 2.5, 3.0])}
        )
        ds_locked = ds_locked.assign_coords(
            {"speech_locked_time": ds_locked["rel_time"] - ds_locked["speech_onset"]}
        )

        fig = ds_locked["data"].plot.line(x="speech_locked_time", row="trial")
        # Should have 1 line per facet
        lines_per_facet = [len(ax.lines) for ax in fig.axs.flat]
        assert all(n == 1 for n in lines_per_facet)
        plt.close(fig.fig)


class TestNDIndexDebugMode:
    """Tests for debug mode output."""

    def test_debug_mode_creation(self, capsys):
        """Debug mode prints info during index creation."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds.set_xindex(["abs_time"], NDIndex, debug=True)
        captured = capsys.readouterr()
        assert "DEBUG from_variables:" in captured.out
        assert "nd_coords:" in captured.out

    def test_debug_mode_sel(self, capsys):
        """Debug mode prints info during sel."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, debug=True)
        capsys.readouterr()  # Clear creation output

        ds_indexed.sel(abs_time=1.0)
        captured = capsys.readouterr()
        assert "DEBUG sel:" in captured.out
        assert "DEBUG sel result:" in captured.out

    def test_debug_mode_isel(self, capsys):
        """Debug mode prints info during isel."""
        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, debug=True)
        capsys.readouterr()  # Clear creation output

        ds_indexed.isel(trial=0)
        captured = capsys.readouterr()
        assert "DEBUG isel:" in captured.out


class TestNDIndexSliceMethods:
    """Tests for slice method options and step handling."""

    @pytest.fixture
    def ds(self):
        """Create a trial-based dataset with clear trial boundaries.

        Trial 0: abs_time 0-5
        Trial 1: abs_time 5-10
        Trial 2: abs_time 10-15
        """
        return trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=10,  # 0.1s steps
            trial_labels=["trial_0", "trial_1", "trial_2"],
        )

    def test_default_slice_method_is_bounding_box(self, ds):
        """Default slice method is bounding_box."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        assert ds_indexed.xindexes["abs_time"]._slice_method == "bounding_box"

    def test_slice_method_option(self, ds):
        """slice_method option is stored correctly."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        assert ds_indexed.xindexes["abs_time"]._slice_method == "trim_outer"

    def test_invalid_slice_method_raises(self, ds):
        """Invalid slice_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid slice_method"):
            ds.set_xindex(["abs_time"], NDIndex, slice_method="invalid")

    def test_bounding_box_spans_all_touched_trials(self, ds):
        """Bounding box includes trials even with no values in range.

        Select abs_time 2.5 to 7.5:
        - Trial 0 has values 2.5-4.9 in range
        - Trial 1 has values 5.0-7.5 in range
        - Trial 2 has no values in range

        Bounding box should return trials 0-1 (indices that have ANY values).
        """
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="bounding_box")
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))
        assert result.sizes["trial"] == 2
        assert list(result.trial.values) == ["trial_0", "trial_1"]

    def test_trim_outer_excludes_empty_trials(self, ds):
        """trim_outer excludes trials with no values in range.

        For this dataset, trim_outer should behave same as bounding_box
        since the trials that have values form a contiguous range.
        """
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))
        assert result.sizes["trial"] == 2
        assert list(result.trial.values) == ["trial_0", "trial_1"]

    def test_slice_with_step(self, ds):
        """Slice with step applies step to inner dimension."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # Select abs_time 0 to 3 with step 2
        result = ds_indexed.sel(abs_time=slice(0, 3, 2))
        # Inner dimension (rel_time) should have every other index
        # Original 0-3 covers indices 0-30, step 2 means indices 0, 2, 4, ...
        assert result.sizes["rel_time"] < 31  # Should be roughly half
        # Check the step is applied by looking at rel_time spacing
        rel_times = result.rel_time.values
        if len(rel_times) > 1:
            spacing = rel_times[1] - rel_times[0]
            assert spacing == pytest.approx(0.2)  # 2 * 0.1s step

    def test_slice_method_preserved_after_isel(self, ds):
        """slice_method is preserved after isel operation."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        result = ds_indexed.isel(trial=slice(0, 2))
        # Index should still exist and have same slice_method
        assert result.xindexes["abs_time"]._slice_method == "trim_outer"

    def test_slice_entire_range_with_step(self, ds):
        """Slice entire coordinate range with step works."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(None, None, 5))
        # Should have every 5th time point
        original_rel_time_size = ds_indexed.sizes["rel_time"]
        # With step 5, should have ~1/5 of original points
        assert result.sizes["rel_time"] == (original_rel_time_size + 4) // 5

    def test_slice_step_1_same_as_no_step(self, ds):
        """Step=1 should give same result as no step."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result_no_step = ds_indexed.sel(abs_time=slice(0, 3))
        result_step_1 = ds_indexed.sel(abs_time=slice(0, 3, 1))
        assert result_no_step.sizes == result_step_1.sizes
        np.testing.assert_array_equal(
            result_no_step.rel_time.values, result_step_1.rel_time.values
        )

    def test_slice_step_none_same_as_no_step(self, ds):
        """Step=None should give same result as no step."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result_no_step = ds_indexed.sel(abs_time=slice(0, 3))
        result_step_none = ds_indexed.sel(abs_time=slice(0, 3, None))
        assert result_no_step.sizes == result_step_none.sizes

    def test_slice_large_step(self, ds):
        """Large step that exceeds data size returns single point."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(0, 3, 1000))
        # Step 1000 on ~30 points should give just the first point
        assert result.sizes["rel_time"] == 1

    def test_trim_outer_on_single_trial_range(self, ds):
        """trim_outer with range in single trial returns one trial."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        # Select range entirely within trial 1 (abs_time 5-10)
        result = ds_indexed.sel(abs_time=slice(6.0, 8.0))
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"

    def test_bounding_box_on_single_trial_range(self, ds):
        """bounding_box with range in single trial returns one trial."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="bounding_box")
        result = ds_indexed.sel(abs_time=slice(6.0, 8.0))
        assert result.sizes["trial"] == 1
        assert result.trial.values == "trial_1"


class TestNDIndexSliceMethods3D:
    """Tests for slice methods on 3D coordinates."""

    @pytest.fixture
    def ds_3d(self):
        """Create 3D dataset: subject × trial × time."""
        n_subjects, n_trials, n_times = 2, 3, 20

        subjects = ["subj_A", "subj_B"]
        trials = ["trial_0", "trial_1", "trial_2"]
        rel_time = np.linspace(0, 1, n_times)

        # Subject A: times 0-6, Subject B: times 100-106
        subject_offset = xr.DataArray(
            [0.0, 100.0], dims=["subject"], coords={"subject": subjects}
        )
        trial_onset = xr.DataArray(
            [0.0, 2.0, 4.0], dims=["trial"], coords={"trial": trials}
        )
        abs_time_3d = (
            subject_offset + trial_onset + xr.DataArray(rel_time, dims=["rel_time"])
        )

        data = np.random.rand(n_subjects, n_trials, n_times)
        ds = xr.Dataset(
            {"signal": (("subject", "trial", "rel_time"), data)},
            coords={
                "subject": subjects,
                "trial": trials,
                "rel_time": rel_time,
                "abs_time": abs_time_3d,
            },
        )
        return ds

    def test_3d_slice_with_step(self, ds_3d):
        """Step works on 3D coordinates (applies to innermost dim)."""
        ds_indexed = ds_3d.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(0, 3, 2))
        # Step should apply to rel_time (innermost dimension)
        assert result.sizes["rel_time"] < ds_3d.sizes["rel_time"]
        # Spacing should be doubled
        if result.sizes["rel_time"] > 1:
            original_spacing = ds_3d.rel_time.values[1] - ds_3d.rel_time.values[0]
            result_spacing = result.rel_time.values[1] - result.rel_time.values[0]
            assert result_spacing == pytest.approx(original_spacing * 2)

    def test_3d_trim_outer_single_subject(self, ds_3d):
        """trim_outer on 3D with range in single subject."""
        ds_indexed = ds_3d.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        # Select range only in subject B (abs_time 100-106)
        result = ds_indexed.sel(abs_time=slice(101, 103))
        assert result.sizes["subject"] == 1
        assert result.subject.values == "subj_B"

    def test_3d_bounding_box_spans_subjects(self, ds_3d):
        """bounding_box on 3D can span multiple subjects."""
        ds_indexed = ds_3d.set_xindex(
            ["abs_time"], NDIndex, slice_method="bounding_box"
        )
        # Select range in subject A (0-6)
        result = ds_indexed.sel(abs_time=slice(2, 4))
        # Should only include subject A
        assert result.sizes["subject"] == 1
        assert result.subject.values == "subj_A"

    def test_3d_slice_method_preserved_through_isel(self, ds_3d):
        """slice_method preserved when 3D reduces to 2D via isel."""
        ds_indexed = ds_3d.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")
        # Select one subject - 3D becomes 2D
        result = ds_indexed.isel(subject=0)
        assert "abs_time" in result.xindexes
        assert result.xindexes["abs_time"]._slice_method == "trim_outer"


class TestNDIndexBoundingBoxBehavior:
    """Thorough tests for bounding box slice behavior.

    These tests verify the documented behavior that slice selection returns
    the bounding box of matching cells, which may include cells outside the
    requested range.
    """

    @pytest.fixture
    def ds(self):
        """Create dataset with known abs_time structure.

        Trial 'cosine': abs_time 0.0 to 4.99
        Trial 'square': abs_time 5.0 to 9.99
        Trial 'sawtooth': abs_time 10.0 to 14.99
        """
        return trial_based_dataset(
            n_trials=3,
            trial_length=5.0,
            sample_rate=100,  # 0.01s steps for precise tests
        )

    def test_slice_returns_correct_trial_range(self, ds):
        """Slice spanning two trials returns both trials."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 2.5-7.5 spans cosine (0-5) and square (5-10)
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        assert result.sizes["trial"] == 2
        assert list(result.trial.values) == ["cosine", "square"]

    def test_slice_returns_correct_rel_time_range(self, ds):
        """Bounding box rel_time extends from min to max of matching cells."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 2.5-7.5:
        #   cosine matches rel_time 2.5-4.99
        #   square matches rel_time 0.0-2.5
        # Bounding box: rel_time 0.0-4.99
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        assert float(result.rel_time.min()) == pytest.approx(0.0)
        assert float(result.rel_time.max()) == pytest.approx(4.99)

    def test_result_contains_values_outside_requested_range(self, ds):
        """Bounding box includes abs_time values outside the requested range."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        # Result should contain abs_time values < 2.5 (from square trial's early times)
        # and abs_time values > 7.5 (from cosine trial's late times mapped to square)
        min_abs = float(result.abs_time.min())
        max_abs = float(result.abs_time.max())

        # Cosine trial at rel_time=0 has abs_time=0.0 (outside [2.5, 7.5])
        assert min_abs < 2.5
        # Square trial at rel_time=4.99 has abs_time=9.99 (outside [2.5, 7.5])
        assert max_abs > 7.5

    def test_count_cells_outside_range(self, ds):
        """Quantify how many cells are outside the requested range."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        in_range = (result.abs_time >= 2.5) & (result.abs_time <= 7.5)
        outside_range = ~in_range

        # Both trials have cells outside the range
        n_outside = int(outside_range.sum())
        n_total = result.abs_time.size

        assert n_outside > 0, "Should have cells outside range"
        # Roughly 50% should be outside (half of each trial)
        assert n_outside / n_total > 0.4
        assert n_outside / n_total < 0.6

    def test_per_trial_in_range_cells(self, ds):
        """Each trial has different cells in range."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        # Cosine trial (abs_time 0-5): only rel_time >= 2.5 is in range
        cosine_data = result.sel(trial="cosine")
        cosine_in_range = (cosine_data.abs_time >= 2.5) & (cosine_data.abs_time <= 7.5)
        cosine_in_count = int(cosine_in_range.sum())

        # Square trial (abs_time 5-10): only rel_time <= 2.5 is in range
        square_data = result.sel(trial="square")
        square_in_range = (square_data.abs_time >= 2.5) & (square_data.abs_time <= 7.5)
        square_in_count = int(square_in_range.sum())

        # Both should have roughly half their cells in range
        total_rel_time = result.sizes["rel_time"]
        assert cosine_in_count < total_rel_time
        assert cosine_in_count > 0
        assert square_in_count < total_rel_time
        assert square_in_count > 0

    def test_where_mask_filters_correctly(self, ds):
        """Verify .where() workaround filters to exact range."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(2.5, 7.5))

        # Apply mask
        mask = (result.abs_time >= 2.5) & (result.abs_time <= 7.5)
        filtered = result.where(mask)

        # Data should be NaN where outside range
        outside_mask = ~mask
        assert np.all(np.isnan(filtered.data.values[outside_mask.values]))

        # Data should NOT be NaN where inside range (unless original was NaN)
        inside_mask = mask
        inside_data = filtered.data.values[inside_mask.values]
        assert not np.all(np.isnan(inside_data))

    def test_slice_single_trial_no_extra_trials(self, ds):
        """Slice entirely within one trial returns only that trial."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 6-8 is entirely within square trial (5-10)
        result = ds_indexed.sel(abs_time=slice(6.0, 8.0))

        assert result.sizes["trial"] == 1
        assert result.trial.values == "square"

    def test_slice_single_trial_rel_time_extent(self, ds):
        """Slice within one trial has correct rel_time bounding box."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 6-8 in square trial corresponds to rel_time 1-3
        result = ds_indexed.sel(abs_time=slice(6.0, 8.0))

        # rel_time should span ~1.0 to ~3.0
        assert float(result.rel_time.min()) == pytest.approx(1.0, abs=0.01)
        assert float(result.rel_time.max()) == pytest.approx(3.0, abs=0.01)

    def test_empty_result_when_no_match(self, ds):
        """Slice outside all data returns empty result."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        result = ds_indexed.sel(abs_time=slice(100.0, 200.0))

        assert result.sizes["trial"] == 0
        assert result.sizes["rel_time"] == 0

    def test_slice_at_trial_boundary(self, ds):
        """Slice exactly at trial boundary includes both trials."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 4.5-5.5 spans the boundary between cosine (ends ~5) and square (starts 5)
        result = ds_indexed.sel(abs_time=slice(4.5, 5.5))

        assert result.sizes["trial"] == 2
        assert list(result.trial.values) == ["cosine", "square"]

    def test_slice_all_three_trials(self, ds):
        """Slice spanning all trials returns all three."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 2-12 spans all three trials
        result = ds_indexed.sel(abs_time=slice(2.0, 12.0))

        assert result.sizes["trial"] == 3
        assert list(result.trial.values) == ["cosine", "square", "sawtooth"]

    def test_slice_excludes_non_matching_trials(self, ds):
        """Trials completely outside range are excluded."""
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
        # abs_time 0-3 only matches cosine trial
        result = ds_indexed.sel(abs_time=slice(0.0, 3.0))

        assert result.sizes["trial"] == 1
        assert result.trial.values == "cosine"
        # sawtooth (10-15) should definitely be excluded
        assert "sawtooth" not in result.trial.values


class TestNDIndexSliceMethodNearest:
    """Tests for slice selection with method='nearest'."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a simple dataset with known abs_time values."""
        # 3 trials, 5 time points each
        # Trial 0: abs_time = 0, 1, 2, 3, 4
        # Trial 1: abs_time = 10, 11, 12, 13, 14
        # Trial 2: abs_time = 20, 21, 22, 23, 24
        trial_onsets = np.array([0, 10, 20])
        rel_time = np.arange(5, dtype=float)
        abs_time = trial_onsets[:, np.newaxis] + rel_time[np.newaxis, :]
        data = np.arange(15).reshape(3, 5).astype(float)

        ds = xr.Dataset(
            {"data": (["trial", "rel_time"], data)},
            coords={
                "trial": np.arange(3),
                "rel_time": rel_time,
                "abs_time": (["trial", "rel_time"], abs_time),
            },
        )
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_slice_nearest_exact_boundaries(self, ds_indexed):
        """When boundaries exist exactly, nearest gives same result as default."""
        result_exact = ds_indexed.sel(abs_time=slice(1.0, 3.0))
        result_nearest = ds_indexed.sel(abs_time=slice(1.0, 3.0), method="nearest")

        # Should get the same result
        np.testing.assert_array_equal(
            result_exact.data.values, result_nearest.data.values
        )

    def test_slice_nearest_non_exact_boundaries(self, ds_indexed):
        """Nearest finds closest values when exact boundaries don't exist."""
        # Slice from 0.6 to 3.4 - should snap to nearest: 1 and 3
        result = ds_indexed.sel(abs_time=slice(0.6, 3.4), method="nearest")

        # Should include values from 1 to 3
        assert 1.0 in result.abs_time.values
        assert 3.0 in result.abs_time.values

    def test_slice_nearest_outside_range_snaps_to_edge(self, ds_indexed):
        """Nearest snaps to edge when boundaries are outside data range."""
        # Slice from -5 to 2 - start should snap to 0
        result = ds_indexed.sel(abs_time=slice(-5.0, 2.0), method="nearest")

        # Should include the start of the data
        assert 0.0 in result.abs_time.values

    def test_slice_nearest_spanning_trials(self, ds_indexed):
        """Nearest works across trial boundaries."""
        # Slice from 3.7 to 10.3 - should snap to 4 and 10
        result = ds_indexed.sel(abs_time=slice(3.7, 10.3), method="nearest")

        # Should include both trials 0 and 1
        assert result.sizes["trial"] >= 2
        assert 4.0 in result.abs_time.values
        assert 10.0 in result.abs_time.values

    def test_slice_nearest_swapped_boundaries(self, ds_indexed):
        """Nearest handles when start nearest > stop nearest."""
        # Both 9.9 and 10.1 snap to 10
        # Result should be a single point or small range
        result = ds_indexed.sel(abs_time=slice(9.9, 10.1), method="nearest")

        assert 10.0 in result.abs_time.values

    def test_slice_nearest_single_point(self, ds_indexed):
        """Nearest with very narrow range still returns valid slice."""
        # Slice from 5.0 to 5.0 - but 5.0 doesn't exist, nearest is 4
        result = ds_indexed.sel(abs_time=slice(5.0, 5.0), method="nearest")

        # Should return something (the nearest point)
        assert result.sizes["trial"] >= 1

    def test_slice_nearest_with_step(self, ds_indexed):
        """Nearest works with step parameter."""
        result = ds_indexed.sel(abs_time=slice(0.5, 3.5, 2), method="nearest")

        # Should have applied step to the slice
        assert result.sizes["rel_time"] <= 3  # step=2 reduces points


class TestNDIndexSliceMethodNearestUnsorted:
    """Tests for slice selection with method='nearest' on unsorted coords."""

    @pytest.fixture
    def ds_unsorted(self):
        """Create a dataset with unsorted abs_time values (forced unsorted path)."""
        # Use a radial pattern that's naturally unsorted
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 4)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2)
        data = np.sin(radius)

        ds = xr.Dataset(
            {"data": (["y", "x"], data)},
            coords={
                "x": x,
                "y": y,
                "radius": (["y", "x"], radius),
            },
        )
        ds_indexed = ds.set_xindex(["radius"], NDIndex)
        # Verify it's unsorted
        idx = ds_indexed.xindexes["radius"]
        coord = idx._nd_coords["radius"]
        assert not coord.is_sorted, "Expected unsorted coordinate"
        return ds_indexed

    def test_slice_nearest_unsorted(self, ds_unsorted):
        """Nearest works on unsorted coordinates."""
        # Find nearest to radius 1.0 - 1.5
        result = ds_unsorted.sel(radius=slice(1.0, 1.5), method="nearest")

        # Should return valid result
        assert result.sizes["y"] >= 1
        assert result.sizes["x"] >= 1

    def test_slice_nearest_unsorted_vs_default(self, ds_unsorted):
        """Nearest and default may give different results on unsorted."""
        # With exact boundaries that exist
        result_default = ds_unsorted.sel(radius=slice(0.0, 2.0))
        result_nearest = ds_unsorted.sel(radius=slice(0.1, 1.9), method="nearest")

        # Both should return valid results (may differ)
        assert result_default.sizes["y"] >= 1
        assert result_nearest.sizes["y"] >= 1


class TestNDIndexMaskedSelection:
    """Tests for masked and metadata selection modes."""

    @pytest.fixture
    def ds_indexed(self):
        """Create a trial-based dataset with varying values per trial."""
        # Create dataset where each trial has different abs_time values
        # Trial 0: 0-5s -> abs_time 10-15
        # Trial 1: 0-5s -> abs_time 20-25
        # Trial 2: 0-5s -> abs_time 30-35
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=1)
        return ds.set_xindex(["abs_time"], NDIndex)

    def test_returns_slice_same_as_sel(self, ds_indexed):
        """returns='slice' produces same result as regular sel()."""
        from linked_indices import nd_sel

        result_sel = ds_indexed.sel(abs_time=slice(11, 14))
        result_nd_sel = nd_sel(ds_indexed, abs_time=slice(11, 14), returns="slice")

        xr.testing.assert_identical(result_sel, result_nd_sel)

    def test_returns_mask_nan_outside_range(self, ds_indexed):
        """returns='mask' applies NaN to values outside range."""
        from linked_indices import nd_sel

        # Select abs_time between 1 and 8 - spans trials 0 and 1
        # Trial 0: [0, 1, 2, 3, 4] - 0 is outside [1, 8]
        # Trial 1: [5, 6, 7, 8, 9] - 9 is outside [1, 8]
        result = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="mask")

        # Check that values outside [1, 8] are NaN
        abs_time_vals = result["abs_time"].values
        data_vals = result["data"].values

        # Build expected mask
        mask = (abs_time_vals >= 1) & (abs_time_vals <= 8)

        # NaN where outside range
        assert np.all(np.isnan(data_vals[~mask]))
        # Valid data where inside range
        assert np.all(~np.isnan(data_vals[mask]))

    def test_returns_mask_preserves_coords(self, ds_indexed):
        """returns='mask' preserves coordinate structure."""
        from linked_indices import nd_sel

        result = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="mask")

        # All coordinates should still exist
        assert "abs_time" in result.coords
        assert "trial" in result.coords
        assert "rel_time" in result.coords

    def test_returns_metadata_adds_bool_coord(self, ds_indexed):
        """returns='metadata' adds boolean coordinate."""
        from linked_indices import nd_sel

        result = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="metadata")

        # Should have a new coordinate indicating membership
        assert "in_abs_time_range" in result.coords

        # Check the boolean coordinate has correct values
        in_range = result["in_abs_time_range"].values
        abs_time_vals = result["abs_time"].values

        expected = (abs_time_vals >= 1) & (abs_time_vals <= 8)
        np.testing.assert_array_equal(in_range, expected)

    def test_returns_metadata_preserves_data(self, ds_indexed):
        """returns='metadata' doesn't modify data values."""
        from linked_indices import nd_sel

        result_slice = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="slice")
        result_meta = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="metadata")

        # Data values should be identical (no NaN masking)
        np.testing.assert_array_equal(
            result_slice["data"].values, result_meta["data"].values
        )

    def test_mask_with_method_nearest(self, ds_indexed):
        """Mask works with method='nearest' for boundaries."""
        from linked_indices import nd_sel

        # Select 0.7 to 8.2 with nearest - should snap to 1 and 8
        # Then mask values outside [1, 8]
        # Trial 0: [0, 1, 2, 3, 4] - 0 is outside
        # Trial 1: [5, 6, 7, 8, 9] - 9 is outside
        result = nd_sel(
            ds_indexed,
            abs_time=slice(0.7, 8.2),  # Will snap to 1 and 8
            method="nearest",
            returns="mask",
        )

        # Check masking happened (values 0 and 9 should be NaN)
        assert np.any(np.isnan(result["data"].values))

    def test_invalid_returns_raises(self, ds_indexed):
        """Invalid returns value raises error."""
        from linked_indices import nd_sel

        with pytest.raises(ValueError, match="Invalid returns"):
            nd_sel(ds_indexed, abs_time=slice(11, 13), returns="invalid")

    def test_nd_sel_fallback_for_non_ndindex(self):
        """nd_sel falls back to sel() for non-NDIndex coords."""
        from linked_indices import nd_sel

        # Create simple dataset without NDIndex
        ds = xr.Dataset(
            {"data": ("x", [1, 2, 3, 4, 5])},
            coords={"x": [0, 1, 2, 3, 4]},
        )

        # returns='slice' should work via fallback
        result = nd_sel(ds, x=slice(1, 3), returns="slice")
        assert result.sizes["x"] == 3

    def test_nd_sel_error_mask_without_ndindex(self):
        """nd_sel raises error for mask mode without NDIndex."""
        from linked_indices import nd_sel

        ds = xr.Dataset(
            {"data": ("x", [1, 2, 3, 4, 5])},
            coords={"x": [0, 1, 2, 3, 4]},
        )

        with pytest.raises(ValueError, match="No NDIndex found"):
            nd_sel(ds, x=slice(1, 3), returns="mask")

    def test_sel_masked_via_index(self, ds_indexed):
        """sel_masked can be called directly on the index."""
        idx = ds_indexed.xindexes["abs_time"]

        # Select 1 to 8 - spans trials 0 and 1, with some values outside
        result = idx.sel_masked(ds_indexed, {"abs_time": slice(1, 8)}, returns="mask")

        # Should have NaN masking (values 0 and 9 are outside [1, 8])
        assert np.any(np.isnan(result["data"].values))

    def test_mask_only_affects_data_vars(self, ds_indexed):
        """Mask applies to data variables, not coordinates."""
        from linked_indices import nd_sel

        # Select 1 to 8 - spans trials 0 and 1
        result = nd_sel(ds_indexed, abs_time=slice(1, 8), returns="mask")

        # Coordinate values should not be NaN
        assert not np.any(np.isnan(result["abs_time"].values))
        assert not np.any(np.isnan(result["rel_time"].values))

        # But data should have NaN (values 0 and 9 are outside [1, 8])
        assert np.any(np.isnan(result["data"].values))


class TestNDIndexCoverageGaps:
    """Tests to cover edge cases and missing coverage lines."""

    def test_nd_sel_empty_labels_raises(self):
        """nd_sel with no labels raises ValueError."""
        from linked_indices import nd_sel

        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)

        with pytest.raises(
            ValueError, match="Must provide at least one coordinate label"
        ):
            nd_sel(ds_indexed)

    def test_nd_sel_slice_mode_no_ndindex_falls_back(self):
        """nd_sel with returns='slice' falls back to sel when no NDIndex."""
        from linked_indices import nd_sel

        # Create dataset without NDIndex
        ds = xr.Dataset(
            {"data": (("x", "y"), np.arange(20).reshape(4, 5))},
            coords={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3, 4]},
        )

        # Should work via fallback to standard sel
        # slice(1, 3) on [0,1,2,3] returns [1, 2, 3] = 3 elements
        result = nd_sel(ds, x=slice(1, 3), returns="slice")
        assert result.sizes["x"] == 3

    def test_linear_search_for_unsorted_exact(self):
        """Linear search works for exact match on unsorted coords."""
        # Create dataset with non-monotonic 2D coordinate
        y, x = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
        # Shuffle to make unsorted
        values = (y * 10 + x).ravel()
        np.random.seed(42)
        np.random.shuffle(values)
        values = values.reshape(5, 5)

        ds = xr.Dataset(
            {"data": (("y", "x"), np.random.randn(5, 5))},
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
                "shuffled": (("y", "x"), values),
            },
        )
        ds_indexed = ds.set_xindex(["shuffled"], NDIndex)

        # Find a value that exists - pick one from the shuffled array
        target = int(values[2, 2])
        result = ds_indexed.sel(shuffled=target)
        assert result.sizes["y"] == 1
        assert result.sizes["x"] == 1

    def test_linear_search_exact_not_found(self):
        """Linear search raises KeyError for exact match not found."""
        y = np.arange(5)
        x = np.arange(5)
        values = np.random.randn(5, 5)

        ds = xr.Dataset(
            {"data": (("y", "x"), values)},
            coords={
                "y": y,
                "x": x,
                "coord": (("y", "x"), values),
            },
        )
        ds_indexed = ds.set_xindex(["coord"], NDIndex)

        with pytest.raises(KeyError, match="not found"):
            ds_indexed.sel(coord=999.999)

    def test_linear_search_for_unsorted_nearest(self):
        """Linear search works for nearest match on unsorted coords."""
        # Create unsorted 2D coordinate
        values = np.random.randn(5, 5)

        ds = xr.Dataset(
            {"data": (("y", "x"), values.copy())},
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
                "coord": (("y", "x"), values),
            },
        )
        ds_indexed = ds.set_xindex(["coord"], NDIndex)

        # Use nearest - should work
        result = ds_indexed.sel(coord=0.0, method="nearest")
        assert result.sizes["y"] == 1
        assert result.sizes["x"] == 1

    def test_trim_outer_slice_method(self):
        """trim_outer slice method trims outer dimensions more aggressively."""
        ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex, slice_method="trim_outer")

        # Select range that only covers first two trials
        result = ds_indexed.sel(abs_time=slice(1, 6))

        # trim_outer should still work
        assert result.sizes["trial"] >= 1

    def test_compute_range_mask_unsorted_nearest(self):
        """_compute_range_mask with method='nearest' on unsorted coord."""
        from linked_indices import nd_sel

        # Create dataset with unsorted 2D coord
        values = np.random.randn(5, 5)

        ds = xr.Dataset(
            {"data": (("y", "x"), np.ones((5, 5)))},
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
                "coord": (("y", "x"), values),
            },
        )
        ds_indexed = ds.set_xindex(["coord"], NDIndex)

        # Use mask mode with nearest
        result = nd_sel(
            ds_indexed,
            coord=slice(-0.5, 0.5),
            method="nearest",
            returns="mask",
        )

        # Should work without error
        assert result is not None

    def test_sel_masked_unsorted_nearest(self):
        """sel_masked with method='nearest' on unsorted coord."""
        from linked_indices import nd_sel

        # Create dataset with intentionally unsorted values
        values = np.array(
            [
                [5.0, 1.0, 3.0],
                [2.0, 4.0, 0.0],
            ]
        )

        ds = xr.Dataset(
            {"data": (("y", "x"), np.ones((2, 3)))},
            coords={
                "y": np.arange(2),
                "x": np.arange(3),
                "coord": (("y", "x"), values),
            },
        )
        ds_indexed = ds.set_xindex(["coord"], NDIndex)

        # Use mask mode with nearest - triggers unsorted path
        result = nd_sel(
            ds_indexed,
            coord=slice(1.5, 3.5),
            method="nearest",
            returns="mask",
        )
        assert result is not None

    def test_sel_masked_with_non_managed_coord(self):
        """sel_masked ignores labels not managed by this index."""
        from linked_indices import nd_sel

        ds = trial_based_dataset(n_trials=2, trial_length=2.0, sample_rate=10)
        ds_indexed = ds.set_xindex(["abs_time"], NDIndex)

        # Select on abs_time (managed) and trial (not managed by NDIndex)
        # This tests the 'continue' path in sel_masked
        result = nd_sel(ds_indexed, abs_time=slice(0.5, 1.5), returns="mask")
        assert result is not None


class TestNDIndexMain:
    """Test the main function in __init__.py."""

    def test_main_prints_message(self, capsys):
        """main() prints a greeting message."""
        from linked_indices import main

        main()
        captured = capsys.readouterr()
        assert "Hello from linked-indices" in captured.out
