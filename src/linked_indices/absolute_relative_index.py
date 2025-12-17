"""AbsoluteRelativeIndex: Custom xarray Index for trial-based data with absolute/relative time.

This index handles datasets where:
- Data has dimensions (trial, rel_time)
- abs_time is a 2D coordinate mapping (trial, rel_time) -> absolute time
- Selections on abs_time automatically find the correct (trial, rel_time) indices
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from xarray import Index
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable

__all__ = ["AbsoluteRelativeIndex"]


@dataclass
class TrialInfo:
    """Information about trial structure."""

    trial_dim: str
    rel_time_dim: str
    trial_index: PandasIndex
    rel_time_index: PandasIndex
    # 2D array of absolute time values: shape (n_trials, n_rel_time)
    abs_time_values: np.ndarray
    # 1D array of trial onset times
    trial_onsets: np.ndarray


class AbsoluteRelativeIndex(Index):
    """
    Custom xarray Index for trial-based data with both absolute and relative time.

    This index manages datasets where:
    - Data has dimensions (trial, rel_time)
    - `abs_time` is a 2D coordinate of shape (trial, rel_time) representing absolute time
    - `rel_time` is relative time within each trial (e.g., 0 to trial_length)
    - `trial` identifies each trial

    The index enables efficient selection by:
    - `abs_time`: finds which (trial, rel_time) indices correspond to a given absolute time
    - `trial`: selects specific trials, with abs_time automatically constrained
    - `rel_time`: selects relative time points across all trials

    Example
    -------
    >>> import xarray as xr
    >>> from linked_indices import AbsoluteRelativeIndex
    >>> from linked_indices.example_data import trial_based_dataset
    >>> ds = trial_based_dataset(n_trials=3, trial_length=5.0, sample_rate=10)
    >>> ds_indexed = ds.set_xindex(
    ...     ["abs_time", "trial", "rel_time"],
    ...     AbsoluteRelativeIndex,
    ... )
    >>> # Select by absolute time
    >>> ds_indexed.sel(abs_time=7.5, method="nearest")  # In trial 1
    >>> # Select by trial
    >>> ds_indexed.sel(trial="trial_1")  # All of trial 1
    >>> # Select by relative time
    >>> ds_indexed.sel(rel_time=2.5, method="nearest")  # Same rel_time across all trials
    """

    _abs_time_coord: str
    _trial_info: TrialInfo
    _debug: bool

    def __init__(
        self,
        abs_time_coord: str,
        trial_info: TrialInfo,
        debug: bool = False,
    ):
        self._abs_time_coord = abs_time_coord
        self._trial_info = trial_info
        self._debug = debug

    @classmethod
    def from_variables(cls, variables, *, options):
        debug = options.get("debug", False) if options else False

        # Identify which variable is the 2D abs_time coordinate
        abs_time_var = None
        abs_time_name = None
        dim_vars = {}

        for name, var in variables.items():
            if var.ndim == 2:
                # This is the abs_time coordinate
                abs_time_var = var
                abs_time_name = name
            elif var.ndim == 1:
                # This is a dimension coordinate (trial or rel_time)
                dim_vars[var.dims[0]] = (name, var)

        if abs_time_var is None:
            raise ValueError(
                "AbsoluteRelativeIndex requires exactly one 2D coordinate (abs_time). "
                f"Got coordinates: {list(variables.keys())}"
            )

        if len(abs_time_var.dims) != 2:
            raise ValueError(
                f"abs_time coordinate must be 2D, got dims={abs_time_var.dims}"
            )

        # Get the dimension names from abs_time
        trial_dim, rel_time_dim = abs_time_var.dims

        # Check that we have dimension coordinates for both dims
        if trial_dim not in dim_vars:
            raise ValueError(
                f"Missing coordinate for trial dimension '{trial_dim}'. "
                f"Available: {list(dim_vars.keys())}"
            )
        if rel_time_dim not in dim_vars:
            raise ValueError(
                f"Missing coordinate for rel_time dimension '{rel_time_dim}'. "
                f"Available: {list(dim_vars.keys())}"
            )

        trial_name, trial_var = dim_vars[trial_dim]
        rel_time_name, rel_time_var = dim_vars[rel_time_dim]

        # Build PandasIndex for each dimension
        trial_index = PandasIndex.from_variables(
            {trial_name: trial_var}, options=options
        )
        rel_time_index = PandasIndex.from_variables(
            {rel_time_name: rel_time_var}, options=options
        )

        # Extract abs_time values and compute trial onsets
        abs_time_values = abs_time_var.values
        # Trial onset is the first abs_time value in each trial
        trial_onsets = abs_time_values[:, 0]

        trial_info = TrialInfo(
            trial_dim=trial_dim,
            rel_time_dim=rel_time_dim,
            trial_index=trial_index,
            rel_time_index=rel_time_index,
            abs_time_values=abs_time_values,
            trial_onsets=trial_onsets,
        )

        if debug:
            print(
                f"DEBUG from_variables:\n"
                f"\tabs_time_name={abs_time_name}\n"
                f"\ttrial_dim={trial_dim}, rel_time_dim={rel_time_dim}\n"
                f"\tabs_time shape={abs_time_values.shape}\n"
                f"\ttrial_onsets={trial_onsets}"
            )

        return cls(
            abs_time_coord=abs_time_name,
            trial_info=trial_info,
            debug=debug,
        )

    def create_variables(self, variables):
        """Create index variables for the dataset."""
        from xarray.core.variable import Variable as XrVariable

        idx_variables = {}

        # Create variables for trial and rel_time indexes
        idx_variables.update(self._trial_info.trial_index.create_variables(variables))
        idx_variables.update(
            self._trial_info.rel_time_index.create_variables(variables)
        )

        # The abs_time variable is a 2D coordinate that needs to match the current
        # shape of the index (which may have been sliced via isel)
        trial_dim = self._trial_info.trial_dim
        rel_time_dim = self._trial_info.rel_time_dim
        idx_variables[self._abs_time_coord] = XrVariable(
            dims=(trial_dim, rel_time_dim),
            data=self._trial_info.abs_time_values,
        )

        return idx_variables

    def _find_trial_and_rel_time_for_abs_time(
        self, abs_time_value: float
    ) -> tuple[int, int]:
        """Find the (trial_idx, rel_time_idx) for a given absolute time value.

        Returns the indices of the closest point in the 2D abs_time array.
        """
        abs_time_values = self._trial_info.abs_time_values
        trial_onsets = self._trial_info.trial_onsets

        # Find which trial contains this abs_time
        # Trial i contains abs_time if trial_onsets[i] <= abs_time < trial_onsets[i+1]
        trial_idx = np.searchsorted(trial_onsets, abs_time_value, side="right") - 1
        trial_idx = max(0, min(trial_idx, len(trial_onsets) - 1))

        # Within that trial, find the closest rel_time
        trial_abs_times = abs_time_values[trial_idx]
        rel_time_idx = np.argmin(np.abs(trial_abs_times - abs_time_value))

        return int(trial_idx), int(rel_time_idx)

    def _find_trials_for_abs_time_range(
        self, start: float, stop: float
    ) -> tuple[slice, slice]:
        """Find the trial and rel_time slices for an abs_time range.

        Returns (trial_slice, rel_time_slice) that cover the given abs_time range.
        """
        abs_time_values = self._trial_info.abs_time_values
        n_trials, n_rel_time = abs_time_values.shape

        # Find the range of trials that overlap with [start, stop]
        # A trial overlaps if its abs_time range intersects [start, stop]
        trial_start = None
        trial_stop = None

        for i in range(n_trials):
            trial_min = abs_time_values[i].min()
            trial_max = abs_time_values[i].max()

            # Check if this trial overlaps with [start, stop]
            if trial_max >= start and trial_min <= stop:
                if trial_start is None:
                    trial_start = i
                trial_stop = i + 1

        if trial_start is None:
            # No overlap found - return empty slice
            return slice(0, 0), slice(0, 0)

        # For rel_time, we need to find the range that covers [start, stop]
        # across all selected trials
        rel_time_start = n_rel_time
        rel_time_stop = 0

        for trial_idx in range(trial_start, trial_stop):
            trial_abs_times = abs_time_values[trial_idx]
            # Find indices where abs_time is in [start, stop]
            in_range = (trial_abs_times >= start) & (trial_abs_times <= stop)
            indices = np.where(in_range)[0]
            if len(indices) > 0:
                rel_time_start = min(rel_time_start, indices[0])
                rel_time_stop = max(rel_time_stop, indices[-1] + 1)

        if rel_time_start >= rel_time_stop:
            # No valid rel_time range found
            rel_time_start = 0
            rel_time_stop = 0

        return slice(trial_start, trial_stop), slice(rel_time_start, rel_time_stop)

    def sel(self, labels, method=None, tolerance=None):
        """Label-based selection.

        Handles selection on:
        - abs_time: finds (trial, rel_time) indices for the given absolute time
        - trial: selects specific trials
        - rel_time: selects specific relative times
        """
        if self._debug:
            print(f"DEBUG sel: labels={labels}, method={method}")

        trial_dim = self._trial_info.trial_dim
        rel_time_dim = self._trial_info.rel_time_dim
        results = []

        # Handle abs_time selection
        if self._abs_time_coord in labels:
            value = labels[self._abs_time_coord]

            if isinstance(value, slice):
                # Range selection
                start = (
                    value.start
                    if value.start is not None
                    else self._trial_info.abs_time_values.min()
                )
                stop = (
                    value.stop
                    if value.stop is not None
                    else self._trial_info.abs_time_values.max()
                )
                trial_slice, rel_time_slice = self._find_trials_for_abs_time_range(
                    start, stop
                )
                results.append(
                    IndexSelResult(
                        {trial_dim: trial_slice, rel_time_dim: rel_time_slice}
                    )
                )
            else:
                # Scalar selection
                trial_idx, rel_time_idx = self._find_trial_and_rel_time_for_abs_time(
                    float(value)
                )
                # Return slices to preserve dimensions
                results.append(
                    IndexSelResult(
                        {
                            trial_dim: slice(trial_idx, trial_idx + 1),
                            rel_time_dim: slice(rel_time_idx, rel_time_idx + 1),
                        }
                    )
                )

        # Handle trial selection
        trial_coord_name = list(self._trial_info.trial_index.index.names)[0]
        if trial_coord_name in labels:
            value = labels[trial_coord_name]
            trial_result = self._trial_info.trial_index.sel(
                {trial_coord_name: value}, method=method, tolerance=tolerance
            )
            results.append(trial_result)

        # Handle rel_time selection
        rel_time_coord_name = list(self._trial_info.rel_time_index.index.names)[0]
        if rel_time_coord_name in labels:
            value = labels[rel_time_coord_name]
            rel_time_result = self._trial_info.rel_time_index.sel(
                {rel_time_coord_name: value}, method=method, tolerance=tolerance
            )
            results.append(rel_time_result)

        # Merge results
        dim_indexers = {}
        for res in results:
            dim_indexers.update(res.dim_indexers)

        if self._debug:
            print(f"DEBUG sel result: dim_indexers={dim_indexers}")

        return IndexSelResult(dim_indexers)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "AbsoluteRelativeIndex | None":
        """Integer-based selection.

        Updates the index state when dimensions are sliced.
        """
        if self._debug:
            print(f"DEBUG isel: indexers={indexers}")

        trial_dim = self._trial_info.trial_dim
        rel_time_dim = self._trial_info.rel_time_dim

        # Get indexers for each dimension
        trial_indexer = indexers.get(trial_dim)
        rel_time_indexer = indexers.get(rel_time_dim)

        # Start with current state
        new_trial_index = self._trial_info.trial_index
        new_rel_time_index = self._trial_info.rel_time_index
        new_abs_time_values = self._trial_info.abs_time_values
        new_trial_onsets = self._trial_info.trial_onsets

        # Apply trial indexer
        if trial_indexer is not None:
            if isinstance(trial_indexer, Integral):
                trial_slice = slice(trial_indexer, trial_indexer + 1)
            else:
                trial_slice = trial_indexer

            new_trial_index = self._trial_info.trial_index.isel(
                {trial_dim: trial_slice}
            )
            if new_trial_index is None:
                return None
            new_abs_time_values = new_abs_time_values[trial_slice]
            new_trial_onsets = new_trial_onsets[trial_slice]

        # Apply rel_time indexer
        if rel_time_indexer is not None:
            if isinstance(rel_time_indexer, Integral):
                rel_time_slice = slice(rel_time_indexer, rel_time_indexer + 1)
            else:
                rel_time_slice = rel_time_indexer

            new_rel_time_index = self._trial_info.rel_time_index.isel(
                {rel_time_dim: rel_time_slice}
            )
            if new_rel_time_index is None:
                return None
            new_abs_time_values = new_abs_time_values[:, rel_time_slice]

        new_trial_info = TrialInfo(
            trial_dim=trial_dim,
            rel_time_dim=rel_time_dim,
            trial_index=new_trial_index,
            rel_time_index=new_rel_time_index,
            abs_time_values=new_abs_time_values,
            trial_onsets=new_trial_onsets,
        )

        return AbsoluteRelativeIndex(
            abs_time_coord=self._abs_time_coord,
            trial_info=new_trial_info,
            debug=self._debug,
        )

    def should_add_coord_to_array(self, name, var, dims) -> bool:
        """Whether to add a coordinate to a DataArray."""
        return True
