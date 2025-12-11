from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from collections import defaultdict

import numpy as np
import pandas as pd

from xarray import Index
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable

__all__ = [
    "DimensionIntervalMulti",
]

# Sentinel for .get() lookups since None is a valid hashable dim name
_MISSING = object()


def merge_sel_results(results: list[IndexSelResult]) -> IndexSelResult:
    dim_indexers = {}
    indexes = {}
    variables = {}
    drop_coords = []
    drop_indexes = []
    rename_dims = {}

    for res in results:
        dim_indexers.update(res.dim_indexers)
        indexes.update(res.indexes)
        variables.update(res.variables)
        drop_coords += res.drop_coords
        drop_indexes += res.drop_indexes
        rename_dims.update(res.rename_dims)

    return IndexSelResult(
        dim_indexers, indexes, variables, drop_coords, drop_indexes, rename_dims
    )


@dataclass
class IntervalDimInfo:
    """Information about one interval dimension.

    Multiple label coordinates are supported for the same interval dimension.
    For example, a "word" dimension could have both "word" labels and
    "part_of_speech" labels that share the same intervals - all non-interval
    coordinates on the dimension are stored in label_indexes.
    """

    dim_name: str  # e.g., "word"
    coord_name: str  # e.g., "word_intervals"
    interval_index: PandasIndex
    label_indexes: dict[str, PandasIndex] = field(
        default_factory=dict
    )  # e.g., {"word": ..., "part_of_speech": ...}


class DimensionIntervalMulti(Index):
    """
    Custom xarray Index supporting multiple interval dimensions over a single
    continuous dimension.

    Example structure:
        Dimensions: (time: 1000, words: 4, phonemes: 20)
        Coordinates:
          * time               (time) int64
          * word_intervals     (words) interval[int64]
          * phoneme_intervals  (phonemes) interval[int64]

    When selecting on any dimension, all other dimensions are automatically
    constrained to overlapping values.

    Options:
        debug (bool): If True, print debug information during operations.
            Pass via set_xindex(..., DimensionIntervalMulti, debug=True)

    Known Limitations:
        - Intervals are assumed to be contiguous (no gaps). Disjoint intervals
          may produce unexpected results when computing time ranges.
        - When multiple indexers conflict (partially overlapping), we take the
          intersection which may result in empty or minimal selections.
        - Array indexers (fancy indexing) are not fully supported for interval
          dimensions - only Integral and slice indexers work.
        - When intersecting intervals with different closed properties (e.g., 'left'
          vs 'both'), the closed property from the most recent interval is used.
    """

    _continuous_index: PandasIndex
    _continuous_name: str
    _interval_dims: dict[str, IntervalDimInfo]  # keyed by dim_name
    _coord_to_dim: dict[str, str]  # maps interval coord name -> dim name
    _label_to_dim: dict[str, str]  # maps label name -> dim name
    _debug: bool

    def __init__(
        self,
        continuous_index: PandasIndex,
        continuous_dim_name: str,
        interval_dims: dict[str, IntervalDimInfo],
        debug: bool = False,
        coord_to_dim: dict[str, str] | None = None,
        label_to_dim: dict[str, str] | None = None,
    ):
        assert isinstance(continuous_index.index, pd.Index)
        for info in interval_dims.values():
            assert isinstance(info.interval_index.index, pd.IntervalIndex)
            for label_idx in info.label_indexes.values():
                assert isinstance(label_idx.index, pd.Index)

        self._continuous_index = continuous_index
        self._continuous_name = continuous_dim_name
        self._interval_dims = interval_dims
        self._debug = debug

        # Use provided lookup maps or build them (O(1) access in sel())
        # These don't change when slicing, so we can reuse them from parent.
        self._coord_to_dim = coord_to_dim or {
            info.coord_name: dim_name for dim_name, info in interval_dims.items()
        }
        self._label_to_dim = label_to_dim or {
            label_name: dim_name
            for dim_name, info in interval_dims.items()
            for label_name in info.label_indexes
        }

    @classmethod
    def from_variables(cls, variables, *, options):
        debug = options.get("debug", False) if options else False

        vars_by_dim: dict[str, list[tuple[str, Variable]]] = defaultdict(list)
        interval_coords: dict[str, str] = {}  # coord_name -> dim_name

        # Group variables by dimension and detect interval coordinates
        for name, var in variables.items():
            assert var.ndim == 1
            dim = var.dims[0]
            vars_by_dim[dim].append((name, var))

            if isinstance(var.dtype, pd.IntervalDtype):
                interval_coords[name] = str(dim)

        dims = list(vars_by_dim.keys())
        if len(dims) < 2:
            raise ValueError(
                f"Expected at least 2 dimensions (1 continuous + 1 interval), got {dims}"
            )
        if len(interval_coords) < 1:
            raise ValueError(
                f"Expected at least 1 interval coordinate, got {interval_coords}"
            )
        if debug:
            print(
                f"DEBUG from_variables:\n"
                f"\tdims={dims}\n"
                f"\tinterval_coords={interval_coords}"
            )

        # Identify continuous dimension (the one without any interval coord)
        interval_dim_names = set(interval_coords.values())
        continuous_dims = [d for d in dims if d not in interval_dim_names]
        if len(continuous_dims) != 1:
            raise ValueError(
                f"Expected exactly one continuous dimension, got {continuous_dims}"
            )
        continuous_dim = continuous_dims[0]

        # Build continuous index
        if len(cont_vars := vars_by_dim[continuous_dim]) != 1:
            raise ValueError(
                f"Expected one coordinate for continuous dimension, got {cont_vars}"
            )
        ((cont_name, cont_var),) = cont_vars
        continuous_index = PandasIndex.from_variables(
            {cont_name: cont_var}, options=options
        )

        # Build IntervalDimInfo for each interval dimension
        interval_dims: dict[str, IntervalDimInfo] = {}
        for coord_name, dim_name in interval_coords.items():
            # Find the interval coord's variable
            dim_vars = vars_by_dim[dim_name]

            interval_index = None
            label_indexes: dict[str, PandasIndex] = {}

            for name, var in dim_vars:
                if name == coord_name:
                    interval_index = PandasIndex.from_variables(
                        {name: var}, options=options
                    )
                else:
                    label_indexes[name] = PandasIndex.from_variables(
                        {name: var}, options=options
                    )

            interval_dims[dim_name] = IntervalDimInfo(
                dim_name=dim_name,
                coord_name=coord_name,
                interval_index=interval_index,
                label_indexes=label_indexes,
            )

        if debug:
            print(
                f"DEBUG from_variables result:\n"
                f"\tcontinuous_dim={continuous_dim}\n"
                f"\tinterval_dims={list(interval_dims.keys())}"
            )

        # TODO: Should we be enforcing contiguousness here or allowing disjoint intervals?
        # Currently we assume intervals are contiguous when computing time ranges.

        return cls(
            continuous_index=continuous_index,
            continuous_dim_name=continuous_dim,
            interval_dims=interval_dims,
            debug=debug,
        )

    def create_variables(self, variables):
        idx_variables = {}

        idx_variables.update(self._continuous_index.create_variables(variables))

        for info in self._interval_dims.values():
            idx_variables.update(info.interval_index.create_variables(variables))
            for label_idx in info.label_indexes.values():
                idx_variables.update(label_idx.create_variables(variables))

        return idx_variables

    @staticmethod
    def _interval_idx_min_max(intervals: pd.IntervalIndex | pd.Interval) -> pd.Interval:
        """Return extremes of an interval or collection of intervals as a single Interval.

        Preserves the closed property from the source intervals.

        TODO: For disjoint intervals, this incorrectly returns the bounding interval
        from the first interval's left to the last interval's right, which includes
        gaps. Should instead return multiple intervals or handle gaps explicitly.
        """
        if isinstance(intervals, pd.IntervalIndex):
            return pd.Interval(
                intervals[0].left, intervals[-1].right, closed=intervals.closed
            )
        elif isinstance(intervals, pd.Interval):
            return intervals
        raise TypeError(f"Expected IntervalIndex or Interval, got {type(intervals)}")

    def _slice_interval_dim(
        self, info: IntervalDimInfo, dim_slice: slice | int
    ) -> IntervalDimInfo:
        """Apply a slice to an interval dimension, including all its label indexes."""
        new_int_index = info.interval_index.isel({info.dim_name: dim_slice})
        # We should always get something back here because of how we constructed our slice.
        # Also makes the typing way easier below.
        assert new_int_index is not None

        new_label_indexes = {}
        for label_name, label_idx in info.label_indexes.items():
            new_label_idx = label_idx.isel({info.dim_name: dim_slice})
            if new_label_idx is not None:
                new_label_indexes[label_name] = new_label_idx

        return IntervalDimInfo(
            dim_name=info.dim_name,
            coord_name=info.coord_name,
            interval_index=new_int_index,
            label_indexes=new_label_indexes,
        )

    def _get_overlapping_slice(
        self,
        interval_index: pd.IntervalIndex,
        time_range: pd.Interval | slice,
    ) -> slice:
        """
        Find the slice of intervals that overlap with the given time range.

        Uses pd.IntervalIndex.overlaps() to find all intervals that have any
        overlap with the specified time range.

        Args:
            interval_index: The IntervalIndex to search for overlaps
            time_range: Either a pd.Interval (with closed property) or a slice
                       (for continuous dimension ranges, treated as closed='both')
        """
        # Convert to pd.Interval if needed, preserving closed property
        if isinstance(time_range, pd.Interval):
            query_interval = time_range
        else:
            # For slices from continuous dimension, use 'both' since we want
            # to include intervals that touch either boundary
            query_interval = pd.Interval(
                time_range.start, time_range.stop, closed="both"
            )

        # Find which intervals overlap
        overlaps = interval_index.overlaps(query_interval)
        overlap_indices = np.where(overlaps)[0]

        if self._debug:
            print(
                f"DEBUG _get_overlapping_slice:\n"
                f"\tquery_interval={query_interval}\n"
                f"\toverlaps={overlap_indices}\n"
                f"\ttarget_closed={interval_index.closed}"
            )

        if len(overlap_indices) == 0:
            # No overlap - return empty slice
            return slice(0, 0)

        return slice(int(overlap_indices[0]), int(overlap_indices[-1]) + 1)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "DimensionIntervalMulti | None":
        # Start with current state
        new_continuous_index = self._continuous_index
        # (shallow copy - we replace entries, not mutate them)
        new_interval_dims = dict(self._interval_dims)

        # Get indexer for continuous dimension
        continuous_indexer = indexers.get(self._continuous_name)

        # Get indexers for each interval dimension
        interval_indexers = {
            dim_name: indexers[dim_name]
            for dim_name in self._interval_dims
            if dim_name in indexers
        }

        if self._debug:
            print(
                f"DEBUG isel:\n\tcontinuous_indexer={continuous_indexer}\n "
                f"\tinterval_indexers={interval_indexers}"
            )

        # Track the time range constraint from indexing operations.
        # NOTE: time_range is over coordinate VALUES (e.g., Interval(17.5, 55.1)),
        # not array indexes. It represents the min/max time values that constrain
        # which intervals overlap. Uses pd.Interval to preserve closed property.
        time_range: pd.Interval | None = None

        # Handle continuous indexer first
        if continuous_indexer is not None:
            # Convert scalar indexers to slices to preserve the dimension.
            # We need to take special care that we don't eliminate a dim here
            # because we can't return from isel that only one of our dims is gone.
            if isinstance(continuous_indexer, Integral):
                cont_slice = slice(continuous_indexer, continuous_indexer + 1)
            elif isinstance(continuous_indexer, np.ndarray):
                # Handle numpy array indexers (from method='nearest')
                if continuous_indexer.ndim == 0:
                    # Scalar array
                    idx = int(continuous_indexer)
                    cont_slice = slice(idx, idx + 1)
                else:
                    # 1D array - for now just use it directly
                    cont_slice = continuous_indexer
            elif isinstance(continuous_indexer, slice):
                cont_slice = continuous_indexer
            else:
                raise NotImplementedError(
                    f"Unsupported continuous indexer type: {type(continuous_indexer)}"
                )

            new_continuous_index = self._continuous_index.isel(
                {self._continuous_name: cont_slice}
            )
            assert new_continuous_index is not None

            # Get the time range for constraining interval dimensions
            # Use closed='both' since continuous values include both endpoints
            time_values = new_continuous_index.index
            time_range = pd.Interval(
                time_values.min(), time_values.max(), closed="both"
            )

            if self._debug:
                print(
                    f"DEBUG isel continuous:\n"
                    f"\ttime_values={time_values}\n"
                    f"\ttime_range={time_range}"
                )

        # Handle interval indexers
        for dim_name, idxr in interval_indexers.items():
            info = self._interval_dims[dim_name]

            # Convert scalar indexers to slices to preserve the dimension.
            if isinstance(idxr, Integral):
                int_slice = slice(idxr, idxr + 1)
            elif isinstance(idxr, slice):
                int_slice = idxr
            else:
                raise NotImplementedError(
                    f"Unsupported interval indexer type: {type(idxr)}"
                )

            # Update this interval dimension
            new_interval_dims[dim_name] = self._slice_interval_dim(info, int_slice)

            # Get time range from selected intervals
            selected_intervals = new_interval_dims[dim_name].interval_index.index
            interval_time_range = self._interval_idx_min_max(selected_intervals)

            # Intersect with existing time_range if any.
            # NOTE: When multiple indexers conflict (partially overlapping), we take
            # the most restrictive approach (intersection). This may result in empty
            # or minimal selections if the indexers don't overlap well.
            if time_range is not None:
                # Intersect intervals, preserving closed from the new interval
                time_range = pd.Interval(
                    max(time_range.left, interval_time_range.left),
                    min(time_range.right, interval_time_range.right),
                    closed=interval_time_range.closed,
                )
            else:
                time_range = interval_time_range

        # Now constrain all dimensions based on the computed time_range
        if time_range is not None:
            # Constrain continuous dimension if it wasn't explicitly indexed
            if continuous_indexer is None:
                # Convert Interval to slice for continuous index sel()
                time_slice = slice(time_range.left, time_range.right)
                cont_sel_result = self._continuous_index.sel(
                    {self._continuous_name: time_slice}
                )
                cont_slice = cont_sel_result.dim_indexers[self._continuous_name]
                new_continuous_index = self._continuous_index.isel(
                    {self._continuous_name: cont_slice}
                )
                assert new_continuous_index is not None

            # Constrain all interval dimensions to overlapping intervals
            for dim_name, info in self._interval_dims.items():
                # Skip if this dimension was explicitly indexed
                if dim_name in interval_indexers:
                    continue

                overlap_slice = self._get_overlapping_slice(
                    info.interval_index.index,
                    time_range,
                )

                if overlap_slice.start == overlap_slice.stop:
                    # No overlap - but we must return something
                    if self._debug:
                        print(f"DEBUG isel: no overlap for {dim_name}")
                    # Keep at least one element
                    overlap_slice = slice(0, 1)

                new_interval_dims[dim_name] = self._slice_interval_dim(
                    info, overlap_slice
                )

        return DimensionIntervalMulti(
            continuous_index=new_continuous_index,
            continuous_dim_name=self._continuous_name,
            interval_dims=new_interval_dims,
            debug=self._debug,
            coord_to_dim=self._coord_to_dim,
            label_to_dim=self._label_to_dim,
        )

    def should_add_coord_to_array(self, name, var, dims) -> bool:
        # TODO: This may not be correct for all cases.
        # This method is passed the name of the DataArray, a single coord (var),
        # and dims of the DataArray (not the coord). We always return True for now.
        return True

    def sel(self, labels, method=None, tolerance=None):
        if self._debug:
            print(f"DEBUG sel:\n\tlabels={labels}\n\tmethod={method}")
        results = []

        # Track time range constraints (pd.Interval to preserve closed property)
        time_range: pd.Interval | None = None

        # cannot only handle this complexity in isel, because if you select on the continuous dimension
        # to a point we can't properly drop the interval coord without passing both through the indexing here
        # e.g. ds.sel(time=10) . we have to pass both indexers from here. we also need to indepdently handle that case
        # inside of isel. Can probably consolidate the code for handling this in the future.
        for key, value in labels.items():
            # Check if selecting on continuous dimension, e.g. ds.sel(time=slice(0, 50))
            if key == self._continuous_name:
                cont_res = self._continuous_index.sel(
                    {key: value}, method=method, tolerance=tolerance
                )
                results.append(cont_res)

                # Get time range from selection (use closed='both' for continuous)
                indexer = cont_res.dim_indexers[self._continuous_name]
                if isinstance(indexer, Integral):
                    time_val = self._continuous_index.index[indexer]
                    time_range = pd.Interval(time_val, time_val, closed="both")
                elif isinstance(indexer, slice):
                    time_vals = self._continuous_index.index[indexer]
                    time_range = pd.Interval(
                        time_vals.min(), time_vals.max(), closed="both"
                    )

            # Check if selecting on an interval coord (e.g. ds.sel(word_intervals=50))
            # or a label coord (e.g. ds.sel(word="red") or ds.sel(part_of_speech="noun"))
            # this `or` will grab the first one that actually works here, never overwriting each
            # other.
            elif (
                dim_name := self._coord_to_dim.get(key, _MISSING)
            ) is not _MISSING or (
                dim_name := self._label_to_dim.get(key, _MISSING)
            ) is not _MISSING:
                info = self._interval_dims[dim_name]
                # Use label index if key is a label, otherwise use interval index
                idx = info.label_indexes.get(key, info.interval_index)
                sel_res = idx.sel({key: value}, method=method, tolerance=tolerance)
                results.append(sel_res)

                # Get time range from selected intervals to constrain other dimensions
                indexer = sel_res.dim_indexers[dim_name]
                selected_intervals = info.interval_index.index[indexer]
                interval_time_range = self._interval_idx_min_max(selected_intervals)

                # Intersect with existing time_range if multiple interval selections
                if time_range is not None:
                    time_range = pd.Interval(
                        max(time_range.left, interval_time_range.left),
                        min(time_range.right, interval_time_range.right),
                        closed=interval_time_range.closed,
                    )
                else:
                    time_range = interval_time_range

        # If we have a time range constraint, add selections for all dimensions
        if time_range is not None:
            # Add continuous constraint if not already selected
            if self._continuous_name not in labels:
                # Convert Interval to slice for continuous index sel()
                time_slice = slice(time_range.left, time_range.right)
                cont_res = self._continuous_index.sel(
                    {self._continuous_name: time_slice},
                    method=method,
                    tolerance=tolerance,
                )
                results.append(cont_res)

            # Identify which interval dimensions were already selected (O(1) lookups)
            selected_dims = set()
            for key in labels:
                if key in self._coord_to_dim:
                    selected_dims.add(self._coord_to_dim[key])
                elif key in self._label_to_dim:
                    selected_dims.add(self._label_to_dim[key])

            # Add constraints for interval dimensions not already selected
            for dim_name, info in self._interval_dims.items():
                if dim_name in selected_dims:
                    continue

                # Find overlapping intervals
                overlap_slice = self._get_overlapping_slice(
                    info.interval_index.index,
                    time_range,
                )

                if overlap_slice.start != overlap_slice.stop:
                    # Create a result for this dimension
                    results.append(IndexSelResult({dim_name: overlap_slice}))

        res = merge_sel_results(results)
        if self._debug:
            print(f"DEBUG sel result:\n\tdim_indexers={res.dim_indexers}")
        return res
