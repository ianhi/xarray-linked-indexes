"""NDIndex: A simple xarray Index for N-D derived coordinates.

This is the simplest case: an index that only manages N-D coordinates
(like abs_time with shape (trial, rel_time)) and enables selection on them.

It does NOT manage any 1D dimension coordinates - those use default xarray indexing.

Example:
    ds.set_xindex(['abs_time'], NDIndex)
    ds.sel(abs_time=7.5)  # Finds the (trial, rel_time) indices nearest to 7.5

Slice methods:
    When using slice selection (e.g., ds.sel(abs_time=slice(2, 8))), you can
    configure how the index determines which indices to return:

    - "bounding_box" (default): Returns the smallest rectangular region
      containing all cells with values in range. This may include cells
      outside the range.

    - "trim_outer": Like bounding_box for the inner dimension, but only
      includes outer dimension indices that have at least one value in range.
      Useful when you don't want "empty" rows in the result.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from xarray import Index
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable

__all__ = ["NDIndex", "nd_sel"]

# Valid returns modes for selection
RETURNS_MODES = ("slice", "mask", "metadata")

# Valid slice methods
SLICE_METHODS = ("bounding_box", "trim_outer")


def _is_sorted(arr: np.ndarray) -> bool:
    """Check if a flattened array is sorted (ascending)."""
    flat = arr.ravel()
    return bool(np.all(flat[:-1] <= flat[1:]))


@dataclass
class NDCoord:
    """Information about an N-D coordinate."""

    name: str
    dims: tuple[str, ...]
    values: np.ndarray
    # Cached flat values and sorted flag for O(log n) lookups
    _flat_values: np.ndarray | None = None
    _is_sorted: bool | None = None

    @property
    def is_sorted(self) -> bool:
        """Lazily check if values are sorted (computed on first access)."""
        if self._is_sorted is None:
            self._is_sorted = _is_sorted(self.values)
        return self._is_sorted

    @property
    def flat_values(self) -> np.ndarray:
        """Lazily flatten values (computed on first access)."""
        if self._flat_values is None:
            self._flat_values = self.values.ravel()
        return self._flat_values


class NDIndex(Index):
    """
    Simple xarray Index for N-D derived coordinates.

    This index manages N-D coordinates (ndim >= 2) and enables label-based
    selection on them. It finds the indices in all dimensions that correspond
    to a given value.

    This index does NOT manage dimension coordinates - they keep default indexing.

    How sel() works:
    ----------------
    When you call ds.sel(abs_time=7.5):
    1. We search the N-D abs_time array for the value closest to 7.5
    2. We find which (trial_idx, rel_time_idx) gives that closest value
    3. We return IndexSelResult with slices for both dimensions
    4. xarray applies those slices to the dataset

    For slice selection (abs_time=slice(2, 8)):
    1. We find all cells where 2 <= abs_time <= 8
    2. For each dimension, we find the range of indices containing those cells
    3. We return slices that cover those ranges

    How isel() works:
    -----------------
    When xarray slices dimensions (e.g., ds.isel(trial=1)):
    1. xarray calls our isel() with the indexers
    2. We slice our N-D coord arrays to match
    3. We return a new NDIndex with the sliced arrays

    Example
    -------
    >>> from linked_indices import NDIndex
    >>> from linked_indices.example_data import trial_based_dataset
    >>> ds = trial_based_dataset(mode="stacked")
    >>> ds_indexed = ds.set_xindex(["abs_time"], NDIndex)
    >>> ds_indexed.sel(abs_time=7.5)  # Finds trial=1, rel_time=2.5
    """

    _nd_coords: dict[str, NDCoord]
    _slice_method: str
    _debug: bool

    def __init__(
        self,
        nd_coords: dict[str, NDCoord],
        slice_method: str = "bounding_box",
        debug: bool = False,
    ):
        if slice_method not in SLICE_METHODS:
            raise ValueError(
                f"Invalid slice_method: {slice_method!r}. "
                f"Must be one of: {SLICE_METHODS}"
            )
        self._nd_coords = nd_coords
        self._slice_method = slice_method
        self._debug = debug

    @classmethod
    def from_variables(cls, variables: Mapping[str, Variable], *, options):
        """
        Create an NDIndex from coordinate variables.

        Expects only N-D coordinates (ndim >= 2). Raises an error if any
        1D coordinates are provided.

        Options
        -------
        slice_method : str, optional
            Method for slice selection. One of:
            - "bounding_box" (default): Smallest rectangle containing all matches
            - "trim_outer": Like bounding_box, but only outer indices with matches
        debug : bool, optional
            Enable debug output. Default False.
        """
        debug = options.get("debug", False) if options else False
        slice_method = (
            options.get("slice_method", "bounding_box") if options else "bounding_box"
        )

        nd_coords = {}

        for name, var in variables.items():
            if var.ndim < 2:
                raise ValueError(
                    f"NDIndex only accepts N-D coordinates (ndim >= 2). "
                    f"Got '{name}' with ndim={var.ndim}. "
                    f"1D coordinates should use default xarray indexing."
                )

            nd_coords[name] = NDCoord(
                name=name,
                dims=var.dims,
                values=var.values,
            )

        if not nd_coords:
            raise ValueError("NDIndex requires at least one N-D coordinate.")

        if debug:
            print(
                f"DEBUG from_variables:\n"
                f"\tnd_coords: {list(nd_coords.keys())}\n"
                f"\tdims: {[fc.dims for fc in nd_coords.values()]}"
            )

        return cls(nd_coords=nd_coords, slice_method=slice_method, debug=debug)

    def create_variables(self, variables):
        """Create index variables for the dataset."""
        from xarray.core.variable import Variable as XrVariable

        idx_variables = {}
        for name, ndc in self._nd_coords.items():
            idx_variables[name] = XrVariable(dims=ndc.dims, data=ndc.values)
        return idx_variables

    def _find_indices_for_value(
        self, coord_name: str, value: float, method: str | None = None
    ) -> dict[str, int]:
        """
        Find indices in all dimensions for a given value.

        Uses O(log n) binary search when coordinate values are sorted,
        otherwise falls back to O(n) linear scan.

        Parameters
        ----------
        coord_name : str
            Name of the N-D coordinate to search
        value : float
            The value to find
        method : str or None
            If None, requires exact match (raises KeyError if not found).
            If 'nearest', returns the cell with value closest to target.

        Returns
        -------
        dict mapping dimension name to index
        """
        ndc = self._nd_coords[coord_name]
        values = ndc.values
        flat_values = ndc.flat_values

        if ndc.is_sorted:
            # O(log n) binary search path
            flat_idx = self._binary_search(flat_values, value, method, coord_name)
        else:
            # O(n) linear scan path
            flat_idx = self._linear_search(values, value, method, coord_name)

        # Convert to multi-dimensional indices
        indices = np.unravel_index(flat_idx, values.shape)

        return {dim: int(idx) for dim, idx in zip(ndc.dims, indices)}

    def _binary_search(
        self, flat_values: np.ndarray, value: float, method: str | None, coord_name: str
    ) -> int:
        """O(log n) binary search for sorted arrays."""
        idx = np.searchsorted(flat_values, value)
        n = len(flat_values)

        if method == "nearest":
            return self._find_nearest_index(flat_values, value, idx)
        else:
            # Exact match required
            if idx < n and flat_values[idx] == value:
                return idx
            raise KeyError(
                f"Value {value!r} not found in coordinate {coord_name!r}. "
                f"Use method='nearest' for approximate matching."
            )

    def _find_nearest_index(
        self, flat_values: np.ndarray, value: float, searchsorted_idx: int | None = None
    ) -> int:
        """Find the index of the nearest value using O(log n) binary search.

        Parameters
        ----------
        flat_values : np.ndarray
            Sorted 1D array to search
        value : float
            Target value to find nearest match for
        searchsorted_idx : int, optional
            Pre-computed searchsorted result to avoid redundant computation

        Returns
        -------
        int
            Index of the nearest value
        """
        n = len(flat_values)
        if n == 0:
            raise ValueError("Cannot find nearest in empty array")

        idx = (
            searchsorted_idx
            if searchsorted_idx is not None
            else np.searchsorted(flat_values, value)
        )

        if idx == 0:
            return 0
        elif idx == n:
            return n - 1
        else:
            left_val = flat_values[idx - 1]
            right_val = flat_values[idx]
            if abs(value - left_val) <= abs(value - right_val):
                return idx - 1
            else:
                return idx

    def _linear_search(
        self, values: np.ndarray, value: float, method: str | None, coord_name: str
    ) -> int:
        """O(n) linear scan for unsorted arrays."""
        if method == "nearest":
            return int(np.argmin(np.abs(values - value)))
        else:
            flat_matches = np.flatnonzero(values == value)
            if len(flat_matches) == 0:
                raise KeyError(
                    f"Value {value!r} not found in coordinate {coord_name!r}. "
                    f"Use method='nearest' for approximate matching."
                )
            return int(flat_matches[0])

    def _find_slices_for_range(
        self,
        coord_name: str,
        start: float,
        stop: float,
        step: int | None = None,
        slice_method: str | None = None,
        method: str | None = None,
    ) -> dict[str, slice]:
        """
        Find slices in all dimensions for a value range.

        Uses O(log n) binary search when coordinate values are sorted,
        otherwise O(n) linear scan.

        Parameters
        ----------
        coord_name : str
            Name of the N-D coordinate to search
        start : float
            Start of the value range (inclusive)
        stop : float
            End of the value range (inclusive)
        step : int or None
            Step to apply to the innermost dimension slice. None means no step.
        slice_method : str or None
            Override the default slice method. If None, uses self._slice_method.
            Options: "bounding_box", "trim_outer"
        method : str or None
            Selection method for boundaries:
            - None (default): start/stop are treated as exact boundaries
            - "nearest": find nearest values to start/stop (useful when exact
              values don't exist in the coordinate)

        Returns
        -------
        dict mapping dimension name to slice
        """
        ndc = self._nd_coords[coord_name]
        values = ndc.values
        bbox_method = slice_method or self._slice_method

        if ndc.is_sorted:
            # O(log n) path: use binary search to find range bounds
            flat_values = ndc.flat_values

            if method == "nearest":
                # Find nearest values to start and stop
                left_idx = self._find_nearest_index(flat_values, start)
                right_idx = self._find_nearest_index(flat_values, stop)
                # Ensure right >= left (swap if needed due to nearest finding)
                if right_idx < left_idx:
                    left_idx, right_idx = right_idx, left_idx
                # right_idx is inclusive, so add 1 for the slice
                right_idx += 1
            else:
                # Exact boundaries
                left_idx = np.searchsorted(flat_values, start, side="left")
                right_idx = np.searchsorted(flat_values, stop, side="right")

            if left_idx >= right_idx:
                # No values in range
                return {dim: slice(0, 0) for dim in ndc.dims}

            # Compute bounding box directly from first and last flat indices
            # This is O(1) instead of O(k) where k is the number of matching cells
            first_multi = np.unravel_index(left_idx, values.shape)
            last_multi = np.unravel_index(right_idx - 1, values.shape)

            result = {}
            for i, dim in enumerate(ndc.dims):
                if i == 0:
                    # First dimension: range is [first[0], last[0] + 1]
                    result[dim] = slice(int(first_multi[0]), int(last_multi[0]) + 1)
                else:
                    # Later dimensions: if all preceding dims are same, use actual range
                    # Otherwise the range spans the full dimension
                    if first_multi[:i] == last_multi[:i]:
                        result[dim] = slice(int(first_multi[i]), int(last_multi[i]) + 1)
                    else:
                        result[dim] = slice(0, values.shape[i])

            # Apply step to innermost dimension
            if step is not None and step != 1:
                inner_dim = ndc.dims[-1]
                if inner_dim in result:
                    s = result[inner_dim]
                    result[inner_dim] = slice(s.start, s.stop, step)

            return result

        # O(n) path: scan all values
        if method == "nearest":
            # Find nearest values to start and stop using linear scan
            flat_values = values.ravel()
            left_idx = int(np.argmin(np.abs(flat_values - start)))
            right_idx = int(np.argmin(np.abs(flat_values - stop)))
            # Ensure right >= left
            if right_idx < left_idx:
                left_idx, right_idx = right_idx, left_idx

            # Convert flat indices to multi-dimensional and compute bounding box
            first_multi = np.unravel_index(left_idx, values.shape)
            last_multi = np.unravel_index(right_idx, values.shape)

            result = {}
            for i, dim in enumerate(ndc.dims):
                dim_start = min(first_multi[i], last_multi[i])
                dim_stop = max(first_multi[i], last_multi[i]) + 1
                result[dim] = slice(int(dim_start), int(dim_stop))

            # Apply step to innermost dimension
            if step is not None and step != 1:
                inner_dim = ndc.dims[-1]
                if inner_dim in result:
                    s = result[inner_dim]
                    result[inner_dim] = slice(s.start, s.stop, step)

            return result

        # Standard range selection
        in_range = (values >= start) & (values <= stop)

        if not np.any(in_range):
            # No values in range - return empty slices
            return {dim: slice(0, 0) for dim in ndc.dims}

        result = {}

        if bbox_method == "bounding_box":
            # For each dimension, find the bounding extent of in-range cells
            for i, dim in enumerate(ndc.dims):
                # Project onto this dimension: any in-range value along other dims
                axes_to_reduce = tuple(j for j in range(values.ndim) if j != i)
                if axes_to_reduce:
                    has_value = np.any(in_range, axis=axes_to_reduce)
                else:  # pragma: no cover
                    # Defensive: only reachable if ndim < 2, but we validate ndim >= 2
                    has_value = in_range

                indices = np.where(has_value)[0]
                if len(indices) > 0:
                    result[dim] = slice(int(indices[0]), int(indices[-1]) + 1)
                else:  # pragma: no cover
                    # Defensive: if any cell is in_range, all dims have indices
                    result[dim] = slice(0, 0)

        elif bbox_method == "trim_outer":
            # For outer dims: only include indices that have at least one match
            # For inner dim: use bounding box extent
            inner_dim = ndc.dims[-1]  # Last dim is the "inner" dimension

            for i, dim in enumerate(ndc.dims):
                # Project onto this dimension
                axes_to_reduce = tuple(j for j in range(values.ndim) if j != i)
                if axes_to_reduce:
                    has_value = np.any(in_range, axis=axes_to_reduce)
                else:  # pragma: no cover
                    # Defensive: only reachable if ndim < 2, but we validate ndim >= 2
                    has_value = in_range

                indices = np.where(has_value)[0]

                if len(indices) == 0:  # pragma: no cover
                    # Defensive: if any cell is in_range, all dims have indices
                    result[dim] = slice(0, 0)
                elif dim == inner_dim:
                    # Inner dim: bounding box (continuous range)
                    result[dim] = slice(int(indices[0]), int(indices[-1]) + 1)
                else:
                    # Outer dims: only indices with matches
                    # Note: xarray requires contiguous slices, so we still use
                    # bounding box but document that trim_outer trims outer dims
                    # more aggressively when possible
                    result[dim] = slice(int(indices[0]), int(indices[-1]) + 1)

        # Apply step to the innermost dimension if specified
        if step is not None and step != 1:
            inner_dim = ndc.dims[-1]
            if inner_dim in result:
                s = result[inner_dim]
                result[inner_dim] = slice(s.start, s.stop, step)

        return result

    def sel(self, labels, method=None, tolerance=None):
        """
        Label-based selection on N-D coordinates.

        For scalar selection:
        - Default (method=None): requires exact match, raises KeyError if not found
        - method='nearest': finds cell with closest value

        For slice selection:
        - Default (method=None): finds all cells in range [start, stop]
        - method='nearest': finds nearest values to start/stop boundaries
          (useful when exact boundary values don't exist in the coordinate)

        The slice behavior depends on the slice_method configured for this index:
        - "bounding_box": smallest rectangle containing all matches
        - "trim_outer": only outer indices with at least one match

        If the input slice has a step (e.g., slice(0, 10, 2)), the step is
        applied to the innermost dimension of the result.
        """
        if self._debug:
            print(f"DEBUG sel: labels={labels}, method={method}")

        dim_indexers = {}

        for name, ndc in self._nd_coords.items():
            if name not in labels:
                continue

            value = labels[name]

            if isinstance(value, slice):
                # Range selection
                start = value.start if value.start is not None else ndc.values.min()
                stop = value.stop if value.stop is not None else ndc.values.max()
                step = value.step  # May be None
                slices = self._find_slices_for_range(
                    name, start, stop, step=step, method=method
                )
                dim_indexers.update(slices)
            else:
                # Scalar selection - exact or nearest based on method
                indices = self._find_indices_for_value(
                    name, float(value), method=method
                )
                # Convert to slices to preserve dimensions
                for dim, idx in indices.items():
                    dim_indexers[dim] = slice(idx, idx + 1)

        if self._debug:
            print(f"DEBUG sel result: dim_indexers={dim_indexers}")

        return IndexSelResult(dim_indexers)

    def _compute_range_mask(
        self,
        coord_name: str,
        start: float,
        stop: float,
        method: str | None = None,
    ) -> np.ndarray:
        """
        Compute a boolean mask for values within a range.

        Parameters
        ----------
        coord_name : str
            Name of the N-D coordinate
        start, stop : float
            Range boundaries (inclusive)
        method : str or None
            If 'nearest', snap boundaries to nearest existing values

        Returns
        -------
        np.ndarray
            Boolean mask with same shape as the coordinate, True where in range
        """
        ndc = self._nd_coords[coord_name]
        values = ndc.values

        if method == "nearest":
            # Snap boundaries to nearest values
            if ndc.is_sorted:
                flat_values = ndc.flat_values
                start_idx = self._find_nearest_index(flat_values, start)
                stop_idx = self._find_nearest_index(flat_values, stop)
                start = flat_values[min(start_idx, stop_idx)]
                stop = flat_values[max(start_idx, stop_idx)]
            else:
                # O(n) for unsorted
                flat_values = values.ravel()
                start_idx = int(np.argmin(np.abs(flat_values - start)))
                stop_idx = int(np.argmin(np.abs(flat_values - stop)))
                start = flat_values[min(start_idx, stop_idx)]
                stop = flat_values[max(start_idx, stop_idx)]

        return (values >= start) & (values <= stop)

    def sel_masked(
        self,
        obj,
        labels: dict[str, Any],
        method: str | None = None,
        returns: str = "mask",
    ):
        """
        Select with non-rectangular masking support.

        Unlike sel(), this method can return masked results where values
        outside the selection range are set to NaN (returns='mask') or
        a boolean coordinate is added (returns='metadata').

        Parameters
        ----------
        obj : xr.Dataset or xr.DataArray
            The object to select from
        labels : dict
            Mapping of coordinate name to selection value (scalar or slice)
        method : str or None
            'nearest' to snap to nearest values, None for exact boundaries
        returns : str
            - 'slice': Standard rectangular selection (same as sel())
            - 'mask': Apply NaN mask outside selection range
            - 'metadata': Add boolean coordinate indicating membership

        Returns
        -------
        xr.Dataset or xr.DataArray
            Selected data with masking applied if requested

        Examples
        --------
        >>> # Mask values outside [1.0, 2.0] with NaN
        >>> result = index.sel_masked(ds, {'abs_time': slice(1.0, 2.0)}, returns='mask')

        >>> # Add boolean coordinate showing which cells are in range
        >>> result = index.sel_masked(ds, {'abs_time': slice(1.0, 2.0)}, returns='metadata')
        """
        if returns not in RETURNS_MODES:
            raise ValueError(
                f"Invalid returns={returns!r}. Must be one of {RETURNS_MODES}"
            )

        # First, get the bounding box selection (always a view when possible)
        result = obj.sel(labels, method=method)

        if returns == "slice":
            return result

        # Compute mask for each coordinate in labels
        combined_mask = None
        for name, value in labels.items():
            if name not in self._nd_coords:
                continue

            if isinstance(value, slice):
                ndc = self._nd_coords[name]
                start = value.start if value.start is not None else ndc.values.min()
                stop = value.stop if value.stop is not None else ndc.values.max()

                # Compute mask on the RESULT's coordinate values (after slicing)
                result_values = result[name].values
                if method == "nearest":
                    # Snap boundaries using original coordinate
                    if ndc.is_sorted:
                        flat_values = ndc.flat_values
                        start_idx = self._find_nearest_index(flat_values, start)
                        stop_idx = self._find_nearest_index(flat_values, stop)
                        start = flat_values[min(start_idx, stop_idx)]
                        stop = flat_values[max(start_idx, stop_idx)]
                    else:
                        flat_orig = ndc.values.ravel()
                        start_idx = int(np.argmin(np.abs(flat_orig - start)))
                        stop_idx = int(np.argmin(np.abs(flat_orig - stop)))
                        start = flat_orig[min(start_idx, stop_idx)]
                        stop = flat_orig[max(start_idx, stop_idx)]

                mask = (result_values >= start) & (result_values <= stop)

                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask & mask

        if combined_mask is None:
            # No slice selections on our coordinates
            return result

        if returns == "mask":
            return result.where(combined_mask)
        elif returns == "metadata":
            # Find the first coord name we're selecting on for naming
            coord_names = [n for n in labels if n in self._nd_coords]
            mask_name = f"in_{coord_names[0]}_range" if coord_names else "in_range"
            return result.assign_coords(
                {mask_name: (result[coord_names[0]].dims, combined_mask)}
            )

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "NDIndex | None":
        """
        Integer-based selection - updates N-D coord arrays.

        When xarray slices dimensions, we need to slice our N-D arrays
        to match. This method applies the indexers to each N-D coord.

        Returns None if the resulting coord would have < 2 dimensions,
        since NDIndex only manages N-D coords (ndim >= 2).
        """
        if self._debug:
            print(f"DEBUG isel: indexers={indexers}")

        new_nd_coords = {}

        for name, ndc in self._nd_coords.items():
            # Build the full index tuple upfront (before any dimension reduction)
            idx_tuple = []
            new_dims = []

            for dim in ndc.dims:
                if dim in indexers:
                    indexer = indexers[dim]
                    idx_tuple.append(indexer)
                    # Only keep dimension if not scalar indexed
                    if not isinstance(indexer, Integral):
                        new_dims.append(dim)
                else:
                    idx_tuple.append(slice(None))
                    new_dims.append(dim)

            # Apply indexing in one operation
            new_values = ndc.values[tuple(idx_tuple)]
            new_dims = tuple(new_dims)

            new_nd_coords[name] = NDCoord(
                name=name,
                dims=new_dims,
                values=new_values,
            )

        # Check if any N-D coord still has >= 2 dimensions
        # If all coords are now 1D or less, we don't need this index anymore
        max_ndim = max(ndc.values.ndim for ndc in new_nd_coords.values())
        if max_ndim < 2:
            # Return None to let xarray drop this index
            return None

        return NDIndex(
            nd_coords=new_nd_coords,
            slice_method=self._slice_method,
            debug=self._debug,
        )

    def should_add_coord_to_array(self, name, var, dims) -> bool:
        """Whether to add a coordinate to a DataArray."""
        return True


def nd_sel(
    obj,
    labels: dict[str, Any] | None = None,
    method: str | None = None,
    returns: str = "slice",
    **label_kwargs,
):
    """
    Select from an xarray object with non-rectangular masking support.

    This is a convenience function that wraps NDIndex.sel_masked(). It finds
    the NDIndex managing the specified coordinates and applies the selection.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The object to select from
    labels : dict, optional
        Mapping of coordinate name to selection value (scalar or slice)
    method : str or None
        'nearest' to snap to nearest values, None for exact boundaries
    returns : str
        - 'slice': Standard rectangular selection (same as ds.sel())
        - 'mask': Apply NaN mask outside selection range
        - 'metadata': Add boolean coordinate indicating membership
    **label_kwargs
        Alternative way to specify labels as keyword arguments

    Returns
    -------
    xr.Dataset or xr.DataArray
        Selected data with masking applied if requested

    Examples
    --------
    >>> from linked_indices import nd_sel
    >>> # Standard slice selection (equivalent to ds.sel())
    >>> result = nd_sel(ds, abs_time=slice(1.0, 2.0))

    >>> # Mask values outside range with NaN
    >>> result = nd_sel(ds, abs_time=slice(1.0, 2.0), returns='mask')

    >>> # Add boolean coordinate showing membership
    >>> result = nd_sel(ds, abs_time=slice(1.0, 2.0), returns='metadata')

    >>> # Combine with method='nearest'
    >>> result = nd_sel(ds, abs_time=slice(0.95, 2.1), method='nearest', returns='mask')
    """
    # Merge labels dict with kwargs
    if labels is None:
        labels = {}
    labels = {**labels, **label_kwargs}

    if not labels:
        raise ValueError("Must provide at least one coordinate label")

    # Find the NDIndex that manages these coordinates
    nd_index = None
    for name in labels:
        if name in obj.xindexes:
            idx = obj.xindexes[name]
            if isinstance(idx, NDIndex):
                nd_index = idx
                break

    if nd_index is None:
        # No NDIndex found - fall back to standard sel for 'slice' mode
        if returns == "slice":
            return obj.sel(labels, method=method)
        else:
            raise ValueError(
                f"No NDIndex found for coordinates {list(labels.keys())}. "
                f"returns='{returns}' requires an NDIndex-managed coordinate."
            )

    return nd_index.sel_masked(obj, labels, method=method, returns=returns)
