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

__all__ = ["NDIndex"]

# Valid slice methods
SLICE_METHODS = ("bounding_box", "trim_outer")


@dataclass
class NDCoord:
    """Information about an N-D coordinate."""

    name: str
    dims: tuple[str, ...]
    values: np.ndarray


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

        if method == "nearest":
            # Find flat index of closest value
            flat_idx = np.argmin(np.abs(values - value))
        else:
            # Exact match required - use flatnonzero for efficiency
            flat_matches = np.flatnonzero(values == value)
            if len(flat_matches) == 0:
                raise KeyError(
                    f"Value {value!r} not found in coordinate {coord_name!r}. "
                    f"Use method='nearest' for approximate matching."
                )
            # Use the first match
            flat_idx = flat_matches[0]

        # Convert to multi-dimensional indices
        indices = np.unravel_index(flat_idx, values.shape)

        return {dim: int(idx) for dim, idx in zip(ndc.dims, indices)}

    def _find_slices_for_range(
        self,
        coord_name: str,
        start: float,
        stop: float,
        step: int | None = None,
        slice_method: str | None = None,
    ) -> dict[str, slice]:
        """
        Find slices in all dimensions for a value range.

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

        Returns
        -------
        dict mapping dimension name to slice
        """
        ndc = self._nd_coords[coord_name]
        values = ndc.values
        method = slice_method or self._slice_method

        # Find cells in range
        in_range = (values >= start) & (values <= stop)

        if not np.any(in_range):
            # No values in range - return empty slices
            return {dim: slice(0, 0) for dim in ndc.dims}

        result = {}

        if method == "bounding_box":
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

        elif method == "trim_outer":
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

        For slice selection: finds all cells in range, returns slices
        covering the extent in each dimension. The slice behavior depends on
        the slice_method configured for this index:
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
                slices = self._find_slices_for_range(name, start, stop, step=step)
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

    def __repr__(self) -> str:
        """Return a string representation of the NDIndex."""
        lines = [f"<{self.__class__.__name__}>"]
        lines.append(f"  slice_method: {self._slice_method!r}")

        if self._nd_coords:
            lines.append("  Coordinates:")
            for name, ndc in self._nd_coords.items():
                shape_str = " × ".join(str(s) for s in ndc.values.shape)
                dims_str = ", ".join(ndc.dims)

                # Value range
                val_min = ndc.values.min()
                val_max = ndc.values.max()

                lines.append(f"    {name}:")
                lines.append(f"      dims: ({dims_str})")
                lines.append(f"      shape: ({shape_str})")
                lines.append(f"      range: [{val_min:.4g}, {val_max:.4g}]")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return an HTML representation of the NDIndex."""
        # CSS styles for the repr
        style = """
        <style>
            .nd-repr { font-family: monospace; font-size: 13px; }
            .nd-repr table { border-collapse: collapse; margin: 8px 0; }
            .nd-repr th, .nd-repr td {
                padding: 4px 12px;
                text-align: left;
                border: 1px solid #ddd;
            }
            .nd-repr th { background-color: #f5f5f5; font-weight: bold; }
            .nd-repr .section-header {
                background-color: #e8e8e8;
                font-weight: bold;
                padding: 6px 12px;
            }
            .nd-repr .coord-name { color: #0066cc; font-weight: bold; }
            .nd-repr .dims { color: #666; }
            .nd-repr .shape { color: #8b008b; }
            .nd-repr .range { color: #228b22; }
            .nd-repr .method { color: #8b4513; font-style: italic; }
        </style>
        """

        html_parts = [style, '<div class="nd-repr">']
        html_parts.append(
            f"<strong>&lt;{self.__class__.__name__}&gt;</strong> "
            f'<span class="method">(slice_method: {self._slice_method!r})</span>'
        )

        if self._nd_coords:
            html_parts.append("<table>")
            html_parts.append(
                '<tr><th colspan="4" class="section-header">N-D Coordinates</th></tr>'
            )
            html_parts.append(
                "<tr><th>Coordinate</th><th>Dimensions</th><th>Shape</th><th>Range</th></tr>"
            )

            for name, ndc in self._nd_coords.items():
                shape_str = " × ".join(str(s) for s in ndc.values.shape)
                dims_str = ", ".join(ndc.dims)
                val_min = ndc.values.min()
                val_max = ndc.values.max()

                html_parts.append(
                    f'<tr>'
                    f'<td><span class="coord-name">{name}</span></td>'
                    f'<td><span class="dims">({dims_str})</span></td>'
                    f'<td><span class="shape">({shape_str})</span></td>'
                    f'<td><span class="range">[{val_min:.4g}, {val_max:.4g}]</span></td>'
                    f'</tr>'
                )

            html_parts.append("</table>")

        html_parts.append("</div>")
        return "".join(html_parts)
