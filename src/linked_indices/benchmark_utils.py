"""Benchmark utilities for linked_indices performance testing.

This module provides helpers for rigorous performance measurement using
Python's timeit module.
"""

from __future__ import annotations

import timeit
from typing import Any, Callable

import numpy as np

__all__ = ["timeit_benchmark", "benchmark_selection_scaling"]


def timeit_benchmark(
    stmt: str | Callable[[], Any],
    setup: str = "pass",
    globals: dict[str, Any] | None = None,
    repeat: int = 7,
) -> dict[str, float | int]:
    """
    Benchmark a statement or callable using timeit with automatic loop count.

    Uses timeit's autorange to determine an appropriate number of loops,
    then runs multiple trials to get reliable timing statistics.

    Parameters
    ----------
    stmt : str or callable
        The statement or callable to benchmark.
    setup : str
        Setup code to run once before the benchmark. Default: "pass"
    globals : dict, optional
        Global namespace for the benchmark. Required when stmt uses
        variables from the calling scope.
    repeat : int
        Number of trials to run. Default: 7 (timeit default)

    Returns
    -------
    dict
        Dictionary with timing statistics:
        - best_ms: Minimum time per call in milliseconds (best measure of algorithm cost)
        - mean_ms: Mean time per call in milliseconds (typical real-world performance)
        - std_ms: Standard deviation of times in milliseconds
        - n_loops: Number of loops per trial (determined by autorange)

    Examples
    --------
    >>> from linked_indices.benchmark_utils import timeit_benchmark
    >>> import numpy as np
    >>> arr = np.random.randn(1000)
    >>> result = timeit_benchmark(
    ...     lambda: np.sum(arr),
    ...     globals={"np": np, "arr": arr}
    ... )
    >>> result["best_ms"] < 1.0  # Should be fast
    True
    """
    timer = timeit.Timer(stmt, setup=setup, globals=globals)

    # Let timeit determine appropriate number of loops
    n_loops, _ = timer.autorange()

    # Run multiple trials
    times = timer.repeat(repeat=repeat, number=n_loops)
    times_per_call = np.array(times) / n_loops

    return {
        "best_ms": float(times_per_call.min() * 1000),
        "mean_ms": float(times_per_call.mean() * 1000),
        "std_ms": float(times_per_call.std() * 1000),
        "n_loops": n_loops,
    }


def benchmark_selection_scaling(
    create_dataset: Callable[[int, int], Any],
    sizes: list[tuple[int, int]],
    coord_name: str = "abs_time",
    slice_fraction: float = 0.5,
    force_unsorted: bool = False,
    method: str | None = None,
    cold_repeats: int = 3,
    print_results: bool = True,
) -> list[dict[str, Any]]:
    """
    Benchmark selection performance across different dataset sizes.

    This function benchmarks index lookup, cold sel(), and warm sel() for
    slice selection operations, providing a consistent way to measure
    performance scaling.

    Parameters
    ----------
    create_dataset : callable
        Function that takes (n_outer, n_inner) and returns an xarray Dataset
        with an NDIndex-indexed coordinate.
    sizes : list of (int, int) tuples
        List of (n_outer, n_inner) sizes to benchmark.
    coord_name : str
        Name of the coordinate to select on. Default: "abs_time"
    slice_fraction : float
        Fraction of the value range to select (centered). Default: 0.5 (50%)
    force_unsorted : bool
        If True, force the unsorted code path for comparison. Default: False
    method : str or None
        Selection method to use ('nearest' or None for exact). Default: None
    cold_repeats : int
        Number of cold (first call) measurements to take. Default: 3
    print_results : bool
        If True, print results as a formatted table. Default: True

    Returns
    -------
    list of dict
        List of result dictionaries, one per size, containing:
        - n_cells: Total number of cells
        - shape: Shape string (e.g., "100 × 1,000")
        - index_ms: Index lookup time in milliseconds
        - cold_ms: Cold (first call) sel() time in milliseconds
        - warm_ms: Warm (repeated) sel() time in milliseconds

    Examples
    --------
    >>> from linked_indices.benchmark_utils import benchmark_selection_scaling
    >>> from linked_indices.example_data import create_trial_ndindex_dataset
    >>> results = benchmark_selection_scaling(
    ...     create_trial_ndindex_dataset,
    ...     sizes=[(10, 100)],
    ...     print_results=False,
    ... )
    >>> len(results)
    1
    >>> 'index_ms' in results[0] and 'warm_ms' in results[0]
    True
    """
    results = []

    if print_results:
        path_label = "Unsorted" if force_unsorted else "Sorted"
        method_label = f", method='{method}'" if method else ""
        print(
            f"{path_label} Slice Selection: Index Lookup vs Full sel() ({int(slice_fraction * 100)}% slice{method_label})"
        )
        print("=" * 115)
        print(
            f"{'Shape':>20} | {'Cells':>10} | {'Index (ms)':>12} | {'Cold sel (ms)':>14} | {'Warm sel (ms)':>14}"
        )
        print("-" * 115)

    for n_outer, n_inner in sizes:
        n_cells = n_outer * n_inner
        shape_str = f"{n_outer:,} × {n_inner:,}"
        ds = create_dataset(n_outer, n_inner)

        # Get coordinate value range
        coord_values = ds[coord_name].values
        vmin, vmax = coord_values.min(), coord_values.max()
        half_frac = slice_fraction / 2
        start = vmin + (vmax - vmin) * (0.5 - half_frac)
        stop = vmin + (vmax - vmin) * (0.5 + half_frac)

        # Get the index object for direct benchmarking
        index = ds.xindexes[coord_name]

        # Optionally force unsorted path
        if force_unsorted:
            coord = index._nd_coords[coord_name]
            coord._is_sorted = False

        # Benchmark index lookup only
        result_index = timeit_benchmark(
            lambda: index._find_slices_for_range(
                coord_name, start, stop, method=method
            ),
            globals={
                "index": index,
                "coord_name": coord_name,
                "start": start,
                "stop": stop,
                "method": method,
            },
        )

        # Measure cold (first call) time
        cold_times = []
        for _ in range(cold_repeats):
            ds_fresh = create_dataset(n_outer, n_inner)
            if force_unsorted:
                idx_fresh = ds_fresh.xindexes[coord_name]
                coord_fresh = idx_fresh._nd_coords[coord_name]
                coord_fresh._is_sorted = False

            coord_values_f = ds_fresh[coord_name].values
            vmin_f, vmax_f = coord_values_f.min(), coord_values_f.max()
            start_f = vmin_f + (vmax_f - vmin_f) * (0.5 - half_frac)
            stop_f = vmin_f + (vmax_f - vmin_f) * (0.5 + half_frac)

            t0 = timeit.default_timer()
            _ = ds_fresh.sel({coord_name: slice(start_f, stop_f)}, method=method)
            t1 = timeit.default_timer()
            cold_times.append((t1 - t0) * 1000)
        cold_ms = min(cold_times)

        # Benchmark warm (repeated calls)
        result_warm = timeit_benchmark(
            lambda: ds.sel({coord_name: slice(start, stop)}, method=method),
            globals={
                "ds": ds,
                "coord_name": coord_name,
                "start": start,
                "stop": stop,
                "method": method,
            },
        )

        result = {
            "n_cells": n_cells,
            "shape": shape_str,
            "index_ms": result_index["best_ms"],
            "cold_ms": cold_ms,
            "warm_ms": result_warm["best_ms"],
        }
        results.append(result)

        if print_results:
            print(
                f"{shape_str:>20} | {n_cells:>10,} | {result_index['best_ms']:>12.4f} | {cold_ms:>14.3f} | {result_warm['best_ms']:>14.4f}"
            )

    return results
