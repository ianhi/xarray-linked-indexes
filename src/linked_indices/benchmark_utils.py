"""Benchmark utilities for linked_indices performance testing.

This module provides helpers for rigorous performance measurement using
Python's timeit module.
"""

from __future__ import annotations

import timeit
from typing import Any

import numpy as np

__all__ = ["timeit_benchmark"]


def timeit_benchmark(
    stmt: str | callable,
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
