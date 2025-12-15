---
kernelspec:
  name: python3
  display_name: Python 3
---

# Alternatives

There are two alternative solutions already built into xarray for solving the problems of this package.

You have time series data with metadata that spans intervals of time (e.g., words spoken during a recording). You want to:

1. Select data by time range
2. Select data by metadata (e.g., "give me all data during word X")
3. Keep the time and metadata dimensions synchronized

| Approach | Pros | Cons |
|----------|------|------|
| [**Direct Coords**](alt-direct-coords.ipynb) | Simple to understand; works with standard xarray | Must construct dense arrays; loses interval boundary info; no `isel` by metadata |
| [**MultiIndex**](alt-multiindex.ipynb) | Built-in pandas/xarray support; familiar API | Awkward slicing behavior; duplicates time values; can't easily get interval boundaries |
| **Linked Indices** (this library) | Preserves interval structure; automatic cross-slicing; natural representation | Requires custom Index; newer xarray feature |


If either of multindex or direct coords works for you use case then you should prefer to use them over this custom index.


## Use MultiIndex when:
- You're already familiar with pandas MultiIndex
- Your intervals perfectly tile the time axis (no gaps)
- You don't need to query interval boundaries

## Use Direct Coords when:
- You have simple, non-overlapping intervals
- You don't need `isel` access by metadata
- You want to avoid dependencies

## Use Linked Indices when:
- You need to preserve interval boundary information
- You have hierarchical intervals (e.g., words containing phonemes)
- You need natural `sel()` and `isel()` on both time and metadata

## Detailed Examples

- [MultiIndex Approach](alt-multiindex.ipynb) - Using pandas MultiIndex with xarray
- [Direct Coords Approach](alt-direct-coords.ipynb) - Encoding metadata as coordinates on the time dimension
