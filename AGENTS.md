# Agent Instructions

## Important: Always use `uv run`

**Always use `uv run python` for running Python commands and `uv run pytest` for tests.** Never use bare `python` or `pytest` commands.

## Project Overview

Custom xarray Index implementations for keeping multiple coordinates in sync across shared dimensions. Primary use case: time series data with interval-based metadata (e.g., speech data with word/phoneme annotations).

## Key Files

- `src/linked_indices/multi_interval_index.py` - Main implementation: `DimensionInterval`
- `src/linked_indices/util.py` - Dataset generators for examples/tests
- `tests/test_interval_index.py` - Test suite
- `docs/` - MyST/Jupyter Book documentation

## Development

```bash
uv run pytest                        # Run tests
uv run jupyter execute <notebook>    # Execute a notebook in place
uv run jupyter-book build docs/      # Build docs
uv run jupyter-book start docs/      # Preview docs locally
```

## Architecture Notes

- `DimensionInterval` manages one continuous dimension (e.g., time) and multiple interval dimensions (e.g., words, phonemes)
- Uses xarray's custom Index API (`Index.from_variables`, `sel`, `isel`)
- `sel()` returns `IndexSelResult` with indexers for ALL affected dimensions
- `isel()` returns a new index instance with updated internal state
- Both must handle cross-dimension constraints (selecting on one constrains others)

## Testing Quirk

Tests use `_ = result * 1` to force lazy evaluation. Without this, xarray's lazy indexing means our `isel()` isn't called and dimension sizes aren't constrained.

## Known Limitations

- Assumes contiguous intervals (no gaps)
- Array indexers not fully supported for interval dimensions
- `isel` on interval dims can't propagate to other interval dims (xarray limitation, tests marked xfail)
