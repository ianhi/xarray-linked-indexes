---
kernelspec:
  name: python3
  display_name: Python 3
---

# Linked Indices

Custom xarray indexes for keeping multiple coordinates in sync across shared dimensions.

## Overview

This library provides custom [xarray Index](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html) implementations that automatically constrain related dimensions when you select on any one of them.

### Use Cases

- **Speech/audio data** with hierarchical annotations (words, phonemes, time)
- **Time series** with multiple granularities of events
- **Any data** where intervals at different scales need to stay synchronized

## Installation

```bash
pip install linked-indices
```

Or install from source:

```bash
git clone https://github.com/ianhi/xarray-linked-indexes
cd xarray-linked-indexes
pip install -e .
```

## Quick Start

```{code-cell} python
from linked_indices.util import multi_interval_dataset
from linked_indices import DimensionIntervalMulti

# Load example dataset with time, words, and phonemes
ds = multi_interval_dataset()

# Apply the linked index
ds = ds.drop_indexes(["time", "word", "phoneme"]).set_xindex(
    ["time", "word_intervals", "phoneme_intervals", "word", "part_of_speech", "phoneme"],
    DimensionIntervalMulti,
)
ds
```

Now selecting on any dimension automatically constrains all other dimensions to overlapping values:

```{code-cell} python
# Select word "red" - time and phonemes are auto-constrained to [0, 40)
ds.sel(word="red")
```

## Examples

See the example notebooks in the sidebar:

- [Multi-Interval Index](multi_interval_example.ipynb) - Multiple interval types (words, phonemes) over a shared continuous dimension
- [Dimension Interval](DimensionInterval.ipynb) - Single interval dimension linked to continuous time

## API Reference

### DimensionIntervalMulti

The main index class for linking multiple interval dimensions over a single continuous dimension.

**Features:**
- Automatic cross-slicing between all linked dimensions
- Support for multiple label coordinates per interval dimension
- Works with both `sel()` and `isel()` operations

**Known Limitations:**
- Intervals are assumed to be contiguous (no gaps)
- Array indexers (fancy indexing) not fully supported for interval dimensions
