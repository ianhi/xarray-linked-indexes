"""Utilities for linked_indices.

Note: Dataset generators have been moved to the example_data module.
This module re-exports them for backward compatibility.
"""

# Re-export from example_data for backward compatibility
from linked_indices.example_data import multi_interval_dataset, onset_duration_dataset

__all__ = ["multi_interval_dataset", "onset_duration_dataset"]
