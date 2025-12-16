try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .multi_interval_index import DimensionInterval
from . import example_data

__all__ = ["DimensionInterval", "__version__", "example_data"]


def main() -> None:
    print("Hello from linked-indices!")
