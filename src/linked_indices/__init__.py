try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .interval_index import DimensionInterval
from .multi_interval_index import DimensionIntervalMulti

__all__ = ["DimensionInterval", "DimensionIntervalMulti", "__version__"]


def main() -> None:
    print("Hello from linked-indices!")
