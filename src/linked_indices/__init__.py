try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .multi_interval_index import DimensionInterval
from .nd_index import NDIndex, nd_sel
from . import example_data
from . import viz

__all__ = [
    "DimensionInterval",
    "NDIndex",
    "nd_sel",
    "__version__",
    "example_data",
    "viz",
]


def main() -> None:
    print("Hello from linked-indices!")
