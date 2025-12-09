try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from .interval_index import DimensionInterval

__all__ = ["DimensionInterval", "__version__"]


def main() -> None:
    print("Hello from linked-indices!")
