"""Tests for visualization utilities."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")  # Use non-interactive backend for tests

from linked_indices import NDIndex
from linked_indices.viz import visualize_radial_selection, visualize_slice_selection


@pytest.fixture
def da_with_linear_coord():
    """Create a DataArray with a linear 2D coordinate."""
    ny, nx = 100, 80
    data = np.random.rand(ny, nx)
    y_coord = np.arange(ny)
    x_coord = np.arange(nx)

    # Linear coordinate: derived = y * nx + x (unique values)
    derived_coord = y_coord[:, np.newaxis] * nx + x_coord[np.newaxis, :]

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": y_coord,
            "x": x_coord,
            "derived": (["y", "x"], derived_coord),
        },
    )
    return da.set_xindex(["derived"], NDIndex)


@pytest.fixture
def da_with_radial_coord():
    """Create a DataArray with a radial 2D coordinate."""
    ny, nx = 100, 80
    data = np.random.rand(ny, nx)
    y_coord = np.arange(ny)
    x_coord = np.arange(nx)

    # Radial coordinate centered on the image
    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")
    radial_coord = np.sqrt(xx**2 + yy**2)

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": y_coord,
            "x": x_coord,
            "radius": (["y", "x"], radial_coord),
        },
    )
    return da.set_xindex(["radius"], NDIndex)


class TestVisualizeSliceSelection:
    """Tests for visualize_slice_selection function."""

    def test_basic_visualization(self, da_with_linear_coord):
        """Test that visualize_slice_selection returns a figure."""
        import matplotlib.pyplot as plt

        fig = visualize_slice_selection(da_with_linear_coord, 1000, 2000)
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_custom_coord_name(self, da_with_linear_coord):
        """Test visualization with explicit coord_name."""
        import matplotlib.pyplot as plt

        fig = visualize_slice_selection(
            da_with_linear_coord, 1000, 2000, coord_name="derived"
        )
        assert fig is not None
        plt.close(fig)

    def test_narrow_range(self, da_with_linear_coord):
        """Test visualization with a narrow range."""
        import matplotlib.pyplot as plt

        fig = visualize_slice_selection(da_with_linear_coord, 4000, 4100)
        assert fig is not None
        plt.close(fig)


class TestVisualizeRadialSelection:
    """Tests for visualize_radial_selection function."""

    def test_basic_visualization(self, da_with_radial_coord):
        """Test that visualize_radial_selection returns a figure."""
        import matplotlib.pyplot as plt

        fig = visualize_radial_selection(da_with_radial_coord, 10, 30)
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_custom_coord_name(self, da_with_radial_coord):
        """Test visualization with explicit coord_name."""
        import matplotlib.pyplot as plt

        fig = visualize_radial_selection(
            da_with_radial_coord, 10, 30, coord_name="radius"
        )
        assert fig is not None
        plt.close(fig)

    def test_center_selection(self, da_with_radial_coord):
        """Test visualization selecting center region."""
        import matplotlib.pyplot as plt

        fig = visualize_radial_selection(da_with_radial_coord, 0, 20)
        assert fig is not None
        plt.close(fig)

    def test_annulus_selection(self, da_with_radial_coord):
        """Test visualization selecting an annulus (ring)."""
        import matplotlib.pyplot as plt

        fig = visualize_radial_selection(da_with_radial_coord, 20, 40)
        assert fig is not None
        plt.close(fig)
