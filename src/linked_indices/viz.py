"""Visualization utilities for NDIndex."""

import numpy as np

__all__ = ["visualize_slice_selection", "visualize_radial_selection"]


def visualize_slice_selection(da_indexed, start, stop, coord_name="derived"):
    """
    Visualize a slice selection showing the bounding box behavior.

    Parameters
    ----------
    da_indexed : xr.DataArray
        DataArray with an NDIndex-managed 2D coordinate
    start, stop : float
        Range boundaries for the slice selection
    coord_name : str
        Name of the coordinate to select on (default: "derived")

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    result = da_indexed.sel({coord_name: slice(start, stop)})

    y_min, y_max = result.y.values[0], result.y.values[-1]
    x_min, x_max = result.x.values[0], result.x.values[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original with bounding box
    ax = axes[0]
    ax.imshow(da_indexed.values, cmap="gray")
    rect = plt.Rectangle(
        (x_min - 0.5, y_min - 0.5),
        x_max - x_min + 1,
        y_max - y_min + 1,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )
    ax.add_patch(rect)
    ax.set_title(f"Original with Bounding Box\n{coord_name} in [{start}, {stop}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Derived coordinate with mask
    ax = axes[1]
    coord_values = da_indexed[coord_name].values
    in_range = (coord_values >= start) & (coord_values <= stop)

    ax.imshow(coord_values, cmap="viridis", alpha=0.5)
    ax.imshow(np.where(in_range, 1, np.nan), cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    rect = plt.Rectangle(
        (x_min - 0.5, y_min - 0.5),
        x_max - x_min + 1,
        y_max - y_min + 1,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )
    ax.add_patch(rect)
    ax.set_title("Cells in range (red) vs bounding box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Result
    ax = axes[2]
    ax.imshow(result.values, cmap="gray")
    ax.set_title(f"Result: {result.shape}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    return fig


def visualize_radial_selection(da_indexed, start, stop, coord_name="radius"):
    """
    Visualize selection on a radial coordinate.

    Parameters
    ----------
    da_indexed : xr.DataArray
        DataArray with an NDIndex-managed radial coordinate
    start, stop : float
        Range boundaries for the slice selection
    coord_name : str
        Name of the radial coordinate (default: "radius")

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    result = da_indexed.sel({coord_name: slice(start, stop)})

    y_min, y_max = result.y.values[0], result.y.values[-1]
    x_min, x_max = result.x.values[0], result.x.values[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.imshow(da_indexed.values, cmap="gray")
    rect = plt.Rectangle(
        (x_min - 0.5, y_min - 0.5),
        x_max - x_min + 1,
        y_max - y_min + 1,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )
    ax.add_patch(rect)
    ax.set_title(f"Bounding box for {coord_name} in [{start}, {stop}]")

    ax = axes[1]
    coord_values = da_indexed[coord_name].values
    in_range = (coord_values >= start) & (coord_values <= stop)
    # Get the underlying image data for display
    gray = da_indexed.values
    ax.imshow(np.where(in_range, gray, gray * 0.3), cmap="gray")
    ax.set_title(f"Cells with {coord_name} in [{start}, {stop}]")

    ax = axes[2]
    ax.imshow(result.values, cmap="gray")
    ax.set_title(f"Result (bounding box): {result.shape}")

    plt.tight_layout()
    return fig
