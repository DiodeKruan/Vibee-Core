"""Data sampling and aggregation utilities for large datasets."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def sample_data(
    df: pd.DataFrame,
    max_points: int = 10000,
    method: str = "random",
    weight_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sample data points from a large DataFrame.

    Args:
        df: Input DataFrame
        max_points: Maximum number of points to return
        method: Sampling method ('random', 'systematic', 'weighted')
        weight_column: Column to use for weighted sampling

    Returns:
        Sampled DataFrame
    """
    if len(df) <= max_points:
        return df

    if method == "random":
        return df.sample(n=max_points, random_state=42)

    elif method == "systematic":
        # Take every nth row
        step = len(df) // max_points
        return df.iloc[::step].head(max_points)

    elif method == "weighted" and weight_column and weight_column in df.columns:
        # Weighted sampling based on a column
        weights = df[weight_column] / df[weight_column].sum()
        return df.sample(n=max_points, weights=weights, random_state=42)

    return df.sample(n=max_points, random_state=42)


def aggregate_to_hexagons(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    hex_size: float = 0.002,
) -> pd.DataFrame:
    """
    Aggregate points into hexagonal bins.

    Args:
        df: Input DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        hex_size: Size of hexagon in degrees

    Returns:
        DataFrame with aggregated hexagon centers and counts
    """
    if df.empty:
        return pd.DataFrame(columns=[lat_col, lon_col, "count", "weight"])

    # Simple grid-based aggregation (approximation of hexagons)
    df = df.copy()
    df["hex_lat"] = (df[lat_col] / hex_size).round() * hex_size
    df["hex_lon"] = (df[lon_col] / hex_size).round() * hex_size

    aggregated = (
        df.groupby(["hex_lat", "hex_lon"])
        .size()
        .reset_index(name="count")
    )

    aggregated = aggregated.rename(columns={"hex_lat": lat_col, "hex_lon": lon_col})
    aggregated["weight"] = aggregated["count"].astype(float)

    return aggregated


def compute_density_grid(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    grid_resolution: int = 100,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Compute a density grid for heatmap visualization.

    Args:
        df: Input DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        grid_resolution: Number of cells per dimension
        bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat)

    Returns:
        Tuple of (density_grid, bbox)
    """
    if df.empty:
        return np.zeros((grid_resolution, grid_resolution)), (0, 0, 1, 1)

    if bbox is None:
        min_lon = df[lon_col].min()
        max_lon = df[lon_col].max()
        min_lat = df[lat_col].min()
        max_lat = df[lat_col].max()
        bbox = (min_lon, min_lat, max_lon, max_lat)

    min_lon, min_lat, max_lon, max_lat = bbox

    # Create grid
    lon_bins = np.linspace(min_lon, max_lon, grid_resolution + 1)
    lat_bins = np.linspace(min_lat, max_lat, grid_resolution + 1)

    # Compute 2D histogram
    density, _, _ = np.histogram2d(
        df[lon_col].values,
        df[lat_col].values,
        bins=[lon_bins, lat_bins],
    )

    return density, bbox


def filter_by_viewport(
    df: pd.DataFrame,
    viewport: dict,
    lat_col: str = "lat",
    lon_col: str = "lon",
    buffer: float = 0.1,
) -> pd.DataFrame:
    """
    Filter data to only include points within the map viewport.

    Args:
        df: Input DataFrame
        viewport: Dict with 'latitude', 'longitude', 'zoom' keys
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        buffer: Buffer zone as fraction of viewport

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    # Approximate viewport bounds based on zoom level
    center_lat = viewport.get("latitude", 13.7563)
    center_lon = viewport.get("longitude", 100.5018)
    zoom = viewport.get("zoom", 11)

    # Approximate degrees per pixel at this zoom level
    # At zoom 11, roughly 0.1 degrees visible
    lat_range = 180 / (2 ** zoom) * (1 + buffer)
    lon_range = 360 / (2 ** zoom) * (1 + buffer)

    min_lat = center_lat - lat_range
    max_lat = center_lat + lat_range
    min_lon = center_lon - lon_range
    max_lon = center_lon + lon_range

    return df[
        (df[lat_col] >= min_lat)
        & (df[lat_col] <= max_lat)
        & (df[lon_col] >= min_lon)
        & (df[lon_col] <= max_lon)
    ]

