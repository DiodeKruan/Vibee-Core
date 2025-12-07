"""PyDeck layer factories for different visualization types."""

from typing import List

import pandas as pd
import pydeck as pdk

from config.settings import settings


def create_layer(
    layer_type: str,
    data: pd.DataFrame,
    **kwargs,
) -> pdk.Layer:
    """
    Factory function to create PyDeck layers.

    Args:
        layer_type: Type of layer ('scatter', 'heatmap', 'hexagon', 'cluster')
        data: DataFrame with lat/lon data
        **kwargs: Additional layer-specific parameters

    Returns:
        PyDeck Layer object
    """
    layer_factories = {
        "scatter": create_scatter_layer,
        "heatmap": create_heatmap_layer,
        "hexagon": create_hexagon_layer,
        "cluster": create_cluster_layer,
    }

    factory = layer_factories.get(layer_type, create_scatter_layer)
    return factory(data, **kwargs)


def get_color_for_category(category: str) -> List[int]:
    """Get RGBA color for a category."""
    return list(settings.categories.colors.get(category, (100, 100, 100, 200)))


def create_scatter_layer(
    data: pd.DataFrame,
    radius: int = 100,
    opacity: float = 0.8,
    pickable: bool = True,
    **kwargs,
) -> pdk.Layer:
    """
    Create a ScatterplotLayer for point markers.

    Args:
        data: DataFrame with columns: lat, lon, category (optional)
        radius: Point radius in meters
        opacity: Layer opacity (0-1)
        pickable: Whether points are clickable

    Returns:
        ScatterplotLayer
    """
    if data.empty:
        return pdk.Layer(
            "ScatterplotLayer",
            data=[],
            get_position="[lon, lat]",
        )

    # Add color based on category if available
    if "category" in data.columns:
        data = data.copy()
        data["color"] = data["category"].apply(get_color_for_category)
    else:
        data = data.copy()
        data["color"] = [[255, 140, 0, 200]] * len(data)

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=radius,
        radius_min_pixels=3,
        radius_max_pixels=30,
        opacity=opacity,
        pickable=pickable,
        auto_highlight=True,
    )


def create_heatmap_layer(
    data: pd.DataFrame,
    radius: int = 30,
    opacity: float = 0.8,
    intensity: float = 1,
    threshold: float = 0.05,
    **kwargs,
) -> pdk.Layer:
    """
    Create a HeatmapLayer for density visualization.

    Args:
        data: DataFrame with columns: lat, lon, weight (optional)
        radius: Radius of influence in pixels
        opacity: Layer opacity (0-1)
        intensity: Intensity multiplier
        threshold: Minimum threshold for display

    Returns:
        HeatmapLayer
    """
    if data.empty:
        return pdk.Layer(
            "HeatmapLayer",
            data=[],
            get_position="[lon, lat]",
        )

    # Use weight column if available
    weight_col = "weight" if "weight" in data.columns else None

    layer_config = {
        "type": "HeatmapLayer",
        "data": data,
        "get_position": "[lon, lat]",
        "radiusPixels": radius,
        "opacity": opacity,
        "intensity": intensity,
        "threshold": threshold,
        "aggregation": "SUM",
    }

    if weight_col:
        layer_config["get_weight"] = weight_col

    return pdk.Layer(**layer_config)


def create_hexagon_layer(
    data: pd.DataFrame,
    radius: int = 200,
    elevation_scale: int = 4,
    elevation_range: List[int] = None,
    opacity: float = 0.8,
    extruded: bool = True,
    coverage: float = 0.8,
    **kwargs,
) -> pdk.Layer:
    """
    Create a HexagonLayer for 3D hexagonal binning.

    Args:
        data: DataFrame with columns: lat, lon
        radius: Hexagon radius in meters
        elevation_scale: Multiplier for elevation
        elevation_range: [min, max] elevation values
        opacity: Layer opacity (0-1)
        extruded: Whether to show 3D elevation
        coverage: Hexagon coverage (0-1)

    Returns:
        HexagonLayer
    """
    if elevation_range is None:
        elevation_range = [0, 1000]

    if data.empty:
        return pdk.Layer(
            "HexagonLayer",
            data=[],
            get_position="[lon, lat]",
        )

    return pdk.Layer(
        "HexagonLayer",
        data=data,
        get_position="[lon, lat]",
        radius=radius,
        elevation_scale=elevation_scale,
        elevation_range=elevation_range,
        extruded=extruded,
        coverage=coverage,
        opacity=opacity,
        pickable=True,
        auto_highlight=True,
        color_range=[
            [255, 255, 178],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [240, 59, 32],
            [189, 0, 38],
        ],
    )


def create_cluster_layer(
    data: pd.DataFrame,
    radius: int = 100,
    opacity: float = 0.8,
    **kwargs,
) -> pdk.Layer:
    """
    Create a cluster visualization using ScatterplotLayer with size encoding.
    Note: True clustering requires pre-aggregation in the data layer.

    Args:
        data: DataFrame with columns: lat, lon, count (pre-aggregated)
        radius: Base radius for clusters
        opacity: Layer opacity (0-1)

    Returns:
        ScatterplotLayer configured for clustering visualization
    """
    if data.empty:
        return pdk.Layer(
            "ScatterplotLayer",
            data=[],
            get_position="[lon, lat]",
        )

    # If count column exists, use it for sizing
    if "count" in data.columns:
        data = data.copy()
        # Normalize count to radius (sqrt for area-proportional sizing)
        max_count = data["count"].max()
        data["cluster_radius"] = (
            (data["count"] / max_count).apply(lambda x: x ** 0.5) * radius * 3 + radius
        )
        # Color intensity based on count
        data["color"] = data["count"].apply(
            lambda c: [
                min(255, int(200 + (c / max_count) * 55)),
                max(0, int(140 - (c / max_count) * 100)),
                0,
                200,
            ]
        )
    else:
        data = data.copy()
        data["cluster_radius"] = radius
        data["color"] = [[255, 140, 0, 200]] * len(data)

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_color="color",
        get_radius="cluster_radius",
        radius_min_pixels=10,
        radius_max_pixels=100,
        opacity=opacity,
        pickable=True,
        auto_highlight=True,
    )


def create_text_layer(
    data: pd.DataFrame,
    text_column: str = "count",
    size: int = 16,
    **kwargs,
) -> pdk.Layer:
    """
    Create a TextLayer for labels (used with cluster layer).

    Args:
        data: DataFrame with lat, lon, and text column
        text_column: Column to use for text labels
        size: Font size

    Returns:
        TextLayer
    """
    if data.empty or text_column not in data.columns:
        return pdk.Layer(
            "TextLayer",
            data=[],
            get_position="[lon, lat]",
        )

    return pdk.Layer(
        "TextLayer",
        data=data,
        get_position="[lon, lat]",
        get_text=text_column,
        get_size=size,
        get_color=[255, 255, 255, 255],
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
    )

