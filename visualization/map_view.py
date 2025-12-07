"""Main PyDeck map component with layer management."""

from typing import Dict, List, Optional

import pandas as pd
import pydeck as pdk
import streamlit as st

from config.settings import settings
from .layers import create_layer, create_text_layer


def get_initial_view_state(
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: Optional[int] = None,
    pitch: Optional[int] = None,
    bearing: Optional[int] = None,
) -> pdk.ViewState:
    """
    Create initial view state for the map.

    Args:
        center_lat: Latitude of map center
        center_lon: Longitude of map center
        zoom: Initial zoom level
        pitch: Tilt angle (0-60)
        bearing: Rotation angle

    Returns:
        PyDeck ViewState object
    """
    return pdk.ViewState(
        latitude=center_lat or settings.map.center_lat,
        longitude=center_lon or settings.map.center_lon,
        zoom=zoom or settings.map.default_zoom,
        pitch=pitch or settings.map.default_pitch,
        bearing=bearing or settings.map.default_bearing,
    )


def get_tooltip_config(layer_type: str) -> Dict:
    """
    Get tooltip configuration based on layer type.

    Args:
        layer_type: Type of visualization layer

    Returns:
        Tooltip configuration dictionary
    """
    if layer_type in ["scatter"]:
        return {
            "html": """
                <div style="
                    background: rgba(20, 20, 30, 0.95);
                    padding: 12px 16px;
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-family: 'Inter', system-ui, sans-serif;
                    max-width: 280px;
                ">
                    <div style="
                        color: #f97316;
                        font-weight: 600;
                        font-size: 13px;
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">{category}</div>
                    <div style="
                        color: #e2e8f0;
                        font-size: 12px;
                        line-height: 1.5;
                        margin-bottom: 6px;
                    ">{description}</div>
                    <div style="
                        color: #94a3b8;
                        font-size: 11px;
                        border-top: 1px solid rgba(255, 255, 255, 0.1);
                        padding-top: 6px;
                        margin-top: 6px;
                    ">
                        <span style="margin-right: 12px;">📍 {lat:.4f}, {lon:.4f}</span>
                    </div>
                </div>
            """,
            "style": {
                "backgroundColor": "transparent",
                "color": "white",
            },
        }

    elif layer_type in ["hexagon", "cluster"]:
        return {
            "html": """
                <div style="
                    background: rgba(20, 20, 30, 0.95);
                    padding: 10px 14px;
                    border-radius: 6px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-family: 'Inter', system-ui, sans-serif;
                ">
                    <div style="color: #f97316; font-weight: 600; font-size: 14px;">
                        {elevationValue} reports
                    </div>
                </div>
            """,
            "style": {
                "backgroundColor": "transparent",
                "color": "white",
            },
        }

    return {}


def render_map(
    data: pd.DataFrame,
    layer_type: str = "scatter",
    layer_params: Optional[Dict] = None,
    view_state: Optional[pdk.ViewState] = None,
    height: int = 700,
) -> Optional[pdk.Deck]:
    """
    Render the PyDeck map with specified layer.

    Args:
        data: DataFrame with visualization data
        layer_type: Type of layer to render
        layer_params: Parameters for the layer
        view_state: Custom view state (uses default if None)
        height: Map height in pixels

    Returns:
        PyDeck Deck object or None if rendering fails
    """
    if layer_params is None:
        layer_params = {}

    # Get or create view state
    if view_state is None:
        view_state = get_initial_view_state()

    # Create the visualization layer
    layers = [create_layer(layer_type, data, **layer_params)]

    # Add text layer for clusters
    if layer_type == "cluster" and "count" in data.columns:
        layers.append(create_text_layer(data, text_column="count"))

    # Get appropriate tooltip
    tooltip = get_tooltip_config(layer_type)

    # Create deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v11",
    )

    # Render in Streamlit
    st.pydeck_chart(deck, use_container_width=True, height=height)

    return deck


def render_map_with_controls(
    data: pd.DataFrame,
    layer_type: str = "scatter",
    params: Optional[Dict] = None,
) -> None:
    """
    Render map with integrated parameter controls stored in session state.

    Args:
        data: DataFrame with visualization data
        layer_type: Type of layer to render
        params: Visualization parameters from sidebar
    """
    if params is None:
        params = {}

    # Build layer parameters from session state or defaults
    layer_params = {
        "radius": params.get("radius", settings.map.default_point_radius),
        "opacity": params.get("opacity", settings.map.default_opacity),
        "elevation_scale": params.get("elevation_scale", settings.map.default_elevation_scale),
    }

    # Get view state from session if available
    view_state = None
    if "map_view_state" in st.session_state:
        vs = st.session_state.map_view_state
        view_state = get_initial_view_state(
            center_lat=vs.get("latitude"),
            center_lon=vs.get("longitude"),
            zoom=vs.get("zoom"),
            pitch=vs.get("pitch"),
            bearing=vs.get("bearing"),
        )

    render_map(
        data=data,
        layer_type=layer_type,
        layer_params=layer_params,
        view_state=view_state,
    )

