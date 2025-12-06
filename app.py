"""
Traffy Fondue Data Visualization App
Bangkok Urban Report Analytics with PyDeck Map
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from components.sidebar import render_sidebar, init_sidebar_state
from components.chatbox import render_chatbox
from visualization.map_view import get_initial_view_state
from visualization.layers import create_layer
from mcp.ui_mcp import ui_mcp
from mcp.data_mcp import data_mcp
from utils.sampling import sample_data, aggregate_to_hexagons

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Traffy Fondue | Bangkok Reports",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    """Inject custom CSS for full-page map with floating components."""
    st.markdown(
        """
        <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Hide Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* Hide bottom container that covers map */
        .st-emotion-cache-qdbtli, 
        .ea3mdgi2,
        [data-testid="stBottomBlockContainer"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Remove all padding for full-screen map */
        .stApp {
            font-family: 'Space Grotesk', system-ui, sans-serif;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Make main content area full height */
        [data-testid="stMain"] {
            height: 100vh;
            overflow: hidden;
        }
        
        [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            max-width: 100% !important;
            height: 100vh;
        }
        
        /* Map takes full screen */
        [data-testid="stDeckGlJsonChart"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 0 !important;
        }
        
        [data-testid="stDeckGlJsonChart"] > div {
            width: 100% !important;
            height: 100% !important;
        }
        
        [data-testid="stDeckGlJsonChart"] iframe {
            width: 100% !important;
            height: 100% !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 10, 15, 0.95), rgba(18, 18, 26, 0.95)) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdown"] {
            color: #f1f5f9;
        }

        /* Floating header card */
        .floating-header {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 50;
            background: linear-gradient(135deg, rgba(10, 10, 15, 0.9), rgba(18, 18, 26, 0.9));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 16px 32px;
            display: flex;
            align-items: center;
            gap: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        .floating-header .title-section {
            text-align: left;
        }
        
        .floating-header .main-title {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #f97316, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }
        
        .floating-header .subtitle {
            font-size: 11px;
            color: #94a3b8;
            margin: 2px 0 0 0;
        }
        
        .floating-header .stats-row {
            display: flex;
            gap: 24px;
        }
        
        .floating-header .stat-item {
            text-align: center;
        }
        
        .floating-header .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 700;
            color: #f97316;
            line-height: 1;
        }
        
        .floating-header .stat-label {
            font-size: 10px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }

        /* Sidebar form elements */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #f97316, #f59e0b);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] [data-testid="stDateInput"] > div > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }
        
        /* Floating chat wrapper */
        .chat-wrapper {
            background: linear-gradient(180deg, rgba(15, 15, 20, 0.96), rgba(11, 11, 16, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.55);
            color: #e2e8f0;
        }
        
        .chat-wrapper .chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .chat-wrapper .chat-title {
            font-weight: 700;
            font-size: 14px;
            color: #f97316;
            margin: 0;
        }
        
        .chat-wrapper .chat-subtitle {
            font-size: 11px;
            color: #94a3b8;
            margin: 0;
        }
        
        .chat-wrapper .chat-history {
            max-height: 280px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 2px 4px 2px 0;
            margin: 6px 0 10px;
        }
        
        .chat-wrapper .chat-bubble {
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 12px;
            line-height: 1.4;
            width: fit-content;
            max-width: 100%;
            border: 1px solid transparent;
        }
        
        .chat-wrapper .chat-bubble.user {
            margin-left: auto;
            background: linear-gradient(135deg, #f97316, #f59e0b);
            color: #0b0b10;
            border-color: rgba(0, 0, 0, 0.12);
        }
        
        .chat-wrapper .chat-bubble.assistant {
            margin-right: auto;
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
        }
        
        .chat-wrapper form {
            margin-top: 6px;
        }
        
        .chat-wrapper [data-testid="stTextInput"] > div > div {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
        }
        
        .chat-wrapper .stButton > button {
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #f97316, #f59e0b);
            color: #0b0b10;
            font-weight: 700;
            border: none;
            box-shadow: 0 10px 20px rgba(249, 115, 22, 0.35);
        }
        
        [data-testid="stVerticalBlock"] {
            display: flex;
        }

        [data-testid="InputInstructions"] {
            display: none !important;
        }
        
        [data-testid="stForm"] > [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 16px !important;
            background: linear-gradient(145deg, rgba(10, 10, 15, 0.95), rgba(15, 15, 22, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0 0 14px 14px;
            border-top: none !important;
            overflow: hidden;
            backdrop-filter: blur(16px);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_demo_data(n_points: int = 5000) -> pd.DataFrame:
    """Generate demo data for visualization."""
    np.random.seed(42)
    
    lat_center, lon_center = settings.map.center_lat, settings.map.center_lon
    
    # Generate clustered points
    n_clusters = 15
    cluster_centers_lat = np.random.normal(lat_center, 0.05, n_clusters)
    cluster_centers_lon = np.random.normal(lon_center, 0.07, n_clusters)
    
    lats, lons = [], []
    for i in range(n_points):
        cluster_idx = np.random.randint(0, n_clusters)
        lats.append(np.random.normal(cluster_centers_lat[cluster_idx], 0.02))
        lons.append(np.random.normal(cluster_centers_lon[cluster_idx], 0.02))
    
    category_weights = [0.25, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.04, 0.03]
    categories = np.random.choice(
        settings.categories.categories,
        size=n_points,
        p=category_weights,
    )
    
    dates = pd.date_range(end=date.today(), periods=90, freq="D")
    created_dates = np.random.choice(dates, size=n_points)
    descriptions = [f"Report #{i+1} - {cat}" for i, cat in enumerate(categories)]
    statuses = np.random.choice(["pending", "in_progress", "resolved"], size=n_points, p=[0.3, 0.25, 0.45])
    
    return pd.DataFrame({
        "id": range(1, n_points + 1),
        "lat": lats,
        "lon": lons,
        "category": categories,
        "created_at": created_dates,
        "description": descriptions,
        "status": statuses,
    })


def filter_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter data based on sidebar parameters."""
    filtered = df.copy()
    
    if params.get("categories"):
        filtered = filtered[filtered["category"].isin(params["categories"])]
    
    if params.get("date_from"):
        filtered = filtered[pd.to_datetime(filtered["created_at"]).dt.date >= params["date_from"]]
    if params.get("date_to"):
        filtered = filtered[pd.to_datetime(filtered["created_at"]).dt.date <= params["date_to"]]
    
    return filtered


def prepare_data_for_layer(df: pd.DataFrame, layer_type: str, params: dict) -> pd.DataFrame:
    """Prepare data for specific layer type."""
    if df.empty:
        return df
    
    if layer_type == "scatter":
        return sample_data(df, max_points=settings.map.max_points_scatter)
    elif layer_type == "heatmap":
        return sample_data(df, max_points=settings.map.max_points_heatmap)
    elif layer_type in ["hexagon", "cluster"]:
        return aggregate_to_hexagons(df, hex_size=params.get("hexagon_radius", 200) / 111000)
    elif layer_type == "icon":
        return sample_data(df, max_points=1000)
    return df


def render_fullscreen_map(data: pd.DataFrame, layer_type: str, layer_params: dict):
    """Render the full-screen PyDeck map."""
    view_state = get_initial_view_state()
    
    layers = [create_layer(layer_type, data, **layer_params)]
    
    # Tooltip config
    tooltip = None
    if layer_type in ["scatter", "icon"]:
        tooltip = {
            "html": """
                <div style="background:rgba(15,15,20,0.95);padding:12px 16px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);font-family:system-ui;max-width:260px;">
                    <div style="color:#f97316;font-weight:600;font-size:13px;margin-bottom:6px;">{category}</div>
                    <div style="color:#e2e8f0;font-size:12px;line-height:1.4;">{description}</div>
                    <div style="color:#64748b;font-size:11px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.1);">
                        📍 {lat:.4f}, {lon:.4f}
                    </div>
                </div>
            """,
            "style": {"backgroundColor": "transparent", "color": "white"},
        }
    elif layer_type in ["hexagon", "cluster"]:
        tooltip = {
            "html": "<div style='background:rgba(15,15,20,0.95);padding:10px 14px;border-radius:6px;color:#f97316;font-weight:600;'>{elevationValue} reports</div>",
            "style": {"backgroundColor": "transparent"},
        }
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v11",
    )
    
    st.pydeck_chart(deck, use_container_width=True, height=1000)


def render_floating_header(total: int, filtered: int, categories: int, layer: str):
    """Render the floating header with stats."""
    st.markdown(
        f"""
        <div class="floating-header">
            <div class="title-section">
                <div class="main-title">Bangkok Urban Reports</div>
                <div class="subtitle">Traffy Fondue Analytics</div>
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">{total:,}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{filtered:,}</div>
                    <div class="stat-label">Filtered</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{categories}</div>
                    <div class="stat-label">Categories</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{layer.title()}</div>
                    <div class="stat-label">Layer</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    # Inject CSS
    inject_custom_css()
    
    # Initialize and render sidebar
    init_sidebar_state()
    params = render_sidebar()
    
    # Load data
    if "demo_data" not in st.session_state:
        st.session_state.demo_data = generate_demo_data(n_points=8000)
    
    df = st.session_state.demo_data
    filtered_df = filter_data(df, params)
    
    # Get params
    layer_type = params.get("layer_type", "scatter")
    viz_data = prepare_data_for_layer(filtered_df, layer_type, params)
    
    layer_params = {
        "radius": params.get("radius", settings.map.default_point_radius),
        "opacity": params.get("opacity", settings.map.default_opacity),
        "elevation_scale": params.get("elevation_scale", settings.map.default_elevation_scale),
    }
    
    # Render full-screen map (this goes behind everything)
    render_fullscreen_map(viz_data, layer_type, layer_params)
    
    # Render floating header
    render_floating_header(
        total=len(df),
        filtered=len(filtered_df),
        categories=len(params.get("categories", [])),
        layer=layer_type,
    )
    
    render_chatbox()
    


if __name__ == "__main__":
    main()
