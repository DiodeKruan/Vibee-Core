"""Sidebar component with parameter controls and filters."""

from datetime import date
from typing import Dict, List

import streamlit as st

from config.settings import settings
from data.queries import get_available_districts, get_available_subdistricts


# Default data fetch settings
DEFAULT_DATA_DATE_FROM = date(2023, 1, 1)
DEFAULT_DATA_DATE_TO = date(2025, 12, 31)
DEFAULT_MAX_RECORDS = 20000
MAX_RECORDS_LIMIT = 500000


def init_sidebar_state() -> None:
  """Initialize sidebar state with defaults."""
  defaults = {
      "layer_type": settings.map.default_layer,
      "selected_categories": settings.categories.categories.copy(),
      "selected_event_types": settings.longdo.event_types.copy(),
      "data_source": "traffy",  # Default data source: traffy or longdo
      "point_radius": settings.map.default_point_radius,
      "opacity": settings.map.default_opacity,
      "elevation_scale": settings.map.default_elevation_scale,
      "hexagon_radius": settings.map.hexagon_radius,
      "cluster_eps": 200,
      "cluster_min_samples": 20,
      # Data fetch settings (also used for filtering)
      "data_date_from": DEFAULT_DATA_DATE_FROM,
      "data_date_to": DEFAULT_DATA_DATE_TO,
      "max_records": DEFAULT_MAX_RECORDS,
      "reload_data": False,
      # Location filters
      "selected_districts": [],
      "selected_subdistricts": [],
      "available_districts": [],
      "available_subdistricts": [],
  }

  for key, value in defaults.items():
    if key not in st.session_state:
      st.session_state[key] = value


def render_sidebar() -> Dict:
  """
  Render the sidebar with all parameter controls.

  Returns:
      Dictionary with all selected parameters
  """
  init_sidebar_state()

  with st.sidebar:
    # Header
    st.markdown(
        """
            <div style="
                padding: 16px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 20px;
            ">
                <h1 style="
                    font-size: 22px;
                    font-weight: 700;
                    color: #f97316;
                    margin: 0;
                    letter-spacing: -0.5px;
                ">Vibee Analytics</h1>
                <p style="
                    font-size: 12px;
                    color: #94a3b8;
                    margin: 4px 0 0 0;
                ">Bangkok Urban Report Analytics</p>
            </div>
            """,
        unsafe_allow_html=True,
    )

    # Data Source Selection
    st.markdown("##### Data Source")
    data_source_options = {
        "traffy": "🎫 Traffy Fondue",
        "longdo": "🗺️ Longdo Events",
    }

    data_source = st.selectbox(
        "Data Source",
        options=list(data_source_options.keys()),
        format_func=lambda x: data_source_options[x],
        index=list(data_source_options.keys()).index(st.session_state.data_source),
        key="data_source_select",
        label_visibility="collapsed",
    )

    # Handle data source change - clear cached data
    if data_source != st.session_state.data_source:
      st.session_state.data_source = data_source
      if "traffy_data" in st.session_state:
        del st.session_state.traffy_data
      if "longdo_data" in st.session_state:
        del st.session_state.longdo_data
      if "processed_data" in st.session_state:
        del st.session_state.processed_data
      st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Data Settings Section
    st.markdown("##### Data Settings")

    # Data date range
    col1, col2 = st.columns(2)
    with col1:
      data_date_from = st.date_input(
          "Data From",
          value=st.session_state.data_date_from,
          key="data_date_from_input",
          min_value=date(2015, 1, 1),
          max_value=date(2025, 12, 31),
      )
      st.session_state.data_date_from = data_date_from
    with col2:
      data_date_to = st.date_input(
          "Data To",
          value=st.session_state.data_date_to,
          key="data_date_to_input",
          min_value=date(2015, 1, 1),
          max_value=date(2025, 12, 31),
      )
      st.session_state.data_date_to = data_date_to

    # Max records slider
    max_records = st.slider(
        "Max Records",
        min_value=10000,
        max_value=MAX_RECORDS_LIMIT,
        value=st.session_state.max_records,
        step=10000,
        key="max_records_slider",
        format="%d",
    )
    st.session_state.max_records = max_records

    # Reload data button
    if st.button("Reload Data", use_container_width=True, key="reload_data_btn"):
      st.session_state.reload_data = True
      # Clear cached data to force reload
      if "traffy_data" in st.session_state:
        del st.session_state.traffy_data
      if "processed_data" in st.session_state:
        del st.session_state.processed_data
      st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Layer Type Selection
    st.markdown("##### Visualization Layer")
    layer_options = {
        "scatter": "📍 Scatter Points",
        "heatmap": "🔥 Heatmap",
        "hexagon": "⬡ 3D Hexagon",
        "cluster": "⚫ Clusters",
    }

    layer_type = st.selectbox(
        "Layer Type",
        options=list(layer_options.keys()),
        format_func=lambda x: layer_options[x],
        index=list(layer_options.keys()).index(st.session_state.layer_type),
        key="layer_type_select",
        label_visibility="collapsed",
    )
    st.session_state.layer_type = layer_type

    st.markdown("<br>", unsafe_allow_html=True)

    # Layer-specific parameters
    st.markdown("##### Layer Parameters")

    # Common: Opacity
    opacity = st.slider(
        "Opacity",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.opacity,
        step=0.1,
        key="opacity_slider",
    )
    st.session_state.opacity = opacity

    # Scatter/Icon specific: Point radius
    if layer_type in ["scatter", "cluster"]:
      point_radius = st.slider(
          "Point Radius",
          min_value=5,
          max_value=100,
          value=st.session_state.point_radius,
          step=1,
          key="radius_slider",
      )
      st.session_state.point_radius = point_radius

    # Hexagon specific: Elevation and radius
    if layer_type == "hexagon":
      elevation_scale = st.slider(
          "Elevation Scale",
          min_value=1,
          max_value=20,
          value=st.session_state.elevation_scale,
          step=1,
          key="elevation_slider",
      )
      st.session_state.elevation_scale = elevation_scale

      hexagon_radius = st.slider(
          "Hexagon Radius (m)",
          min_value=50,
          max_value=500,
          value=st.session_state.hexagon_radius,
          step=50,
          key="hex_radius_slider",
      )
      st.session_state.hexagon_radius = hexagon_radius

    # Cluster specific: EPS and Min Samples
    if layer_type == "cluster":
      cluster_eps = st.slider(
          "Cluster Radius (m)",
          min_value=50,
          max_value=3000,
          value=st.session_state.cluster_eps,
          step=50,
          key="cluster_eps_slider",
          help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
      )
      st.session_state.cluster_eps = cluster_eps

      cluster_min_samples = st.slider(
          "Min Samples",
          min_value=5,
          max_value=200,
          value=st.session_state.cluster_min_samples,
          step=5,
          key="cluster_min_samples_slider",
          help="The number of samples in a neighborhood for a point to be considered as a core point.",
      )
      st.session_state.cluster_min_samples = cluster_min_samples

    # Heatmap specific: Intensity
    if layer_type == "heatmap":
      st.slider(
          "Heat Radius",
          min_value=10,
          max_value=100,
          value=30,
          step=5,
          key="heat_radius_slider",
      )

    st.markdown("<br>", unsafe_allow_html=True)

    # Category/Event Type Filter section - Chip-style multiselect
    # Show different filters based on data source
    if st.session_state.data_source == "traffy":
      st.markdown("##### Ticket Types")

      # Quick select buttons
      col1, col2 = st.columns(2)
      with col1:
        if st.button("Select All", use_container_width=True, key="select_all_btn"):
          st.session_state.selected_categories = settings.categories.categories.copy()
          st.rerun()
      with col2:
        if st.button("Clear All", use_container_width=True, key="clear_all_btn"):
          st.session_state.selected_categories = []
          st.rerun()

      # Chip-style multiselect for categories
      selected_categories = st.multiselect(
          "Filter by ticket type",
          options=settings.categories.categories,
          default=st.session_state.selected_categories,
          key="category_multiselect",
          placeholder="Search and select types...",
          label_visibility="collapsed",
      )
      st.session_state.selected_categories = selected_categories

      # Show count of selected types
      st.caption(f"{len(selected_categories)} of {len(settings.categories.categories)} types selected")
    else:
      # Longdo event types filter
      st.markdown("##### Event Types")

      # Quick select buttons
      col1, col2 = st.columns(2)
      with col1:
        if st.button("Select All", use_container_width=True, key="select_all_events_btn"):
          st.session_state.selected_event_types = settings.longdo.event_types.copy()
          st.rerun()
      with col2:
        if st.button("Clear All", use_container_width=True, key="clear_all_events_btn"):
          st.session_state.selected_event_types = []
          st.rerun()

      # Chip-style multiselect for event types
      selected_event_types = st.multiselect(
          "Filter by event type",
          options=settings.longdo.event_types,
          default=st.session_state.selected_event_types,
          format_func=lambda x: settings.longdo.event_type_labels.get(x, x),
          key="event_type_multiselect",
          placeholder="Search and select types...",
          label_visibility="collapsed",
      )
      st.session_state.selected_event_types = selected_event_types

      # Show count of selected types
      st.caption(f"{len(selected_event_types)} of {len(settings.longdo.event_types)} types selected")

    st.markdown("<br>", unsafe_allow_html=True)

    # District Filter Section (only for Traffy data)
    if st.session_state.data_source == "traffy":
      st.markdown("##### Location Filter")

      # Load available districts if not already loaded
      if not st.session_state.available_districts:
        try:
          districts_data = get_available_districts()
          st.session_state.available_districts = [d["district"] for d in districts_data if d["district"]]
        except Exception:
          st.session_state.available_districts = []

      if st.session_state.available_districts:
        # District multiselect with search
        selected_districts = st.multiselect(
            "District (เขต)",
            options=st.session_state.available_districts,
            default=st.session_state.selected_districts,
            key="district_multiselect",
            placeholder="Select districts...",
        )

        # Handle district selection change - update subdistricts
        if selected_districts != st.session_state.selected_districts:
          st.session_state.selected_districts = selected_districts
          # Reset subdistrict selection when districts change
          st.session_state.selected_subdistricts = []
          st.session_state.available_subdistricts = []
          st.rerun()

        # Show count
        if selected_districts:
          st.caption(f"{len(selected_districts)} district(s) selected")
        else:
          st.caption("All districts")

        # Subdistrict filter (dependent on selected districts)
        if selected_districts:
          # Load subdistricts for selected districts
          if not st.session_state.available_subdistricts:
            try:
              all_subdistricts = []
              for district in selected_districts:
                subdistricts_data = get_available_subdistricts(district)
                all_subdistricts.extend([
                    d["subdistrict"] for d in subdistricts_data if d["subdistrict"]
                ])
              st.session_state.available_subdistricts = list(set(all_subdistricts))
            except Exception:
              st.session_state.available_subdistricts = []

          if st.session_state.available_subdistricts:
            selected_subdistricts = st.multiselect(
                "Subdistrict (แขวง)",
                options=sorted(st.session_state.available_subdistricts),
                default=st.session_state.selected_subdistricts,
                key="subdistrict_multiselect",
                placeholder="Select subdistricts...",
            )
            st.session_state.selected_subdistricts = selected_subdistricts

            # Show count
            if selected_subdistricts:
              st.caption(f"{len(selected_subdistricts)} subdistrict(s) selected")
            else:
              st.caption("All subdistricts in selected districts")

      st.markdown("<br>", unsafe_allow_html=True)

  # Return current parameters
  params = {
      "layer_type": st.session_state.layer_type,
      "data_source": st.session_state.data_source,
      "categories": st.session_state.selected_categories,
      "event_types": st.session_state.selected_event_types,
      # Location filters
      "districts": st.session_state.selected_districts,
      "subdistricts": st.session_state.selected_subdistricts,
      # Use data date range for both fetch and filter
      "date_from": st.session_state.data_date_from,
      "date_to": st.session_state.data_date_to,
      "data_date_from": st.session_state.data_date_from,
      "data_date_to": st.session_state.data_date_to,
      "max_records": st.session_state.max_records,
      "radius": st.session_state.point_radius,
      "opacity": st.session_state.opacity,
      "elevation_scale": st.session_state.elevation_scale,
      "hexagon_radius": st.session_state.hexagon_radius,
      "cluster_eps": st.session_state.cluster_eps,
      "cluster_min_samples": st.session_state.cluster_min_samples,
  }

  return params


def update_param(key: str, value) -> None:
  """
  Update a sidebar parameter programmatically (for MCP integration).

  Args:
      key: Parameter key to update
      value: New value
  """
  if key in st.session_state:
    st.session_state[key] = value


def get_current_params() -> Dict:
  """
  Get current sidebar parameters without rendering.

  Returns:
      Dictionary with current parameters
  """
  init_sidebar_state()

  return {
      "layer_type": st.session_state.layer_type,
      "data_source": st.session_state.data_source,
      "categories": st.session_state.selected_categories,
      "event_types": st.session_state.selected_event_types,
      # Location filters
      "districts": st.session_state.selected_districts,
      "subdistricts": st.session_state.selected_subdistricts,
      # Use data date range for both fetch and filter
      "date_from": st.session_state.data_date_from,
      "date_to": st.session_state.data_date_to,
      "data_date_from": st.session_state.data_date_from,
      "data_date_to": st.session_state.data_date_to,
      "max_records": st.session_state.max_records,
      "radius": st.session_state.point_radius,
      "opacity": st.session_state.opacity,
      "elevation_scale": st.session_state.elevation_scale,
      "hexagon_radius": st.session_state.hexagon_radius,
      "cluster_eps": st.session_state.cluster_eps,
      "cluster_min_samples": st.session_state.cluster_min_samples,
  }
