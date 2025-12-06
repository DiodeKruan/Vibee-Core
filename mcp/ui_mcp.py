"""UI MCP interface for chatbot control of Streamlit parameters."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from config.settings import settings


@dataclass
class UIAction:
    """Represents a UI action from the chatbot."""

    action_type: str  # 'set_layer', 'set_filter', 'set_view', 'toggle_category'
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class UIResponse:
    """Response from UI action."""

    success: bool
    message: str = ""
    state_changed: bool = False
    new_state: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UIMCP:
    """
    UI MCP interface for programmatic control of the Streamlit UI.
    
    This allows the chatbot to:
    - Switch visualization layers
    - Toggle category filters
    - Adjust visualization parameters
    - Navigate the map
    """

    def __init__(self):
        """Initialize UI MCP interface."""
        self._connected = False
        self._action_handlers: Dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up action handlers."""
        self._action_handlers = {
            "set_layer": self._handle_set_layer,
            "set_categories": self._handle_set_categories,
            "toggle_category": self._handle_toggle_category,
            "set_opacity": self._handle_set_opacity,
            "set_radius": self._handle_set_radius,
            "set_view": self._handle_set_view,
            "reset_filters": self._handle_reset_filters,
        }

    def connect(self) -> bool:
        """Connect to the MCP server."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def parse_natural_command(self, text: str) -> UIAction:
        """
        Parse natural language into a UI action.

        Args:
            text: Natural language command

        Returns:
            UIAction object

        Examples:
            "Switch to heatmap" -> UIAction(set_layer, {layer: heatmap})
            "Show only roads" -> UIAction(set_categories, {categories: [roads]})
            "Zoom in" -> UIAction(set_view, {zoom_delta: +1})
        """
        text_lower = text.lower()

        # Layer switching
        for layer in settings.layer_types:
            if layer in text_lower:
                return UIAction(action_type="set_layer", params={"layer": layer})

        # Category toggling
        for category in settings.categories.categories:
            if category.lower() in text_lower:
                if any(word in text_lower for word in ["only", "show only", "just"]):
                    return UIAction(
                        action_type="set_categories",
                        params={"categories": [category]},
                    )
                elif any(word in text_lower for word in ["hide", "remove", "exclude"]):
                    return UIAction(
                        action_type="toggle_category",
                        params={"category": category, "enabled": False},
                    )
                else:
                    return UIAction(
                        action_type="toggle_category",
                        params={"category": category, "enabled": True},
                    )

        # View controls
        if "zoom in" in text_lower:
            return UIAction(action_type="set_view", params={"zoom_delta": 1})
        if "zoom out" in text_lower:
            return UIAction(action_type="set_view", params={"zoom_delta": -1})

        # Reset
        if any(word in text_lower for word in ["reset", "clear", "show all"]):
            return UIAction(action_type="reset_filters")

        # Default: no action
        return UIAction(action_type="unknown")

    def execute_action(self, action: UIAction) -> UIResponse:
        """
        Execute a UI action.

        Args:
            action: UIAction to execute

        Returns:
            UIResponse with result
        """
        if not self._connected:
            return UIResponse(
                success=False,
                error="Not connected to MCP server",
            )

        handler = self._action_handlers.get(action.action_type)
        if handler:
            return handler(action.params)

        return UIResponse(
            success=False,
            error=f"Unknown action type: {action.action_type}",
        )

    def _handle_set_layer(self, params: Dict) -> UIResponse:
        """Handle layer switch action."""
        layer = params.get("layer")
        if layer not in settings.layer_types:
            return UIResponse(
                success=False,
                error=f"Invalid layer type: {layer}",
            )

        st.session_state.layer_type = layer
        return UIResponse(
            success=True,
            message=f"Switched to {layer} layer",
            state_changed=True,
            new_state={"layer_type": layer},
        )

    def _handle_set_categories(self, params: Dict) -> UIResponse:
        """Handle category filter action."""
        categories = params.get("categories", [])

        # Validate categories
        valid = [c for c in categories if c in settings.categories.categories]
        if not valid:
            return UIResponse(
                success=False,
                error="No valid categories specified",
            )

        st.session_state.selected_categories = valid
        return UIResponse(
            success=True,
            message=f"Filtered to categories: {', '.join(valid)}",
            state_changed=True,
            new_state={"selected_categories": valid},
        )

    def _handle_toggle_category(self, params: Dict) -> UIResponse:
        """Handle category toggle action."""
        category = params.get("category")
        enabled = params.get("enabled", True)

        if category not in settings.categories.categories:
            return UIResponse(
                success=False,
                error=f"Invalid category: {category}",
            )

        current = st.session_state.get("selected_categories", [])

        if enabled and category not in current:
            current.append(category)
        elif not enabled and category in current:
            current.remove(category)

        st.session_state.selected_categories = current
        return UIResponse(
            success=True,
            message=f"{'Enabled' if enabled else 'Disabled'} category: {category}",
            state_changed=True,
            new_state={"selected_categories": current},
        )

    def _handle_set_opacity(self, params: Dict) -> UIResponse:
        """Handle opacity change action."""
        opacity = params.get("opacity", 0.8)
        opacity = max(0.1, min(1.0, opacity))

        st.session_state.opacity = opacity
        return UIResponse(
            success=True,
            message=f"Set opacity to {opacity}",
            state_changed=True,
            new_state={"opacity": opacity},
        )

    def _handle_set_radius(self, params: Dict) -> UIResponse:
        """Handle radius change action."""
        radius = params.get("radius", 100)
        radius = max(50, min(500, radius))

        st.session_state.point_radius = radius
        return UIResponse(
            success=True,
            message=f"Set point radius to {radius}",
            state_changed=True,
            new_state={"point_radius": radius},
        )

    def _handle_set_view(self, params: Dict) -> UIResponse:
        """Handle map view change action."""
        zoom_delta = params.get("zoom_delta", 0)
        lat = params.get("latitude")
        lon = params.get("longitude")

        current_view = st.session_state.get("map_view_state", {})

        if zoom_delta:
            current_zoom = current_view.get("zoom", settings.map.default_zoom)
            new_zoom = max(1, min(20, current_zoom + zoom_delta))
            current_view["zoom"] = new_zoom

        if lat is not None:
            current_view["latitude"] = lat
        if lon is not None:
            current_view["longitude"] = lon

        st.session_state.map_view_state = current_view
        return UIResponse(
            success=True,
            message="Updated map view",
            state_changed=True,
            new_state={"map_view_state": current_view},
        )

    def _handle_reset_filters(self, params: Dict) -> UIResponse:
        """Handle filter reset action."""
        st.session_state.selected_categories = settings.categories.categories.copy()
        st.session_state.layer_type = settings.map.default_layer
        st.session_state.opacity = settings.map.default_opacity
        st.session_state.point_radius = settings.map.default_point_radius

        return UIResponse(
            success=True,
            message="Reset all filters to default",
            state_changed=True,
        )

    def get_available_actions(self) -> List[Dict[str, str]]:
        """
        Get list of available UI actions.

        Returns:
            List of action descriptions
        """
        return [
            {
                "action": "set_layer",
                "description": "Switch visualization layer",
                "params": ["layer"],
                "examples": ["Switch to heatmap", "Show hexagon view"],
            },
            {
                "action": "set_categories",
                "description": "Set active category filters",
                "params": ["categories"],
                "examples": ["Show only roads", "Filter to flooding and trash"],
            },
            {
                "action": "toggle_category",
                "description": "Enable/disable a single category",
                "params": ["category", "enabled"],
                "examples": ["Hide roads", "Show flooding reports"],
            },
            {
                "action": "set_opacity",
                "description": "Adjust layer opacity",
                "params": ["opacity"],
                "examples": ["Make it more transparent", "Increase opacity"],
            },
            {
                "action": "set_radius",
                "description": "Adjust point/hexagon radius",
                "params": ["radius"],
                "examples": ["Make points bigger", "Smaller markers"],
            },
            {
                "action": "set_view",
                "description": "Control map viewport",
                "params": ["zoom_delta", "latitude", "longitude"],
                "examples": ["Zoom in", "Go to center"],
            },
            {
                "action": "reset_filters",
                "description": "Reset all filters to default",
                "params": [],
                "examples": ["Reset", "Show all", "Clear filters"],
            },
        ]

    def natural_command(self, text: str) -> UIResponse:
        """
        Process a natural language command end-to-end.

        Args:
            text: Natural language command

        Returns:
            UIResponse with result
        """
        action = self.parse_natural_command(text)
        return self.execute_action(action)

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current UI state.

        Returns:
            Dictionary of current UI state values
        """
        return {
            "layer_type": st.session_state.get("layer_type", settings.map.default_layer),
            "selected_categories": st.session_state.get(
                "selected_categories", settings.categories.categories
            ),
            "opacity": st.session_state.get("opacity", settings.map.default_opacity),
            "point_radius": st.session_state.get("point_radius", settings.map.default_point_radius),
            "date_from": st.session_state.get("date_from"),
            "date_to": st.session_state.get("date_to"),
        }


# Singleton instance
ui_mcp = UIMCP()

