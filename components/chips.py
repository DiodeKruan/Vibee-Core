"""Filter chip components for category selection."""

from typing import Callable, List, Optional

import streamlit as st

from config.settings import settings


def render_chips(
    selected: Optional[List[str]] = None,
    on_change: Optional[Callable[[List[str]], None]] = None,
) -> List[str]:
    """
    Render horizontal filter chips for category selection.

    Args:
        selected: Currently selected categories
        on_change: Callback when selection changes

    Returns:
        List of selected category names
    """
    if selected is None:
        selected = st.session_state.get(
            "selected_categories", settings.categories.categories.copy()
        )

    # CSS for chip styling
    st.markdown(
        """
        <style>
        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 12px 0;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }
        .chip-active {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.2), rgba(249, 115, 22, 0.1));
            border-color: rgba(249, 115, 22, 0.5);
            color: #f97316;
        }
        .chip-inactive {
            background: rgba(100, 100, 100, 0.1);
            border-color: rgba(100, 100, 100, 0.3);
            color: #94a3b8;
        }
        .chip:hover {
            transform: translateY(-1px);
        }
        .chip-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create chip container
    chip_cols = st.columns(len(settings.categories.categories))

    new_selected = []
    for i, category in enumerate(settings.categories.categories):
        color = settings.categories.colors.get(category, (100, 100, 100, 200))
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        is_active = category in selected

        with chip_cols[i]:
            if st.button(
                category,
                key=f"chip_{category}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                # Toggle selection
                if is_active:
                    selected = [c for c in selected if c != category]
                else:
                    selected = selected + [category]

                st.session_state.selected_categories = selected
                if on_change:
                    on_change(selected)
                st.rerun()

            if is_active:
                new_selected.append(category)

    return selected


def render_compact_chips() -> List[str]:
    """
    Render compact inline chips above the map.

    Returns:
        List of selected category names
    """
    selected = st.session_state.get(
        "selected_categories", settings.categories.categories.copy()
    )

    # Build HTML for chips
    chips_html = '<div class="chip-container">'
    for category in settings.categories.categories:
        color = settings.categories.colors.get(category, (100, 100, 100, 200))
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        is_active = category in selected
        chip_class = "chip chip-active" if is_active else "chip chip-inactive"

        chips_html += f"""
            <span class="{chip_class}">
                <span class="chip-dot" style="background-color: {color_hex};"></span>
                {category}
            </span>
        """
    chips_html += "</div>"

    st.markdown(chips_html, unsafe_allow_html=True)

    return selected

