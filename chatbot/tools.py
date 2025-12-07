"""LangChain tools wrapping DataMCP and UIMCP interfaces."""

import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.tools import tool

from config.settings import settings
from data.queries import (
    execute_flexible_query,
    get_schema_info,
    query_aggregation,
    query_crosstab,
    query_statistics,
    query_ticket_details,
    query_time_series,
)
from mcp.data_mcp import DataMCP, DataQuery
from mcp.ui_mcp import UIMCP


# Initialize MCP instances
_data_mcp = DataMCP()
_ui_mcp = UIMCP()
_data_mcp.connect()
_ui_mcp.connect()


# =============================================================================
# Schema & Discovery Tools
# =============================================================================


@tool
def get_data_schema() -> str:
    """
    Get the database schema and available data structure.
    
    ALWAYS call this first before querying data to understand:
    - Available columns and their types
    - Valid dimensions for grouping/filtering (type, district, status, org)
    - Sample values for key dimensions
    - Date range of available data
    - Total record count
    
    Use this to plan your queries and understand what data is available.
    
    Returns:
        JSON-formatted schema information including columns, dimensions, and sample values
    """
    try:
        schema = get_schema_info()
        
        # Format as readable string
        output_parts = [
            "=== DATABASE SCHEMA ===",
            f"Table: {schema['table']}",
            f"Description: {schema['description']}",
            "",
            "== Queryable Dimensions ==",
            "Use these for group_by, filters, and analysis:",
            ", ".join(schema['queryable_dimensions']),
            "",
            "== Time Granularities ==",
            ", ".join(schema['time_granularities']),
            "",
        ]
        
        if "date_range" in schema:
            dr = schema["date_range"]
            output_parts.extend([
                "== Data Range ==",
                f"From: {dr.get('min')} to {dr.get('max')}",
                f"Total records: {dr.get('total_records', 'N/A'):,}",
                "",
            ])
        
        if "sample_types" in schema:
            output_parts.extend([
                "== Top Ticket Types (by count) ==",
            ])
            for item in schema["sample_types"][:10]:
                output_parts.append(f"  - {item['value']}: {item['count']:,}")
            output_parts.append("")
        
        if "sample_statuses" in schema:
            output_parts.extend([
                "== Available Statuses ==",
                ", ".join(schema["sample_statuses"]),
                "",
            ])
        
        if "sample_districts" in schema:
            output_parts.extend([
                "== Top Districts (by count) ==",
            ])
            for item in schema["sample_districts"][:10]:
                output_parts.append(f"  - {item['value']}: {item['count']:,}")
        
        return "\n".join(output_parts)
    except Exception as e:
        return f"Error fetching schema: {str(e)}"


# =============================================================================
# Aggregation & Analysis Tools
# =============================================================================


@tool
def get_ticket_counts(
    group_by: str,
    filters: Optional[str] = None,
    order_by: str = "count_desc",
    limit: int = 20,
) -> str:
    """
    Get ticket counts grouped by a dimension.
    
    Use this to answer questions like:
    - "What ticket type has the highest reoccurrence?" -> group_by="type", order_by="count_desc"
    - "Which district has the most reports?" -> group_by="district"
    - "Show me pending tickets by organization" -> group_by="org", filters='{"status": "รับเรื่องแล้ว"}'
    - "How many flooding reports per district?" -> group_by="district", filters='{"type": "น้ำท่วม"}'
    
    Args:
        group_by: Dimension to group by. Valid values: type, district, status, org, province
        filters: Optional JSON string with filters, e.g. '{"type": "ถนน"}' or '{"district": ["คลองเตย", "วัฒนา"]}'
        order_by: Sort order - count_desc (default), count_asc, name_asc, name_desc
        limit: Maximum number of results (default 20)
    
    Returns:
        Formatted table of dimension values and their counts
    """
    try:
        # Parse filters if provided
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                return f"Error: Invalid filter JSON format. Use format like: {'{\"type\": \"ถนน\"}'}"
        
        results = query_aggregation(
            group_by=group_by,
            filters=filter_dict,
            order_by=order_by,
            limit=limit,
        )
        
        if not results:
            return f"No results found for group_by={group_by} with given filters."
        
        # Format as readable table
        output_lines = [
            f"=== Ticket Counts by {group_by.upper()} ===",
            f"(showing top {len(results)} results, ordered by {order_by})",
            "",
        ]
        
        # Calculate max width for alignment
        max_name_len = max(len(str(r["dimension_value"])) for r in results)
        
        for i, row in enumerate(results, 1):
            name = str(row["dimension_value"])
            count = row["count"]
            output_lines.append(f"{i:2}. {name:<{max_name_len}} : {count:,} tickets")
        
        # Add total
        total = sum(r["count"] for r in results)
        output_lines.extend([
            "",
            f"Total shown: {total:,} tickets",
        ])
        
        return "\n".join(output_lines)
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error executing aggregation: {str(e)}"


@tool
def get_statistics(
    dimension: Optional[str] = None,
    dimension_value: Optional[str] = None,
    filters: Optional[str] = None,
) -> str:
    """
    Get statistical summary of tickets.
    
    Use this to answer questions like:
    - "What's the overall ticket statistics?" -> No args needed
    - "What's the completion rate for road issues?" -> dimension="type", dimension_value="ถนน"
    - "Show statistics for Khlong Toei district" -> dimension="district", dimension_value="คลองเตย"
    - "What's the average resolution time?" -> No args (returns avg_resolution_hours)
    
    Args:
        dimension: Optional dimension to filter by (type, district, status, org)
        dimension_value: Value for the dimension filter
        filters: Optional additional JSON filters
    
    Returns:
        Statistical summary including totals, completion rates, resolution times
    """
    try:
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                return "Error: Invalid filter JSON format"
        
        stats = query_statistics(
            dimension=dimension,
            dimension_value=dimension_value,
            filters=filter_dict,
        )
        
        if "error" in stats:
            return f"Error: {stats['error']}"
        
        # Format output
        title = "=== TICKET STATISTICS ==="
        if dimension and dimension_value:
            title = f"=== STATISTICS FOR {dimension.upper()}: {dimension_value} ==="
        
        output_lines = [
            title,
            "",
            f"Total Tickets: {stats['total_tickets']:,}",
            f"Unique Types: {stats['unique_types']}",
            f"Unique Districts: {stats['unique_districts']}",
            f"Unique Organizations: {stats['unique_organizations']}",
            "",
            "== Date Range ==",
            f"From: {stats['date_range']['from']}",
            f"To: {stats['date_range']['to']}",
            "",
            "== Resolution ==",
            f"Completed: {stats['completed_count']:,} ({stats['completion_rate']}%)",
            f"Pending: {stats['pending_count']:,}",
        ]
        
        if stats['avg_resolution_hours']:
            hours = stats['avg_resolution_hours']
            if hours > 24:
                days = hours / 24
                output_lines.append(f"Avg Resolution Time: {days:.1f} days ({hours:.1f} hours)")
            else:
                output_lines.append(f"Avg Resolution Time: {hours:.1f} hours")
        
        return "\n".join(output_lines)
    except Exception as e:
        return f"Error fetching statistics: {str(e)}"


# =============================================================================
# Time Series Tools
# =============================================================================


@tool
def get_time_series(
    granularity: str = "day",
    metric: str = "count",
    group_by: Optional[str] = None,
    filters: Optional[str] = None,
    days_back: int = 30,
    limit: int = 50,
) -> str:
    """
    Get time series data for trend analysis.
    
    Use this to answer questions like:
    - "What's the daily trend of reports?" -> granularity="day"
    - "Show monthly ticket counts" -> granularity="month"
    - "How has flooding reports changed over time?" -> filters='{"type": "น้ำท่วม"}'
    - "Compare ticket trends by type" -> group_by="type"
    - "What's the trend of resolution time?" -> metric="avg_resolution_hours"
    
    Args:
        granularity: Time bucket - hour, day, week, month, year (default: day)
        metric: What to measure - count (default), avg_resolution_hours
        group_by: Optional dimension to break down by (type, district, status, org)
        filters: Optional JSON string with filters
        days_back: How many days to look back (default 30)
        limit: Maximum number of time periods to return (default 50)
    
    Returns:
        Time series data formatted as a table
    """
    try:
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                return "Error: Invalid filter JSON format"
        
        date_from = date.today() - timedelta(days=days_back)
        date_to = date.today()
        
        results = query_time_series(
            granularity=granularity,
            metric=metric,
            group_by=group_by,
            filters=filter_dict,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        
        if not results:
            return f"No time series data found for the specified parameters."
        
        # Format output
        output_lines = [
            f"=== TIME SERIES: {metric.upper()} by {granularity.upper()} ===",
            f"Period: {date_from} to {date_to}",
            "",
        ]
        
        if group_by:
            # Grouped time series
            output_lines.append(f"Grouped by: {group_by}")
            output_lines.append("")
            
            for row in results[:limit]:
                period = row.get("period", "N/A")[:10]  # Truncate timestamp
                group_val = row.get("group_value", "N/A")
                value = row.get(metric, row.get("count", 0))
                if isinstance(value, float):
                    output_lines.append(f"{period} | {group_val}: {value:.1f}")
                else:
                    output_lines.append(f"{period} | {group_val}: {value:,}")
        else:
            # Simple time series
            for row in results[:limit]:
                period = row.get("period", "N/A")[:10]
                value = row.get(metric, row.get("count", 0))
                if isinstance(value, float):
                    output_lines.append(f"{period}: {value:.1f}")
                else:
                    output_lines.append(f"{period}: {value:,}")
        
        return "\n".join(output_lines)
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error executing time series query: {str(e)}"


# =============================================================================
# Cross-Tabulation Tools
# =============================================================================


@tool
def get_crosstab(
    row_dimension: str,
    col_dimension: str,
    metric: str = "count",
    filters: Optional[str] = None,
) -> str:
    """
    Get cross-tabulation of two dimensions.
    
    Use this to answer questions like:
    - "Which districts have the most flooding?" -> row_dimension="district", col_dimension="type", filters='{"type": "น้ำท่วม"}'
    - "Show ticket types by district" -> row_dimension="district", col_dimension="type"
    - "Compare completion rates by organization and type" -> metric="count", then calculate ratios
    - "What types of issues does each organization handle?" -> row_dimension="org", col_dimension="type"
    
    Args:
        row_dimension: Dimension for rows (district, type, org, status)
        col_dimension: Dimension for columns (district, type, org, status)  
        metric: What to measure - count (default), avg_resolution_hours
        filters: Optional JSON string with filters
    
    Returns:
        Cross-tabulation table showing relationships between two dimensions
    """
    try:
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                return "Error: Invalid filter JSON format"
        
        result = query_crosstab(
            row_dimension=row_dimension,
            col_dimension=col_dimension,
            metric=metric,
            filters=filter_dict,
        )
        
        if not result.get("rows"):
            return "No cross-tabulation data found for the specified dimensions."
        
        # Format as a readable table
        rows = result["rows"]
        cols = result["columns"]
        data = result["data"]
        
        output_lines = [
            f"=== CROSSTAB: {row_dimension.upper()} × {col_dimension.upper()} ===",
            f"Metric: {metric}",
            "",
        ]
        
        # Create header row
        col_width = 12
        header = f"{'':20} | " + " | ".join(f"{str(c)[:col_width]:<{col_width}}" for c in cols)
        output_lines.append(header)
        output_lines.append("-" * len(header))
        
        # Data rows
        for i, row_name in enumerate(rows):
            row_values = data[i] if i < len(data) else []
            formatted_vals = []
            for v in row_values:
                if isinstance(v, float):
                    formatted_vals.append(f"{v:.1f}")
                else:
                    formatted_vals.append(f"{int(v):,}")
            
            row_str = f"{str(row_name)[:20]:<20} | " + " | ".join(
                f"{v:<{col_width}}" for v in formatted_vals
            )
            output_lines.append(row_str)
        
        return "\n".join(output_lines)
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error executing crosstab query: {str(e)}"


# =============================================================================
# Flexible Query Builder
# =============================================================================


@tool
def run_analytical_query(
    select_columns: str,
    aggregations: Optional[str] = None,
    filters: Optional[str] = None,
    group_by: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    Run a flexible analytical query for complex questions.
    
    Use this when other tools don't fit, for questions like:
    - Complex multi-dimension aggregations
    - Custom filtering combinations
    - Specific ordering requirements
    
    Args:
        select_columns: Comma-separated columns: type, district, status, org, province, timestamp
        aggregations: JSON array of aggregations, e.g. '[{"function": "count", "column": "*", "alias": "total"}]'
                     Functions: count, count_distinct, sum, avg, min, max
        filters: JSON object with filters, e.g. '{"type": ["ถนน", "น้ำท่วม"], "date_from": "2024-01-01"}'
        group_by: Comma-separated columns to group by
        order_by: Order expression, e.g. "count DESC" or "total ASC"
        limit: Maximum results (default 50)
    
    Returns:
        Query results as formatted table
    
    Example:
        select_columns="type, district"
        aggregations='[{"function": "count", "column": "*", "alias": "total"}]'
        group_by="type, district"
        order_by="total DESC"
    """
    try:
        # Parse inputs
        cols = [c.strip() for c in select_columns.split(",")]
        
        agg_list = None
        if aggregations:
            try:
                agg_list = json.loads(aggregations)
            except json.JSONDecodeError:
                return "Error: Invalid aggregations JSON format"
        
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                return "Error: Invalid filters JSON format"
        
        group_list = None
        if group_by:
            group_list = [g.strip() for g in group_by.split(",")]
        
        results = execute_flexible_query(
            select_columns=cols,
            aggregations=agg_list,
            filters=filter_dict,
            group_by=group_list,
            order_by=order_by,
            limit=limit,
        )
        
        if not results:
            return "No results found for the query."
        
        # Format output
        output_lines = [
            "=== QUERY RESULTS ===",
            f"Returned {len(results)} rows",
            "",
        ]
        
        # Get column names from first result
        if results:
            columns = list(results[0].keys())
            
            # Create header
            header = " | ".join(f"{c[:15]:<15}" for c in columns)
            output_lines.append(header)
            output_lines.append("-" * len(header))
            
            # Data rows
            for row in results:
                values = []
                for c in columns:
                    v = row.get(c, "")
                    if isinstance(v, float):
                        values.append(f"{v:.2f}")
                    elif isinstance(v, int):
                        values.append(f"{v:,}")
                    else:
                        values.append(str(v)[:15])
                output_lines.append(" | ".join(f"{v:<15}" for v in values))
        
        return "\n".join(output_lines)
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


# =============================================================================
# Record Detail Tools
# =============================================================================


@tool
def get_ticket_detail(ticket_id: str) -> str:
    """
    Get detailed information about a specific ticket.
    
    Use this when user asks about a specific ticket by ID.
    
    Args:
        ticket_id: The ticket ID to look up (e.g., "1234567")
    
    Returns:
        Detailed ticket information including description, location, status
    """
    try:
        details = query_ticket_details(ticket_id)
        
        if not details:
            return f"No ticket found with ID: {ticket_id}"
        
        output_lines = [
            f"=== TICKET DETAILS: {ticket_id} ===",
            "",
            f"Type: {details.get('type', 'N/A')}",
            f"Status: {details.get('status', 'N/A')}",
            f"District: {details.get('district', 'N/A')}",
            f"Province: {details.get('province', 'N/A')}",
            f"Organization: {details.get('organization', 'N/A')}",
            "",
            f"Created: {details.get('created_at', 'N/A')}",
            f"Last Activity: {details.get('last_activity', 'N/A')}",
            "",
            f"Address: {details.get('address', 'N/A')}",
            f"Coordinates: {details.get('lat', 'N/A')}, {details.get('lon', 'N/A')}",
            "",
            "Description:",
            details.get('description', 'N/A')[:500],  # Truncate long descriptions
        ]
        
        if details.get('photo_url'):
            output_lines.extend(["", f"Photo: {details['photo_url']}"])
        
        return "\n".join(output_lines)
    except Exception as e:
        return f"Error fetching ticket details: {str(e)}"


# =============================================================================
# Legacy Data Tools (kept for backward compatibility)
# =============================================================================


@tool
def get_report_summary(
    categories: Optional[List[str]] = None,
    days_back: int = 30,
) -> str:
    """
    Get summary statistics about urban reports.
    
    Use this to answer questions like "How many reports are there?" or 
    "What's the total count of road reports?"
    
    Args:
        categories: Optional list of categories to filter by. 
                   Available: PM2.5, กีดขวาง, การเดินทาง, คนจรจัด, คลอง, ความปลอดภัย, 
                   ความสะอาด, จราจร, ต้นไม้, ถนน, ทางเท้า, ท่อระบายน้ำ, น้ำท่วม, ป้าย, 
                   ป้ายจราจร, ร้องเรียน, สอบถาม, สะพาน, สัตว์จรจัด, สายไฟ, ห้องน้ำ, 
                   เสนอแนะ, เสียงรบกวน, แสงสว่าง, ไม่ระบุ
        days_back: Number of days to look back (default 30)
    
    Returns:
        Summary statistics as a formatted string
    """
    query = DataQuery(
        query_type="summary",
        categories=categories,
        date_from=date.today() - timedelta(days=days_back),
        date_to=date.today(),
    )
    response = _data_mcp.execute_query(query)
    
    if not response.success:
        return f"Error: {response.error}"
    
    summary = response.summary or {}
    cat_filter = f" for {', '.join(categories)}" if categories else ""
    
    return (
        f"Report Summary{cat_filter}:\n"
        f"- Total reports: {summary.get('total_reports', 'N/A'):,}\n"
        f"- Categories included: {summary.get('categories', 'N/A')}\n"
        f"- Date range: {summary.get('date_range', 'N/A')}"
    )


@tool
def get_available_categories() -> str:
    """
    List all available report categories.
    
    Use this when user asks "What categories are available?" or 
    "What types of reports can I filter?"
    
    Returns:
        List of all available categories
    """
    categories = settings.categories.categories
    return f"Available categories ({len(categories)} total):\n" + "\n".join(
        f"- {cat}" for cat in categories
    )


@tool
def get_current_data_stats() -> str:
    """
    Get statistics about the currently loaded dataset.
    
    Use this to answer questions about what data is currently visible on the map.
    
    Returns:
        Statistics about the current dataset
    """
    df = st.session_state.get("traffy_data")
    if df is None or df.empty:
        return "No data currently loaded."
    
    total = len(df)
    
    # Get category counts if available
    cat_col = "category" if "category" in df.columns else "type"
    if cat_col in df.columns:
        top_cats = df[cat_col].value_counts().head(5)
        top_cats_str = "\n".join(
            f"  - {cat}: {count:,}" for cat, count in top_cats.items()
        )
    else:
        top_cats_str = "  (category data not available)"
    
    return (
        f"Currently loaded data:\n"
        f"- Total records: {total:,}\n"
        f"- Top 5 categories:\n{top_cats_str}"
    )


# =============================================================================
# UI Tools
# =============================================================================


@tool
def set_visualization_layer(layer: str) -> str:
    """
    Switch the map visualization layer type.
    
    Use when user wants to change how data is displayed, like 
    "show as heatmap" or "switch to scatter plot".
    
    Args:
        layer: Layer type - must be one of: scatter, heatmap, hexagon, cluster
    
    Returns:
        Confirmation message
    """
    valid_layers = settings.layer_types
    if layer not in valid_layers:
        return f"Invalid layer '{layer}'. Choose from: {', '.join(valid_layers)}"
    
    response = _ui_mcp.execute_action(
        _ui_mcp.parse_natural_command(f"switch to {layer}")
    )
    
    if response.success:
        return f"Visualization changed to {layer} layer."
    return f"Error: {response.error}"


@tool
def filter_categories(categories: List[str]) -> str:
    """
    Filter the map to show only specific report categories.
    
    Use when user wants to focus on certain types of reports, like 
    "show only road and flooding reports" or "filter to traffic issues".
    
    Args:
        categories: List of categories to show. 
                   Available: PM2.5, กีดขวาง, การเดินทาง, คนจรจัด, คลอง, ความปลอดภัย, 
                   ความสะอาด, จราจร, ต้นไม้, ถนน, ทางเท้า, ท่อระบายน้ำ, น้ำท่วม, ป้าย, 
                   ป้ายจราจร, ร้องเรียน, สอบถาม, สะพาน, สัตว์จรจัด, สายไฟ, ห้องน้ำ, 
                   เสนอแนะ, เสียงรบกวน, แสงสว่าง, ไม่ระบุ
    
    Returns:
        Confirmation of filter applied
    """
    # Validate categories
    valid = [c for c in categories if c in settings.categories.categories]
    if not valid:
        return (
            f"No valid categories provided. "
            f"Use get_available_categories to see valid options."
        )
    
    from mcp.ui_mcp import UIAction
    response = _ui_mcp.execute_action(
        UIAction(action_type="set_categories", params={"categories": valid})
    )
    
    if response.success:
        return f"Map filtered to show: {', '.join(valid)}"
    return f"Error: {response.error}"


@tool
def set_layer_opacity(opacity: float) -> str:
    """
    Adjust the opacity/transparency of the map visualization layer.
    
    Use when user wants to make the layer more or less transparent, like 
    "make it more transparent" or "increase opacity".
    
    Args:
        opacity: Opacity value from 0.1 (almost transparent) to 1.0 (fully opaque)
    
    Returns:
        Confirmation message
    """
    from mcp.ui_mcp import UIAction
    response = _ui_mcp.execute_action(
        UIAction(action_type="set_opacity", params={"opacity": opacity})
    )
    
    if response.success:
        pct = int(opacity * 100)
        return f"Layer opacity set to {pct}%"
    return f"Error: {response.error}"


@tool
def set_point_radius(radius: int) -> str:
    """
    Adjust the size of points/markers on the map.
    
    Use when user wants bigger or smaller markers, like 
    "make points bigger" or "smaller markers".
    
    Args:
        radius: Point radius in pixels (5-100)
    
    Returns:
        Confirmation message
    """
    from mcp.ui_mcp import UIAction
    response = _ui_mcp.execute_action(
        UIAction(action_type="set_radius", params={"radius": radius})
    )
    
    if response.success:
        return f"Point radius set to {radius} pixels"
    return f"Error: {response.error}"


@tool
def reset_all_filters() -> str:
    """
    Reset all visualization settings to their defaults.
    
    Use when user wants to start fresh, like "reset everything" or 
    "show all data" or "clear filters".
    
    Returns:
        Confirmation message
    """
    from mcp.ui_mcp import UIAction
    response = _ui_mcp.execute_action(UIAction(action_type="reset_filters"))
    
    if response.success:
        return "All filters reset to default. Showing all categories with scatter layer."
    return f"Error: {response.error}"


@tool
def get_current_view_state() -> str:
    """
    Get the current visualization state (layer, filters, etc).
    
    Use to understand what the user is currently viewing, helpful for 
    context when answering questions.
    
    Returns:
        Current UI state description
    """
    state = _ui_mcp.get_current_state()
    
    cats = state.get("selected_categories", [])
    cat_count = len(cats)
    total_cats = len(settings.categories.categories)
    
    return (
        f"Current view state:\n"
        f"- Layer type: {state.get('layer_type', 'scatter')}\n"
        f"- Categories shown: {cat_count}/{total_cats}\n"
        f"- Opacity: {int(state.get('opacity', 0.8) * 100)}%\n"
        f"- Point radius: {state.get('point_radius', 15)}px"
    )


@tool
def highlight_tickets_on_map(ticket_ids: List[str]) -> str:
    """
    Highlight specific tickets on the map.
    
    Use after querying data to visually highlight specific tickets,
    like "show these tickets on the map" or "highlight the top issues".
    
    Args:
        ticket_ids: List of ticket IDs to highlight
    
    Returns:
        Confirmation message
    """
    if not ticket_ids:
        return "No ticket IDs provided to highlight."
    
    # Store highlighted tickets in session state
    st.session_state.highlighted_tickets = ticket_ids
    
    return f"Highlighted {len(ticket_ids)} tickets on the map. Look for them with distinct markers."


@tool
def zoom_to_district(district: str) -> str:
    """
    Zoom the map to focus on a specific district.
    
    Use when user wants to see a specific area, like 
    "show me Khlong Toei district" or "zoom to Bang Rak".
    
    Args:
        district: Bangkok district name (Thai or English)
    
    Returns:
        Confirmation message
    """
    # Bangkok district coordinates (approximate centers)
    district_coords = {
        "คลองเตย": (13.7123, 100.5591),
        "บางรัก": (13.7285, 100.5234),
        "ปทุมวัน": (13.7466, 100.5349),
        "สาทร": (13.7189, 100.5268),
        "บางคอแหลม": (13.6902, 100.5014),
        "ยานนาวา": (13.6956, 100.5389),
        "วัฒนา": (13.7389, 100.5683),
        "คลองสาน": (13.7273, 100.5024),
        "ธนบุรี": (13.7164, 100.4873),
        "บางกอกน้อย": (13.7632, 100.4703),
        "บางกอกใหญ่": (13.7225, 100.4764),
        "ดุสิต": (13.7808, 100.5141),
        "พญาไท": (13.7776, 100.5408),
        "ราชเทวี": (13.7588, 100.5331),
        "ห้วยขวาง": (13.7773, 100.5732),
        "ดินแดง": (13.7711, 100.5584),
        "จตุจักร": (13.8185, 100.5541),
        "บางซื่อ": (13.8066, 100.5178),
        "ลาดพร้าว": (13.8167, 100.5852),
        "บึงกุ่ม": (13.8072, 100.6441),
        "สะพานสูง": (13.7772, 100.6785),
        "คันนายาว": (13.8243, 100.6741),
        "วังทองหลาง": (13.7868, 100.6089),
        "ประเวศ": (13.7201, 100.6472),
        "สวนหลวง": (13.7267, 100.6115),
        "บางกะปิ": (13.7633, 100.6395),
        "พระโขนง": (13.7017, 100.5918),
        "บางนา": (13.6672, 100.6151),
        "ลาดกระบัง": (13.7229, 100.7519),
        "มีนบุรี": (13.8127, 100.7376),
        "หนองจอก": (13.8593, 100.8462),
        "คลองสามวา": (13.8722, 100.7143),
        "สายไหม": (13.9154, 100.6507),
        "ดอนเมือง": (13.9280, 100.5945),
        "หลักสี่": (13.8829, 100.5669),
        "บางเขน": (13.8615, 100.5858),
        "ตลิ่งชัน": (13.7784, 100.4313),
        "ภาษีเจริญ": (13.7217, 100.4247),
        "บางแค": (13.7108, 100.4004),
        "หนองแขม": (13.6944, 100.3519),
        "บางบอน": (13.6549, 100.3755),
        "จอมทอง": (13.6863, 100.4720),
        "ราษฎร์บูรณะ": (13.6682, 100.5022),
        "ทุ่งครุ": (13.6358, 100.5004),
        "บางพลัด": (13.7893, 100.4981),
        "บางขุนเทียน": (13.6107, 100.4378),
        "ทวีวัฒนา": (13.7646, 100.3578),
    }
    
    # Try to find matching district
    coords = district_coords.get(district)
    if not coords:
        # Try partial match
        for d, c in district_coords.items():
            if district.lower() in d.lower() or d.lower() in district.lower():
                coords = c
                district = d
                break
    
    if not coords:
        return f"District '{district}' not found. Try Thai names like คลองเตย, บางรัก, ปทุมวัน."
    
    # Update map view state
    from mcp.ui_mcp import UIAction
    response = _ui_mcp.execute_action(
        UIAction(
            action_type="set_view",
            params={
                "latitude": coords[0],
                "longitude": coords[1],
                "zoom_delta": 3,  # Zoom in
            }
        )
    )
    
    if response.success:
        return f"Map zoomed to {district} district (center: {coords[0]:.4f}, {coords[1]:.4f})"
    return f"Error zooming to district: {response.error}"


@tool
def set_date_filter(days_back: int = 30) -> str:
    """
    Set the date filter for displayed data.
    
    Use when user wants to see data from a specific time period,
    like "show last week" or "show data from last 90 days".
    
    Args:
        days_back: Number of days to look back from today (default 30)
    
    Returns:
        Confirmation message
    """
    date_from = date.today() - timedelta(days=days_back)
    date_to = date.today()
    
    st.session_state.date_from = date_from
    st.session_state.date_to = date_to
    
    return f"Date filter set: {date_from} to {date_to} ({days_back} days)"


# =============================================================================
# Tool Collections
# =============================================================================

# New analytical data tools
analytical_tools = [
    get_data_schema,
    get_ticket_counts,
    get_statistics,
    get_time_series,
    get_crosstab,
    run_analytical_query,
    get_ticket_detail,
]

# Legacy data tools (kept for backward compatibility)
legacy_data_tools = [
    get_report_summary,
    get_available_categories,
    get_current_data_stats,
]

# All data tools
data_tools = analytical_tools + legacy_data_tools

# UI tools
ui_tools = [
    set_visualization_layer,
    filter_categories,
    set_layer_opacity,
    set_point_radius,
    reset_all_filters,
    get_current_view_state,
    highlight_tickets_on_map,
    zoom_to_district,
    set_date_filter,
]

# Combined tool list for agent
all_tools = data_tools + ui_tools

