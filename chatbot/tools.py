"""LangChain tools wrapping DataMCP and UIMCP interfaces."""

import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.tools import tool

from config.settings import settings
from mcp.data_mcp import DataMCP, DataQuery, DataResponse
from mcp.ui_mcp import UIMCP, UIAction


# Initialize MCP instances
_data_mcp = DataMCP()
_ui_mcp = UIMCP()
_data_mcp.connect()
_ui_mcp.connect()


# =============================================================================
# Response Formatters
# =============================================================================


def _format_aggregation_response(response: DataResponse, group_by: str) -> str:
  """Format aggregation query response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  results = response.data
  if not results:
    return f"No results found for group_by={group_by}."

  output_lines = [
      f"=== Ticket Counts by {group_by.upper()} ===",
      f"(showing {len(results)} results)",
      "",
  ]

  max_name_len = max(len(str(r["dimension_value"])) for r in results)

  for i, row in enumerate(results, 1):
    name = str(row["dimension_value"])
    count = row["count"]
    output_lines.append(f"{i:2}. {name:<{max_name_len}} : {count:,} tickets")

  total = sum(r["count"] for r in results)
  output_lines.extend(["", f"Total shown: {total:,} tickets"])

  return "\n".join(output_lines)


def _format_statistics_response(response: DataResponse, title: str = "TICKET STATISTICS") -> str:
  """Format statistics response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  stats = response.data
  if "error" in stats:
    return f"Error: {stats['error']}"

  output_lines = [
      f"=== {title} ===",
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


def _format_time_series_response(response: DataResponse, granularity: str, metric: str, group_by: Optional[str] = None) -> str:
  """Format time series response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  results = response.data
  if not results:
    return "No time series data found for the specified parameters."

  output_lines = [
      f"=== TIME SERIES: {metric.upper()} by {granularity.upper()} ===",
      "",
  ]

  if group_by:
    output_lines.append(f"Grouped by: {group_by}")
    output_lines.append("")

    for row in results:
      period = row.get("period", "N/A")[:10]
      group_val = row.get("group_value", "N/A")
      value = row.get(metric, row.get("count", 0))
      if isinstance(value, float):
        output_lines.append(f"{period} | {group_val}: {value:.1f}")
      else:
        output_lines.append(f"{period} | {group_val}: {value:,}")
  else:
    for row in results:
      period = row.get("period", row.get("time_bucket", "N/A"))[:10]
      value = row.get(metric, row.get("count", 0))
      if isinstance(value, float):
        output_lines.append(f"{period}: {value:.1f}")
      else:
        output_lines.append(f"{period}: {value:,}")

  return "\n".join(output_lines)


def _format_longdo_statistics_response(response: DataResponse, dimension: Optional[str] = None, dimension_value: Optional[str] = None) -> str:
  """Format Longdo statistics response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  stats = response.data
  if "error" in stats:
    return f"Error: {stats['error']}"

  title = "LONGDO EVENT STATISTICS"
  if dimension and dimension_value:
    title = f"LONGDO STATISTICS FOR {dimension.upper()}: {dimension_value}"

  output_lines = [
      f"=== {title} ===",
      "",
      f"Total Events: {stats.get('total_events', 0):,}",
      f"Unique Event Types: {stats.get('unique_types', 0)}",
      f"Unique Posters: {stats.get('unique_posters', 0)}",
      "",
      "== Date Range ==",
      f"From: {stats.get('date_range', {}).get('from', 'N/A')}",
      f"To: {stats.get('date_range', {}).get('to', 'N/A')}",
      "",
      "== Event Status ==",
      f"With End Time: {stats.get('with_end_time', 0):,}",
      f"Ongoing/No End Time: {stats.get('ongoing', 0):,}",
  ]

  return "\n".join(output_lines)


def _format_crosstab_response(response: DataResponse) -> str:
  """Format crosstab response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  result = response.data
  if not result.get("rows"):
    return "No cross-tabulation data found for the specified dimensions."

  rows = result["rows"]
  cols = result["columns"]
  data = result["data"]

  output_lines = [
      f"=== CROSSTAB: {result['row_dimension'].upper()} × {result['col_dimension'].upper()} ===",
      f"Metric: {result['metric']}",
      "",
  ]

  col_width = 12
  header = f"{'':20} | " + " | ".join(f"{str(c)[:col_width]:<{col_width}}" for c in cols)
  output_lines.append(header)
  output_lines.append("-" * len(header))

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


def _format_flexible_response(response: DataResponse) -> str:
  """Format flexible query response as readable text."""
  if not response.success:
    return f"Error: {response.error}"

  results = response.data
  if not results:
    return "No results found for the query."

  output_lines = [
      "=== QUERY RESULTS ===",
      f"Returned {len(results)} rows",
      "",
  ]

  columns = list(results[0].keys())

  header = " | ".join(f"{c[:15]:<15}" for c in columns)
  output_lines.append(header)
  output_lines.append("-" * len(header))

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


# =============================================================================
# Schema & Discovery Tools
# =============================================================================


def _get_current_data_source() -> str:
  """Get the current data source from session state."""
  return st.session_state.get("data_source", "traffy")


@tool
def get_data_schema(data_source: Optional[str] = None) -> str:
  """
  Get the database schema and available data structure.

  ALWAYS call this first before querying data to understand:
  - Available columns and their types
  - Valid dimensions for grouping/filtering
  - Sample values for key dimensions
  - Date range of available data
  - Total record count

  Use this to plan your queries and understand what data is available.

  Args:
      data_source: Data source to query - "traffy" (Traffy Fondue reports) or "longdo" (Longdo Traffic events).
                  If not specified, uses the current data source from the UI.

  Returns:
      JSON-formatted schema information including columns, dimensions, and sample values
  """
  source = data_source or _get_current_data_source()
  query = DataQuery(query_type="schema", data_source=source)
  response = _data_mcp.execute_query(query)

  if not response.success:
    return f"Error fetching schema: {response.error}"

  schema = response.data

  output_parts = [
      f"=== {'LONGDO EVENTS' if source == 'longdo' else 'TRAFFY FONDUE'} SCHEMA ===",
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
      "== Spatial Filters ==",
      "- bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)",
  ]

  if source != "longdo":
    output_parts.append("- districts: Filter by district name(s)")

  output_parts.append("")

  if "date_range" in schema:
    dr = schema["date_range"]
    output_parts.extend([
        "== Data Range ==",
        f"From: {dr.get('min')} to {dr.get('max')}",
        f"Total records: {dr.get('total_records', 'N/A'):,}",
        "",
    ])

  if "sample_types" in schema:
    type_label = "Event Types" if source == "longdo" else "Ticket Types"
    output_parts.extend([f"== Top {type_label} (by count) =="])
    for item in schema["sample_types"][:10]:
      output_parts.append(f"  - {item['value']}: {item['count']:,}")
    output_parts.append("")

  if "event_types" in schema:
    output_parts.extend([
        "== Available Event Types ==",
        ", ".join(schema["event_types"]),
        "",
    ])

  if "sample_statuses" in schema:
    output_parts.extend([
        "== Available Statuses ==",
        ", ".join(schema["sample_statuses"]),
        "",
    ])

  if "sample_districts" in schema:
    output_parts.extend(["== Top Districts (by count) =="])
    for item in schema["sample_districts"][:10]:
      output_parts.append(f"  - {item['value']}: {item['count']:,}")

  if "sample_posters" in schema:
    output_parts.extend(["== Top Posters (by count) =="])
    for item in schema["sample_posters"][:10]:
      output_parts.append(f"  - {item['value']}: {item['count']:,}")

  return "\n".join(output_parts)


# =============================================================================
# Aggregation & Analysis Tools
# =============================================================================


@tool
def get_ticket_counts(
    group_by: str,
    data_source: Optional[str] = None,
    filters: Optional[str] = None,
    districts: Optional[str] = None,
    bbox: Optional[str] = None,
    order_by: str = "count_desc",
    limit: int = 20,
) -> str:
  """
  Get ticket/event counts grouped by a dimension.

  Use this to answer questions like:
  - "What ticket type has the highest reoccurrence?" -> group_by="type", order_by="count_desc"
  - "Which district has the most reports?" -> group_by="district"
  - "Show me pending tickets by organization" -> group_by="org", filters='{"status": "รับเรื่องแล้ว"}'
  - "How many flooding reports per district?" -> group_by="district", filters='{"type": "น้ำท่วม"}'
  - "Count tickets in Khlong Toei" -> group_by="type", districts="คลองเตย"
  - "Show types in this area" -> group_by="type", bbox="100.5,13.7,100.6,13.8"

  For Longdo events:
  - "What event types are most common?" -> group_by="type", data_source="longdo"
  - "Show traffic jam events" -> group_by="type", filters='{"type": "trafficjam"}', data_source="longdo"

  Args:
      group_by: Dimension to group by.
                For Traffy: type, district, status, org, province
                For Longdo: type, event_type, posted_by
      data_source: "traffy" (default) or "longdo". Uses current UI selection if not specified.
      filters: Optional JSON string with filters, e.g. '{"type": "ถนน"}' or '{"event_type": "trafficjam"}'
      districts: Optional comma-separated district names to filter by (Traffy only)
      bbox: Optional bounding box as "min_lon,min_lat,max_lon,max_lat"
      order_by: Sort order - count_desc (default), count_asc, name_asc, name_desc
      limit: Maximum number of results (default 20)

  Returns:
      Formatted table of dimension values and their counts
  """
  source = data_source or _get_current_data_source()

  # Parse filters
  filter_dict = None
  if filters:
    try:
      filter_dict = json.loads(filters)
    except json.JSONDecodeError:
      return f"Error: Invalid filter JSON format. Use format like: {'{\"type\": \"ถนน\"}'}"

  # Parse districts (only for Traffy)
  district_list = None
  if districts and source != "longdo":
    district_list = [d.strip() for d in districts.split(",")]

  # Parse bbox
  bbox_tuple = None
  if bbox:
    try:
      parts = [float(x.strip()) for x in bbox.split(",")]
      if len(parts) == 4:
        bbox_tuple = tuple(parts)
      else:
        return "Error: bbox must have 4 values: min_lon,min_lat,max_lon,max_lat"
    except ValueError:
      return "Error: bbox values must be numbers"

  query = DataQuery(
      query_type="aggregation",
      data_source=source,
      group_by=group_by,
      filters=filter_dict,
      districts=district_list,
      bbox=bbox_tuple,
      order_by=order_by,
      limit=limit,
  )

  response = _data_mcp.execute_query(query)
  label = "Events" if source == "longdo" else "Tickets"
  return _format_aggregation_response(response, group_by).replace("tickets", label.lower()).replace("Ticket", label)


@tool
def get_statistics(
    data_source: Optional[str] = None,
    dimension: Optional[str] = None,
    dimension_value: Optional[str] = None,
    filters: Optional[str] = None,
    districts: Optional[str] = None,
    bbox: Optional[str] = None,
) -> str:
  """
  Get statistical summary of tickets or events.

  Use this to answer questions like:
  - "What's the overall ticket statistics?" -> No args needed
  - "What's the completion rate for road issues?" -> dimension="type", dimension_value="ถนน"
  - "Show statistics for Khlong Toei district" -> dimension="district", dimension_value="คลองเตย"
  - "What's the average resolution time?" -> No args (returns avg_resolution_hours)
  - "Stats for this area" -> bbox="100.5,13.7,100.6,13.8"

  For Longdo events:
  - "Longdo event statistics" -> data_source="longdo"
  - "Traffic jam stats from Longdo" -> data_source="longdo", dimension="type", dimension_value="trafficjam"

  Args:
      data_source: "traffy" (default) or "longdo". Uses current UI selection if not specified.
      dimension: Optional dimension to filter by (type, district, status, org for Traffy; type, posted_by for Longdo)
      dimension_value: Value for the dimension filter
      filters: Optional additional JSON filters
      districts: Optional comma-separated district names (Traffy only)
      bbox: Optional bounding box as "min_lon,min_lat,max_lon,max_lat"

  Returns:
      Statistical summary including totals, completion rates, resolution times (Traffy) or event counts (Longdo)
  """
  source = data_source or _get_current_data_source()

  # Parse filters
  filter_dict = None
  if filters:
    try:
      filter_dict = json.loads(filters)
    except json.JSONDecodeError:
      return "Error: Invalid filter JSON format"

  # Parse districts (only for Traffy)
  district_list = None
  if districts and source != "longdo":
    district_list = [d.strip() for d in districts.split(",")]

  # Parse bbox
  bbox_tuple = None
  if bbox:
    try:
      parts = [float(x.strip()) for x in bbox.split(",")]
      if len(parts) == 4:
        bbox_tuple = tuple(parts)
      else:
        return "Error: bbox must have 4 values: min_lon,min_lat,max_lon,max_lat"
    except ValueError:
      return "Error: bbox values must be numbers"

  query = DataQuery(
      query_type="statistics",
      data_source=source,
      dimension=dimension,
      dimension_value=dimension_value,
      filters=filter_dict,
      districts=district_list,
      bbox=bbox_tuple,
  )

  response = _data_mcp.execute_query(query)

  if source == "longdo":
    return _format_longdo_statistics_response(response, dimension, dimension_value)

  title = "TICKET STATISTICS"
  if dimension and dimension_value:
    title = f"STATISTICS FOR {dimension.upper()}: {dimension_value}"

  return _format_statistics_response(response, title)


# =============================================================================
# Time Series Tools
# =============================================================================


@tool
def get_time_series(
    granularity: str = "day",
    data_source: Optional[str] = None,
    metric: str = "count",
    group_by: Optional[str] = None,
    filters: Optional[str] = None,
    districts: Optional[str] = None,
    bbox: Optional[str] = None,
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
  - "Daily trend in Khlong Toei" -> districts="คลองเตย"

  For Longdo events:
  - "Longdo daily event trend" -> data_source="longdo", granularity="day"
  - "Traffic jam trend from Longdo" -> data_source="longdo", filters='{"type": "trafficjam"}'

  Args:
      granularity: Time bucket - hour, day, week, month, year (default: day)
      data_source: "traffy" (default) or "longdo". Uses current UI selection if not specified.
      metric: What to measure - count (default), avg_resolution_hours (Traffy only)
      group_by: Optional dimension to break down by (type, district, status, org for Traffy; type for Longdo)
      filters: Optional JSON string with filters
      districts: Optional comma-separated district names (Traffy only)
      bbox: Optional bounding box as "min_lon,min_lat,max_lon,max_lat"
      days_back: How many days to look back (default 30)
      limit: Maximum number of time periods to return (default 50)

  Returns:
      Time series data formatted as a table
  """
  source = data_source or _get_current_data_source()

  # Parse filters
  filter_dict = None
  if filters:
    try:
      filter_dict = json.loads(filters)
    except json.JSONDecodeError:
      return "Error: Invalid filter JSON format"

  # Parse districts (only for Traffy)
  district_list = None
  if districts and source != "longdo":
    district_list = [d.strip() for d in districts.split(",")]

  # Parse bbox
  bbox_tuple = None
  if bbox:
    try:
      parts = [float(x.strip()) for x in bbox.split(",")]
      if len(parts) == 4:
        bbox_tuple = tuple(parts)
      else:
        return "Error: bbox must have 4 values"
    except ValueError:
      return "Error: bbox values must be numbers"

  query = DataQuery(
      query_type="time_series",
      data_source=source,
      granularity=granularity,
      metric=metric,
      group_by=group_by,
      filters=filter_dict,
      districts=district_list,
      bbox=bbox_tuple,
      days_back=days_back,
      limit=limit,
  )

  response = _data_mcp.execute_query(query)
  return _format_time_series_response(response, granularity, metric, group_by)


# =============================================================================
# Cross-Tabulation Tools
# =============================================================================


@tool
def get_crosstab(
    row_dimension: str,
    col_dimension: str,
    metric: str = "count",
    filters: Optional[str] = None,
    bbox: Optional[str] = None,
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
      bbox: Optional bounding box as "min_lon,min_lat,max_lon,max_lat"

  Returns:
      Cross-tabulation table showing relationships between two dimensions
  """
  # Parse filters
  filter_dict = None
  if filters:
    try:
      filter_dict = json.loads(filters)
    except json.JSONDecodeError:
      return "Error: Invalid filter JSON format"

  # Parse bbox
  bbox_tuple = None
  if bbox:
    try:
      parts = [float(x.strip()) for x in bbox.split(",")]
      if len(parts) == 4:
        bbox_tuple = tuple(parts)
      else:
        return "Error: bbox must have 4 values"
    except ValueError:
      return "Error: bbox values must be numbers"

  query = DataQuery(
      query_type="crosstab",
      row_dimension=row_dimension,
      col_dimension=col_dimension,
      metric=metric,
      filters=filter_dict,
      bbox=bbox_tuple,
  )

  response = _data_mcp.execute_query(query)
  return _format_crosstab_response(response)


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
    bbox: Optional[str] = None,
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
      bbox: Optional bounding box as "min_lon,min_lat,max_lon,max_lat"
      limit: Maximum results (default 50)

  Returns:
      Query results as formatted table

  Example:
      select_columns="type, district"
      aggregations='[{"function": "count", "column": "*", "alias": "total"}]'
      group_by="type, district"
      order_by="total DESC"
  """
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

  # Parse bbox
  bbox_tuple = None
  if bbox:
    try:
      parts = [float(x.strip()) for x in bbox.split(",")]
      if len(parts) == 4:
        bbox_tuple = tuple(parts)
      else:
        return "Error: bbox must have 4 values"
    except ValueError:
      return "Error: bbox values must be numbers"

  query = DataQuery(
      query_type="flexible",
      select_columns=cols,
      aggregations=agg_list,
      filters=filter_dict,
      group_by=group_by,
      order_by=order_by,
      bbox=bbox_tuple,
      limit=limit,
  )

  response = _data_mcp.execute_query(query)
  return _format_flexible_response(response)


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
  query = DataQuery(
      query_type="ticket_detail",
      ticket_id=ticket_id,
  )

  response = _data_mcp.execute_query(query)

  if not response.success:
    return f"Error: {response.error}"

  details = response.data

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
      str(details.get('description', 'N/A'))[:500],
  ]

  if details.get('photo_url'):
    output_lines.extend(["", f"Photo: {details['photo_url']}"])

  return "\n".join(output_lines)


# =============================================================================
# Spatial Filter Tools
# =============================================================================


@tool
def filter_by_district(districts: str) -> str:
  """
  Get ticket data filtered by specific Bangkok district(s).

  Use when user wants to see data for specific districts, like:
  - "Show me reports from Khlong Toei"
  - "What's happening in Bang Rak and Pathum Wan?"

  Args:
      districts: Comma-separated district names (Thai), e.g. "คลองเตย" or "คลองเตย,บางรัก,ปทุมวัน"

  Returns:
      Summary statistics for the specified district(s)
  """
  district_list = [d.strip() for d in districts.split(",")]

  query = DataQuery(
      query_type="statistics",
      districts=district_list,
  )

  response = _data_mcp.execute_query(query)

  title = f"STATISTICS FOR DISTRICT(S): {', '.join(district_list)}"
  return _format_statistics_response(response, title)


@tool
def filter_by_bounding_box(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
  """
  Get ticket data filtered by a geographic bounding box area.

  Use when user wants to see data in a specific map area, like:
  - "Show me reports in this area" (after getting coordinates)
  - "What issues are near Siam Paragon?" (use known coordinates)

  Args:
      min_lon: Minimum longitude (west boundary), e.g. 100.52
      min_lat: Minimum latitude (south boundary), e.g. 13.72
      max_lon: Maximum longitude (east boundary), e.g. 100.56
      max_lat: Maximum latitude (north boundary), e.g. 13.76

  Returns:
      Summary statistics for tickets within the bounding box
  """
  bbox = (min_lon, min_lat, max_lon, max_lat)

  query = DataQuery(
      query_type="statistics",
      bbox=bbox,
  )

  response = _data_mcp.execute_query(query)

  title = f"STATISTICS FOR AREA ({min_lon:.4f},{min_lat:.4f}) to ({max_lon:.4f},{max_lat:.4f})"
  return _format_statistics_response(response, title)


@tool
def get_tickets_in_area(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    categories: Optional[str] = None,
    limit: int = 100,
) -> str:
  """
  Get list of tickets within a geographic bounding box.

  Use when user wants to see specific tickets in an area, like:
  - "List the tickets near me"
  - "What flooding reports are in this area?"

  Args:
      min_lon: Minimum longitude (west boundary)
      min_lat: Minimum latitude (south boundary)
      max_lon: Maximum longitude (east boundary)
      max_lat: Maximum latitude (north boundary)
      categories: Optional comma-separated category types to filter
      limit: Maximum tickets to return (default 100)

  Returns:
      List of tickets in the area with basic details
  """
  from data.queries import query_by_bbox

  bbox = (min_lon, min_lat, max_lon, max_lat)

  category_list = None
  if categories:
    category_list = [c.strip() for c in categories.split(",")]

  try:
    results = query_by_bbox(
        bbox=bbox,
        categories=category_list,
        limit=limit,
    )

    if not results:
      return f"No tickets found in the area ({min_lon:.4f},{min_lat:.4f}) to ({max_lon:.4f},{max_lat:.4f})"

    output_lines = [
        f"=== TICKETS IN AREA ===",
        f"Bounding box: ({min_lon:.4f},{min_lat:.4f}) to ({max_lon:.4f},{max_lat:.4f})",
        f"Found: {len(results)} tickets",
        "",
    ]

    for i, ticket in enumerate(results[:20], 1):
      output_lines.append(
          f"{i}. [{ticket.get('ticket_id', 'N/A')}] {ticket.get('type', 'N/A')} - "
          f"{ticket.get('status', 'N/A')} ({ticket.get('district', 'N/A')})"
      )

    if len(results) > 20:
      output_lines.append(f"... and {len(results) - 20} more tickets")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error fetching tickets: {str(e)}"


@tool
def get_available_districts() -> str:
  """
  Get list of all available Bangkok districts and their ticket counts.

  Use when user wants to know what districts are available, like:
  - "What districts are available?"
  - "List all districts"

  Returns:
      List of districts with their ticket counts
  """
  from data.queries import get_available_districts as fetch_districts

  try:
    districts = fetch_districts()

    if not districts:
      return "No districts found in the database."

    output_lines = [
        "=== AVAILABLE DISTRICTS ===",
        f"Total: {len(districts)} districts",
        "",
    ]

    for i, d in enumerate(districts, 1):
      output_lines.append(f"{i:2}. {d['district']}: {d['count']:,} tickets")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error fetching districts: {str(e)}"


@tool
def search_traffy_reports(
    categories: Optional[str] = None,
    districts: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    bbox: Optional[str] = None,
    search_text: Optional[str] = None,
    exclude_completed: bool = False,
    limit: int = 20,
) -> str:
  """
  Search and list Traffy reports with detailed location information.

  Use this to answer questions like:
  - "What footpath problems are there near Asok?" -> districts="วัฒนา", categories="ทางเท้า"
  - "Show me unfixed flooding in Khlong Toei" -> districts="คลองเตย", categories="น้ำท่วม", exclude_completed=True
  - "List road issues I should avoid" -> categories="ถนน", exclude_completed=True
  - "Where are the drainage problems?" -> categories="ท่อระบายน้ำ"
  - "Show me broken sidewalks in this area" -> categories="ทางเท้า", bbox="100.55,13.72,100.58,13.76"

  This returns actual locations and addresses, making it useful for:
  - Telling users where specific problems are located
  - Showing which areas to avoid
  - Getting detailed information about urban issues

  Args:
      categories: Comma-separated report types (e.g., "ทางเท้า,ถนน")
                 Common types: ทางเท้า (sidewalk), ถนน (road), น้ำท่วม (flood),
                 ท่อระบายน้ำ (drainage), จราจร (traffic), ความสะอาด (cleanliness),
                 แสงสว่าง (lighting), ต้นไม้ (trees), กีดขวาง (obstruction)
      districts: Comma-separated district names (e.g., "วัฒนา,คลองเตย")
                For Asok area, use "วัฒนา". Other common: คลองเตย, ปทุมวัน, พระโขนง
      status: Comma-separated statuses. Options: เสร็จสิ้น (fixed), รับเรื่องแล้ว (received),
              กำลังดำเนินการ (in progress)
      date_from: Start date in YYYY-MM-DD format
      date_to: End date in YYYY-MM-DD format
      bbox: Bounding box as "min_lon,min_lat,max_lon,max_lat" for area search
      search_text: Search in description/address text
      exclude_completed: If True, only show unfixed/pending reports (default False)
      limit: Maximum reports to return (default 20)

  Returns:
      Detailed list of reports with addresses, coordinates, and status
  """
  from data.queries import search_traffy_reports as query_reports
  from datetime import date as date_type

  try:
    # Parse categories
    category_list = None
    if categories:
      category_list = [c.strip() for c in categories.split(",")]

    # Parse districts
    district_list = None
    if districts:
      district_list = [d.strip() for d in districts.split(",")]

    # Parse status
    status_list = None
    if status:
      status_list = [s.strip() for s in status.split(",")]

    # Parse dates
    from_date = None
    if date_from:
      from_date = date_type.fromisoformat(date_from)

    to_date = None
    if date_to:
      to_date = date_type.fromisoformat(date_to)

    # Parse bbox
    bbox_tuple = None
    if bbox:
      try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) == 4:
          bbox_tuple = tuple(parts)
        else:
          return "Error: bbox must have 4 values: min_lon,min_lat,max_lon,max_lat"
      except ValueError:
        return "Error: bbox values must be numbers"

    # Execute search
    reports = query_reports(
        categories=category_list,
        districts=district_list,
        status=status_list,
        date_from=from_date,
        date_to=to_date,
        bbox=bbox_tuple,
        search_text=search_text,
        exclude_completed=exclude_completed,
        limit=limit,
    )

    if not reports:
      filters = []
      if category_list:
        filters.append(f"types: {', '.join(category_list)}")
      if district_list:
        filters.append(f"districts: {', '.join(district_list)}")
      if exclude_completed:
        filters.append("unfixed only")
      return f"No reports found matching criteria ({'; '.join(filters) if filters else 'no filters'})"

    # Format output with detailed information
    output_lines = [
        "=== TRAFFY REPORTS ===",
        f"Found {len(reports)} reports",
        "",
    ]

    for i, report in enumerate(reports, 1):
      report_type = report.get('type', 'Unknown')
      status_val = report.get('status', 'Unknown')
      district = report.get('district', 'Unknown')
      address = report.get('address', 'No address')
      lat = report.get('lat', 'N/A')
      lon = report.get('lon', 'N/A')
      description = report.get('description', 'No description')
      timestamp = report.get('timestamp', 'N/A')
      ticket_id = report.get('ticket_id', 'N/A')

      # Truncate long descriptions
      if description and len(description) > 150:
        description = description[:150] + "..."

      output_lines.append(f"--- Report {i} [{ticket_id}] ---")
      output_lines.append(f"Type: {report_type}")
      output_lines.append(f"Status: {status_val}")
      output_lines.append(f"District: {district}")
      output_lines.append(f"Address: {address}")
      output_lines.append(f"Coordinates: ({lat}, {lon})")
      output_lines.append(f"Reported: {timestamp}")
      output_lines.append(f"Description: {description}")
      output_lines.append("")

    if len(reports) >= limit:
      output_lines.append(f"(Showing first {limit} results. Use more specific filters for more results.)")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error searching reports: {str(e)}"


# =============================================================================
# Legacy Data Tools (kept for backward compatibility)
# =============================================================================


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
def get_available_longdo_event_types() -> str:
  """
  List all available Longdo event types with their counts.

  Use this when user asks about Longdo event types, like:
  - "What Longdo event types are available?"
  - "Show me Longdo event categories"
  - "What kinds of traffic events are there?"

  Returns:
      List of event types with their counts
  """
  from data.queries import get_available_longdo_event_types as fetch_event_types

  try:
    event_types = fetch_event_types()

    if not event_types:
      return "No Longdo event types found in the database."

    # Add labels from settings
    labels = settings.longdo.event_type_labels

    output_lines = [
        "=== AVAILABLE LONGDO EVENT TYPES ===",
        f"Total: {len(event_types)} types",
        "",
    ]

    for i, et in enumerate(event_types, 1):
      type_name = et['event_type']
      label = labels.get(type_name, type_name)
      output_lines.append(f"{i:2}. {label} ({type_name}): {et['count']:,} events")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error fetching Longdo event types: {str(e)}"


@tool
def search_longdo_events(
    event_types: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 20,
    include_ended: bool = False
) -> str:
  """
  Search Longdo traffic events and get detailed information.

  Use this when user asks questions like:
  - "Which locations have road closures today?"
  - "Show me active accidents"
  - "Where are the floods happening?"
  - "List fire events from last week"
  - "What traffic events are there in the city?"

  Args:
      event_types: Filter by event types. Options: roadclosed, accident, fire,
                   flood, protest, construction, event, etc. If None, all types.
      date_from: Start date in YYYY-MM-DD format. Defaults to today.
      date_to: End date in YYYY-MM-DD format. Defaults to today.
      limit: Max number of events to return (default 20)
      include_ended: If True, include events that have already ended

  Returns:
      Detailed list of matching events with location, title, description
  """
  from data.queries import query_longdo_events_list
  from datetime import date as date_type

  try:
    # Parse dates
    if date_from:
      from_date = date_type.fromisoformat(date_from)
    else:
      from_date = date_type.today()

    if date_to:
      to_date = date_type.fromisoformat(date_to)
    else:
      to_date = date_type.today()

    events = query_longdo_events_list(
        event_types=event_types,
        date_from=from_date,
        date_to=to_date,
        limit=limit,
    )

    if not events:
      filters = []
      if event_types:
        filters.append(f"types: {', '.join(event_types)}")
      filters.append(f"dates: {from_date} to {to_date}")
      if not include_ended:
        filters.append("only active events")
      return f"No Longdo events found matching criteria ({'; '.join(filters)})"

    # Format output with event details
    labels = settings.longdo.event_type_labels
    output_lines = [
        "=== LONGDO EVENTS ===",
        f"Found {len(events)} events",
        "",
    ]

    for i, event in enumerate(events, 1):
      event_type = event.get('event_type', 'unknown')
      label = labels.get(event_type, event_type)
      title = event.get('title', 'No title')
      description = event.get('description', 'No description')
      lat = event.get('lat', 'N/A')
      lon = event.get('lon', 'N/A')
      start_time = event.get('start_time', 'N/A')
      end_time = event.get('end_time', 'Ongoing')
      posted_by = event.get('posted_by', 'Unknown')

      output_lines.append(f"--- Event {i} ---")
      output_lines.append(f"Type: {label}")
      output_lines.append(f"Title: {title}")
      output_lines.append(f"Description: {description}")
      output_lines.append(f"Location: ({lat}, {lon})")
      output_lines.append(f"Start: {start_time}")
      output_lines.append(f"End: {end_time if end_time else 'Ongoing'}")
      output_lines.append(f"Source: {posted_by}")
      output_lines.append("")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error searching Longdo events: {str(e)}"


@tool
def get_recent_longdo_events(
    event_types: Optional[List[str]] = None,
    limit: int = 10,
    only_active: bool = True
) -> str:
  """
  Get the most recent Longdo traffic events.

  Use this when user asks:
  - "What are the latest traffic events?"
  - "Show me recent road closures"
  - "Any new accidents reported?"
  - "What's happening right now?"

  Args:
      event_types: Filter by event types. Options: roadclosed, accident, fire,
                   flood, protest, construction, event, etc. If None, all types.
      limit: Max number of events to return (default 10)
      only_active: If True, only show currently active events (not ended)

  Returns:
      List of recent events with details
  """
  from data.queries import query_recent_longdo_events

  try:
    # query_recent_longdo_events takes event_type (singular), hours_back, and limit
    # If multiple event_types provided, use the first one
    event_type = event_types[0] if event_types else None
    hours_back = 24 if only_active else 168  # 24 hours or 1 week

    events = query_recent_longdo_events(
        event_type=event_type,
        hours_back=hours_back,
        limit=limit,
    )

    if not events:
      status = "active" if only_active else "recent"
      type_filter = f" of type {', '.join(event_types)}" if event_types else ""
      return f"No {status} Longdo events found{type_filter}."

    # Format output
    labels = settings.longdo.event_type_labels
    status_label = "ACTIVE" if only_active else "RECENT"
    output_lines = [
        f"=== {status_label} LONGDO EVENTS ===",
        f"Showing {len(events)} events",
        "",
    ]

    for i, event in enumerate(events, 1):
      event_type = event.get('event_type', 'unknown')
      label = labels.get(event_type, event_type)
      title = event.get('title', 'No title')
      description = event.get('description', 'No description')
      lat = event.get('lat', 'N/A')
      lon = event.get('lon', 'N/A')
      start_time = event.get('start_time', 'N/A')

      output_lines.append(f"{i}. [{label}] {title}")
      output_lines.append(f"   {description[:100]}{'...' if len(description) > 100 else ''}")
      output_lines.append(f"   📍 Location: ({lat}, {lon})")
      output_lines.append(f"   🕐 Started: {start_time}")
      output_lines.append("")

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error fetching recent Longdo events: {str(e)}"


@tool
def get_longdo_event_detail(event_id: str) -> str:
  """
  Get detailed information about a specific Longdo event.

  Use this when user asks about a specific event by ID.

  Args:
      event_id: The event ID to look up

  Returns:
      Full details of the event
  """
  from data.queries import get_longdo_event_detail

  try:
    event = get_longdo_event_detail(event_id)

    if not event:
      return f"No Longdo event found with ID: {event_id}"

    labels = settings.longdo.event_type_labels
    event_type = event.get('event_type', 'unknown')
    label = labels.get(event_type, event_type)

    output_lines = [
        "=== LONGDO EVENT DETAILS ===",
        f"Event ID: {event.get('event_id', 'N/A')}",
        f"Type: {label}",
        f"Title: {event.get('title', 'No title')}",
        f"Description: {event.get('description', 'No description')}",
        f"Location: ({event.get('lat', 'N/A')}, {event.get('lon', 'N/A')})",
        f"Start Time: {event.get('start_time', 'N/A')}",
        f"End Time: {event.get('end_time', 'Ongoing') or 'Ongoing'}",
        f"Posted By: {event.get('posted_by', 'Unknown')}",
        f"Source: {event.get('source', 'N/A')}",
        f"Contribute: {event.get('contribute', 'N/A')}",
    ]

    return "\n".join(output_lines)

  except Exception as e:
    return f"Error fetching Longdo event details: {str(e)}"


@tool
def switch_data_source(data_source: str) -> str:
  """
  Switch between Traffy Fondue and Longdo Events data sources.

  Use when user wants to analyze a different data source, like:
  - "Switch to Longdo data"
  - "Show me Traffy reports"
  - "Use Longdo events"

  Args:
      data_source: "traffy" for Traffy Fondue reports, "longdo" for Longdo Traffic events

  Returns:
      Confirmation of data source switch
  """
  if data_source not in ["traffy", "longdo"]:
    return f"Invalid data source '{data_source}'. Choose 'traffy' or 'longdo'."

  st.session_state.data_source = data_source
  # Update widget state
  st.session_state.data_source_select = data_source

  source_name = "Traffy Fondue" if data_source == "traffy" else "Longdo Traffic Events"
  return f"Switched to {source_name} data. Queries will now use this data source."


@tool
def get_current_data_source() -> str:
  """
  Get the currently active data source.

  Use to check which data source is being used for queries.

  Returns:
      Current data source name and description
  """
  source = _get_current_data_source()
  source_info = {
      "traffy": ("Traffy Fondue", "Bangkok urban problem reports from citizens"),
      "longdo": ("Longdo Traffic Events", "Real-time traffic and event reports"),
  }
  name, desc = source_info.get(source, (source, "Unknown data source"))
  return f"Current data source: {name}\nDescription: {desc}"


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
  Zoom the map to focus on a specific district or area.

  Use when user wants to see a specific area, like:
  - "show me Khlong Toei district" or "zoom to Bang Rak"
  - "โชว์แถวสามย่าน" (show Sam Yan area)
  - "zoom to Asok" or "ดูแถวอโศก"

  Supports both Bangkok district names and popular landmarks/areas like:
  สามย่าน, สยาม, อโศก, ทองหล่อ, เอกมัย, สีลม, ลาดพร้าว, รามคำแหง, etc.

  Args:
      district: Bangkok district name or popular area (Thai or English)

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

  # Popular areas/landmarks mapped to coordinates
  area_coords = {
      # Sam Yan / Siam area (ปทุมวัน)
      "สามย่าน": (13.7323, 100.5292),
      "samyan": (13.7323, 100.5292),
      "sam yan": (13.7323, 100.5292),
      "สยาม": (13.7456, 100.5342),
      "siam": (13.7456, 100.5342),
      "mbk": (13.7447, 100.5299),
      "จุฬา": (13.7387, 100.5313),
      "chula": (13.7387, 100.5313),
      # Asok / Sukhumvit (วัฒนา)
      "อโศก": (13.7379, 100.5605),
      "asok": (13.7379, 100.5605),
      "สุขุมวิท": (13.7313, 100.5670),
      "sukhumvit": (13.7313, 100.5670),
      "ทองหล่อ": (13.7326, 100.5787),
      "thonglor": (13.7326, 100.5787),
      "เอกมัย": (13.7199, 100.5852),
      "ekkamai": (13.7199, 100.5852),
      # Silom / Sathorn
      "สีลม": (13.7280, 100.5341),
      "silom": (13.7280, 100.5341),
      "สาทร": (13.7189, 100.5268),
      "sathorn": (13.7189, 100.5268),
      # Other popular areas
      "ลาดพร้าว": (13.8167, 100.5852),
      "ladprao": (13.8167, 100.5852),
      "รามคำแหง": (13.7581, 100.6229),
      "ramkhamhaeng": (13.7581, 100.6229),
      "อ่อนนุช": (13.7058, 100.6012),
      "onnut": (13.7058, 100.6012),
      "on nut": (13.7058, 100.6012),
      "พระราม9": (13.7581, 100.5655),
      "rama9": (13.7581, 100.5655),
      "พระราม 9": (13.7581, 100.5655),
      "รัชดา": (13.7685, 100.5691),
      "ratchada": (13.7685, 100.5691),
  }

  # Try to find matching district first
  coords = district_coords.get(district)
  matched_name = district

  if not coords:
    # Try area coordinates
    coords = area_coords.get(district.lower())
    if coords:
      matched_name = district

  if not coords:
    # Try partial match in districts
    for d, c in district_coords.items():
      if district.lower() in d.lower() or d.lower() in district.lower():
        coords = c
        matched_name = d
        break

  if not coords:
    # Try partial match in areas
    for a, c in area_coords.items():
      if district.lower() in a.lower() or a.lower() in district.lower():
        coords = c
        matched_name = a
        break

  if not coords:
    return f"Location '{district}' not found. Try Thai names like คลองเตย, บางรัก, ปทุมวัน, สามย่าน, อโศก."

  # Update map view state
  response = _ui_mcp.execute_action(
      UIAction(
          action_type="set_view",
          params={
              "latitude": coords[0],
              "longitude": coords[1],
              "zoom_delta": 3,
          }
      )
  )

  if response.success:
    return f"Map zoomed to {matched_name} (center: {coords[0]:.4f}, {coords[1]:.4f})"
  return f"Error zooming to location: {response.error}"


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

  st.session_state.data_date_from = date_from
  st.session_state.data_date_to = date_to

  # Update widget keys
  st.session_state.data_date_from_input = date_from
  st.session_state.data_date_to_input = date_to

  return f"Date filter set: {date_from} to {date_to} ({days_back} days)"


# =============================================================================
# Tool Collections
# =============================================================================

# Data tools (analytical)
data_tools = [
    get_data_schema,
    get_ticket_counts,
    get_statistics,
    get_time_series,
    get_crosstab,
    run_analytical_query,
    get_ticket_detail,
    get_available_categories,
    get_available_longdo_event_types,
    search_longdo_events,
    get_recent_longdo_events,
    get_longdo_event_detail,
    switch_data_source,
    get_current_data_source,
    search_traffy_reports,
]

# Backward compatibility aliases
analytical_tools = data_tools
legacy_data_tools: List[Any] = []

# Spatial filter tools
spatial_tools = [
    filter_by_district,
    filter_by_bounding_box,
    get_tickets_in_area,
    get_available_districts,
]

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
all_tools = data_tools + spatial_tools + ui_tools
