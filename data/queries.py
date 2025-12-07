"""Optimized SQL queries for Traffy Fondue data with spatial filtering."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config.settings import settings
from .database import get_cursor


# =============================================================================
# Helper Functions for Spatial Filtering
# =============================================================================

def _build_bbox_condition(bbox: Optional[Tuple[float, float, float, float]]) -> Tuple[str, List[Any]]:
  """
  Build a PostGIS bounding box condition.

  Args:
      bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)

  Returns:
      Tuple of (condition_string, params_list)
  """
  if not bbox:
    return "", []

  min_lon, min_lat, max_lon, max_lat = bbox
  condition = "ST_Within(location::geometry, ST_MakeEnvelope(%s, %s, %s, %s, 4326))"
  return condition, [min_lon, min_lat, max_lon, max_lat]


def _build_districts_condition(districts: Optional[List[str]]) -> Tuple[str, List[Any]]:
  """
  Build a district filter condition.

  Args:
      districts: List of district names

  Returns:
      Tuple of (condition_string, params_list)
  """
  if not districts:
    return "", []

  placeholders = ",".join(["%s"] * len(districts))
  condition = f"district IN ({placeholders})"
  return condition, districts


# =============================================================================
# Main Data Fetch Function
# =============================================================================

@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_traffy_data(
    categories: Optional[List[str]] = None,
    districts: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 100000,
) -> pd.DataFrame:
  """
  Fetch traffy_data from the database with PostGIS geometry extraction.

  Args:
      categories: List of category types to filter
      districts: List of district names to filter
      date_from: Start date filter
      date_to: End date filter
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)
      limit: Maximum number of records to return

  Returns:
      DataFrame with columns including org and last_activity for pipeline processing
  """
  conditions = ["location IS NOT NULL"]
  params: List[Any] = []

  # Category filter (using 'type' column)
  if categories:
    placeholders = ",".join(["%s"] * len(categories))
    conditions.append(f"type IN ({placeholders})")
    params.extend(categories)

  # District filter
  district_cond, district_params = _build_districts_condition(districts)
  if district_cond:
    conditions.append(district_cond)
    params.extend(district_params)

  # Date filters
  if date_from:
    conditions.append("timestamp >= %s")
    params.append(date_from)

  if date_to:
    conditions.append("timestamp <= %s")
    params.append(date_to)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  query = f"""
        SELECT
            id,
            ticket_id,
            ST_Y(location::geometry) as lat,
            ST_X(location::geometry) as lon,
            type,
            comment as description,
            photo as photo_url,
            state as status,
            timestamp,
            last_activity,
            organization as org,
            district,
            province,
            address
        FROM traffy_tickets
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s
    """
  params.append(limit)

  # Define column names for the DataFrame
  columns = [
      "id", "ticket_id", "lat", "lon", "type", "description", "photo_url",
      "status", "timestamp", "last_activity", "org",
      "district", "province", "address"
  ]

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      rows = cur.fetchall()

      if not rows:
        return pd.DataFrame(columns=columns)

      return pd.DataFrame(rows)
  except Exception as e:
    st.error(f"Database error: {e}")
    return pd.DataFrame(columns=columns)


@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_longdo_events(
    event_types: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 100000,
) -> pd.DataFrame:
  """
  Fetch longdo_events from the database with PostGIS geometry extraction.

  Args:
      event_types: List of event types to filter (roadclosed, fire, etc.)
      date_from: Start date filter (based on start_time)
      date_to: End date filter (based on end_time)
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)
      limit: Maximum number of records to return

  Returns:
      DataFrame with columns for visualization
  """
  conditions = ["location IS NOT NULL"]
  params: List[Any] = []

  # Event type filter
  if event_types:
    placeholders = ",".join(["%s"] * len(event_types))
    conditions.append(f"event_type IN ({placeholders})")
    params.extend(event_types)

  # Date filters (using start_time and end_time)
  if date_from:
    conditions.append("start_time >= %s")
    params.append(date_from)

  if date_to:
    conditions.append("(end_time <= %s OR end_time IS NULL)")
    params.append(date_to)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  query = f"""
        SELECT
            id,
            event_id,
            event_type as type,
            title,
            description,
            posted_by,
            ST_Y(location::geometry) as lat,
            ST_X(location::geometry) as lon,
            scraped_at,
            start_time as timestamp,
            end_time,
            created_at,
            updated_at
        FROM longdo_events
        WHERE {where_clause}
        ORDER BY start_time DESC
        LIMIT %s
    """
  params.append(limit)

  # Define column names for the DataFrame
  columns = [
      "id", "event_id", "type", "title", "description", "posted_by",
      "lat", "lon", "scraped_at", "timestamp", "end_time",
      "created_at", "updated_at"
  ]

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      rows = cur.fetchall()

      if not rows:
        return pd.DataFrame(columns=columns)

      return pd.DataFrame(rows)
  except Exception as e:
    st.error(f"Database error: {e}")
    return pd.DataFrame(columns=columns)


def get_traffy_date_range() -> Tuple[Optional[date], Optional[date]]:
  """Get the min and max dates in the traffy_data dataset."""
  query = """
        SELECT MIN(timestamp)::date as min_date, MAX(timestamp)::date as max_date
        FROM traffy_data
    """

  try:
    with get_cursor() as cur:
      cur.execute(query)
      row = cur.fetchone()
      if row:
        return row["min_date"], row["max_date"]
      return None, None
  except Exception:
    return None, None


# =============================================================================
# AI Analytics Query Functions
# =============================================================================

# Valid columns and dimensions for query validation
VALID_DIMENSIONS = {
    "type": "type",
    "category": "type",  # Alias
    "district": "district",
    "province": "province",
    "status": "state",
    "state": "state",
    "org": "organization",
    "organization": "organization",
}

VALID_TIME_GRANULARITIES = ["hour", "day", "week", "month", "year"]

VALID_ORDER_BY = ["count_desc", "count_asc", "name_asc", "name_desc", "value_desc", "value_asc"]


def get_schema_info() -> Dict[str, Any]:
  """
  Get database schema information for AI introspection.

  Returns:
      Dictionary with table schema, available dimensions, and sample values
  """
  schema_info = {
      "table": "traffy_tickets",
      "description": "Bangkok urban problem reports (Traffy Fondue)",
      "columns": {
          "id": {"type": "integer", "description": "Unique record ID"},
          "ticket_id": {"type": "string", "description": "Ticket reference ID"},
          "type": {"type": "string", "description": "Report category/type (e.g., ถนน, น้ำท่วม)"},
          "state": {"type": "string", "description": "Ticket status (e.g., เสร็จสิ้น, รับเรื่องแล้ว)"},
          "district": {"type": "string", "description": "Bangkok district name"},
          "province": {"type": "string", "description": "Province (mostly กรุงเทพมหานคร)"},
          "organization": {"type": "string", "description": "Responsible organization"},
          "timestamp": {"type": "datetime", "description": "When ticket was created"},
          "last_activity": {"type": "datetime", "description": "Last update time"},
          "comment": {"type": "text", "description": "User's problem description"},
          "address": {"type": "text", "description": "Location address"},
          "location": {"type": "geometry", "description": "GPS coordinates (PostGIS point)"},
      },
      "queryable_dimensions": list(VALID_DIMENSIONS.keys()),
      "time_granularities": VALID_TIME_GRANULARITIES,
      "spatial_filters": {
          "bbox": "Bounding box filter (min_lon, min_lat, max_lon, max_lat)",
          "districts": "Filter by district name(s)",
      },
  }

  # Fetch sample values for key dimensions
  try:
    with get_cursor() as cur:
      # Get unique types with counts
      cur.execute("""
                SELECT type, COUNT(*) as count
                FROM traffy_tickets
                WHERE type IS NOT NULL
                GROUP BY type
                ORDER BY count DESC
                LIMIT 10
            """)
      schema_info["sample_types"] = [
          {"value": r["type"], "count": r["count"]} for r in cur.fetchall()
      ]

      # Get unique statuses
      cur.execute("""
                SELECT DISTINCT state
                FROM traffy_tickets
                WHERE state IS NOT NULL
                LIMIT 10
            """)
      schema_info["sample_statuses"] = [r["state"] for r in cur.fetchall()]

      # Get unique districts with counts
      cur.execute("""
                SELECT district, COUNT(*) as count
                FROM traffy_tickets
                WHERE district IS NOT NULL
                GROUP BY district
                ORDER BY count DESC
                LIMIT 10
            """)
      schema_info["sample_districts"] = [
          {"value": r["district"], "count": r["count"]} for r in cur.fetchall()
      ]

      # Get date range
      cur.execute("""
                SELECT
                    MIN(timestamp)::date as min_date,
                    MAX(timestamp)::date as max_date,
                    COUNT(*) as total_records
                FROM traffy_tickets
            """)
      row = cur.fetchone()
      if row:
        schema_info["date_range"] = {
            "min": str(row["min_date"]) if row["min_date"] else None,
            "max": str(row["max_date"]) if row["max_date"] else None,
            "total_records": row["total_records"],
        }

  except Exception as e:
    schema_info["error"] = f"Could not fetch sample data: {e}"

  return schema_info


def query_aggregation(
    group_by: str,
    filters: Optional[Dict[str, Any]] = None,
    order_by: str = "count_desc",
    limit: int = 20,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
  """
  Execute an aggregation query grouped by a dimension.

  Args:
      group_by: Dimension to group by (type, district, status, org)
      filters: Optional filters {dimension: value or [values]}
      order_by: Sort order (count_desc, count_asc, name_asc, name_desc)
      limit: Maximum results to return
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)

  Returns:
      List of {dimension_value, count} dictionaries
  """
  # Validate and map dimension
  if group_by not in VALID_DIMENSIONS:
    raise ValueError(f"Invalid group_by dimension: {group_by}. Valid: {list(VALID_DIMENSIONS.keys())}")

  db_column = VALID_DIMENSIONS[group_by]

  # Build query
  conditions = [f"{db_column} IS NOT NULL"]
  params: List[Any] = []

  # Apply filters
  if filters:
    for dim, value in filters.items():
      if dim == "date_from":
        conditions.append("timestamp >= %s")
        params.append(value)
      elif dim == "date_to":
        conditions.append("timestamp <= %s")
        params.append(value)
      elif dim not in VALID_DIMENSIONS:
        continue
      else:
        filter_col = VALID_DIMENSIONS[dim]
        if isinstance(value, list):
          placeholders = ",".join(["%s"] * len(value))
          conditions.append(f"{filter_col} IN ({placeholders})")
          params.extend(value)
        else:
          conditions.append(f"{filter_col} = %s")
          params.append(value)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  # Determine order
  if order_by == "count_desc":
    order = "count DESC"
  elif order_by == "count_asc":
    order = "count ASC"
  elif order_by == "name_asc":
    order = f"{db_column} ASC"
  elif order_by == "name_desc":
    order = f"{db_column} DESC"
  else:
    order = "count DESC"

  query = f"""
        SELECT {db_column} as dimension_value, COUNT(*) as count
        FROM traffy_tickets
        WHERE {where_clause}
        GROUP BY {db_column}
        ORDER BY {order}
        LIMIT %s
    """
  params.append(limit)

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      return [dict(row) for row in cur.fetchall()]
  except Exception as e:
    raise RuntimeError(f"Aggregation query failed: {e}")


def query_time_series(
    granularity: str = "day",
    metric: str = "count",
    group_by: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 100,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
  """
  Execute a time series query.

  Args:
      granularity: Time bucket (hour, day, week, month, year)
      metric: Metric to compute (count, avg_resolution_hours)
      group_by: Optional dimension to break down by
      filters: Optional filters
      date_from: Start date
      date_to: End date
      limit: Maximum results
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)

  Returns:
      List of time-series data points
  """
  if granularity not in VALID_TIME_GRANULARITIES:
    raise ValueError(f"Invalid granularity: {granularity}. Valid: {VALID_TIME_GRANULARITIES}")

  # Build time bucket expression
  time_bucket = {
      "hour": "DATE_TRUNC('hour', timestamp)",
      "day": "DATE_TRUNC('day', timestamp)",
      "week": "DATE_TRUNC('week', timestamp)",
      "month": "DATE_TRUNC('month', timestamp)",
      "year": "DATE_TRUNC('year', timestamp)",
  }[granularity]

  conditions = ["timestamp IS NOT NULL"]
  params: List[Any] = []

  # Date filters
  if date_from:
    conditions.append("timestamp >= %s")
    params.append(date_from)
  if date_to:
    conditions.append("timestamp <= %s")
    params.append(date_to)

  # Apply dimension filters
  if filters:
    for dim, value in filters.items():
      if dim == "date_from":
        conditions.append("timestamp >= %s")
        params.append(value)
      elif dim == "date_to":
        conditions.append("timestamp <= %s")
        params.append(value)
      elif dim not in VALID_DIMENSIONS:
        continue
      else:
        filter_col = VALID_DIMENSIONS[dim]
        if isinstance(value, list):
          placeholders = ",".join(["%s"] * len(value))
          conditions.append(f"{filter_col} IN ({placeholders})")
          params.extend(value)
        else:
          conditions.append(f"{filter_col} = %s")
          params.append(value)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  # Build metric expression
  if metric == "count":
    metric_expr = "COUNT(*)"
    metric_alias = "count"
  elif metric == "avg_resolution_hours":
    metric_expr = "AVG(EXTRACT(EPOCH FROM (last_activity - timestamp)) / 3600)"
    metric_alias = "avg_resolution_hours"
  else:
    metric_expr = "COUNT(*)"
    metric_alias = "count"

  # Build group by and select
  if group_by and group_by in VALID_DIMENSIONS:
    group_col = VALID_DIMENSIONS[group_by]
    select_cols = f"{time_bucket} as period, {group_col} as group_value, {metric_expr} as {metric_alias}"
    group_clause = f"{time_bucket}, {group_col}"
  else:
    select_cols = f"{time_bucket} as period, {metric_expr} as {metric_alias}"
    group_clause = time_bucket

  query = f"""
        SELECT {select_cols}
        FROM traffy_tickets
        WHERE {where_clause}
        GROUP BY {group_clause}
        ORDER BY period DESC
        LIMIT %s
    """
  params.append(limit)

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      results = []
      for row in cur.fetchall():
        item = dict(row)
        # Convert datetime to string for JSON serialization
        if item.get("period"):
          item["period"] = str(item["period"])
        results.append(item)
      return results
  except Exception as e:
    raise RuntimeError(f"Time series query failed: {e}")


def query_crosstab(
    row_dimension: str,
    col_dimension: str,
    metric: str = "count",
    filters: Optional[Dict[str, Any]] = None,
    row_limit: int = 15,
    col_limit: int = 10,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
  """
  Execute a cross-tabulation query.

  Args:
      row_dimension: Dimension for rows (e.g., district)
      col_dimension: Dimension for columns (e.g., type)
      metric: Metric to compute (count, avg_resolution_hours)
      filters: Optional filters
      row_limit: Max rows to return
      col_limit: Max columns to return
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)

  Returns:
      Dictionary with rows, columns, and data matrix
  """
  if row_dimension not in VALID_DIMENSIONS:
    raise ValueError(f"Invalid row_dimension: {row_dimension}")
  if col_dimension not in VALID_DIMENSIONS:
    raise ValueError(f"Invalid col_dimension: {col_dimension}")

  row_col = VALID_DIMENSIONS[row_dimension]
  col_col = VALID_DIMENSIONS[col_dimension]

  conditions = [f"{row_col} IS NOT NULL", f"{col_col} IS NOT NULL"]
  params: List[Any] = []

  if filters:
    for dim, value in filters.items():
      if dim == "date_from":
        conditions.append("timestamp >= %s")
        params.append(value)
      elif dim == "date_to":
        conditions.append("timestamp <= %s")
        params.append(value)
      elif dim not in VALID_DIMENSIONS:
        continue
      else:
        filter_col = VALID_DIMENSIONS[dim]
        if isinstance(value, list):
          placeholders = ",".join(["%s"] * len(value))
          conditions.append(f"{filter_col} IN ({placeholders})")
          params.extend(value)
        else:
          conditions.append(f"{filter_col} = %s")
          params.append(value)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  # Metric expression
  if metric == "count":
    metric_expr = "COUNT(*)"
  elif metric == "avg_resolution_hours":
    metric_expr = "AVG(EXTRACT(EPOCH FROM (last_activity - timestamp)) / 3600)"
  else:
    metric_expr = "COUNT(*)"

  query = f"""
        SELECT
            {row_col} as row_value,
            {col_col} as col_value,
            {metric_expr} as value
        FROM traffy_tickets
        WHERE {where_clause}
        GROUP BY {row_col}, {col_col}
        ORDER BY value DESC
    """

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      rows = cur.fetchall()

      # Build crosstab structure
      row_values = []
      col_values = set()
      data = {}

      for row in rows:
        rv = row["row_value"]
        cv = row["col_value"]
        val = row["value"]

        if rv not in data:
          data[rv] = {}
          row_values.append(rv)

        data[rv][cv] = float(val) if val else 0
        col_values.add(cv)

      # Limit rows and columns
      row_values = row_values[:row_limit]
      col_values = sorted(list(col_values))[:col_limit]

      # Build matrix
      matrix = []
      for rv in row_values:
        row_data = [data.get(rv, {}).get(cv, 0) for cv in col_values]
        matrix.append(row_data)

      return {
          "row_dimension": row_dimension,
          "col_dimension": col_dimension,
          "rows": row_values,
          "columns": col_values,
          "data": matrix,
          "metric": metric,
      }
  except Exception as e:
    raise RuntimeError(f"Crosstab query failed: {e}")


def query_statistics(
    dimension: Optional[str] = None,
    dimension_value: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
  """
  Get statistical summary of tickets.

  Args:
      dimension: Optional dimension to filter by
      dimension_value: Value for the dimension filter
      filters: Additional filters
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)

  Returns:
      Dictionary with statistics
  """
  conditions = ["1=1"]
  params: List[Any] = []

  if dimension and dimension_value and dimension in VALID_DIMENSIONS:
    dim_col = VALID_DIMENSIONS[dimension]
    conditions.append(f"{dim_col} = %s")
    params.append(dimension_value)

  if filters:
    for dim, value in filters.items():
      if dim == "date_from":
        conditions.append("timestamp >= %s")
        params.append(value)
      elif dim == "date_to":
        conditions.append("timestamp <= %s")
        params.append(value)
      elif dim not in VALID_DIMENSIONS:
        continue
      else:
        filter_col = VALID_DIMENSIONS[dim]
        if isinstance(value, list):
          placeholders = ",".join(["%s"] * len(value))
          conditions.append(f"{filter_col} IN ({placeholders})")
          params.extend(value)
        else:
          conditions.append(f"{filter_col} = %s")
          params.append(value)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  query = f"""
        SELECT
            COUNT(*) as total_tickets,
            COUNT(DISTINCT type) as unique_types,
            COUNT(DISTINCT district) as unique_districts,
            COUNT(DISTINCT organization) as unique_orgs,
            MIN(timestamp)::date as earliest_date,
            MAX(timestamp)::date as latest_date,
            AVG(EXTRACT(EPOCH FROM (last_activity - timestamp)) / 3600) as avg_resolution_hours,
            COUNT(CASE WHEN state = 'เสร็จสิ้น' THEN 1 END) as completed_count,
            COUNT(CASE WHEN state != 'เสร็จสิ้น' OR state IS NULL THEN 1 END) as pending_count
        FROM traffy_tickets
        WHERE {where_clause}
    """

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      row = cur.fetchone()

      if not row:
        return {"error": "No data found"}

      total = row["total_tickets"] or 0
      completed = row["completed_count"] or 0

      return {
          "total_tickets": total,
          "unique_types": row["unique_types"],
          "unique_districts": row["unique_districts"],
          "unique_organizations": row["unique_orgs"],
          "date_range": {
              "from": str(row["earliest_date"]) if row["earliest_date"] else None,
              "to": str(row["latest_date"]) if row["latest_date"] else None,
          },
          "avg_resolution_hours": round(row["avg_resolution_hours"], 2) if row["avg_resolution_hours"] else None,
          "completion_rate": round(completed / total * 100, 2) if total > 0 else 0,
          "completed_count": completed,
          "pending_count": row["pending_count"] or 0,
      }
  except Exception as e:
    raise RuntimeError(f"Statistics query failed: {e}")


def query_ticket_details(ticket_id: str) -> Optional[Dict[str, Any]]:
  """
  Get detailed information about a specific ticket.

  Args:
      ticket_id: The ticket ID to look up

  Returns:
      Dictionary with ticket details or None
  """
  query = """
        SELECT
            id,
            ticket_id,
            type,
            state as status,
            comment as description,
            address,
            district,
            province,
            organization,
            timestamp as created_at,
            last_activity,
            photo as photo_url,
            ST_Y(location::geometry) as lat,
            ST_X(location::geometry) as lon
        FROM traffy_tickets
        WHERE ticket_id = %s
        LIMIT 1
    """

  try:
    with get_cursor() as cur:
      cur.execute(query, [ticket_id])
      row = cur.fetchone()

      if not row:
        return None

      result = dict(row)
      # Convert datetimes to strings
      if result.get("created_at"):
        result["created_at"] = str(result["created_at"])
      if result.get("last_activity"):
        result["last_activity"] = str(result["last_activity"])

      return result
  except Exception as e:
    raise RuntimeError(f"Ticket detail query failed: {e}")


def execute_flexible_query(
    select_columns: List[str],
    aggregations: Optional[List[Dict[str, str]]] = None,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[str]] = None,
    order_by: Optional[str] = None,
    limit: int = 100,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
  """
  Execute a flexible analytical query with validation.

  Args:
      select_columns: Columns to select (validated against schema)
      aggregations: List of {column, function, alias} for aggregations
      filters: Dimension filters
      group_by: Columns to group by
      order_by: Order expression (e.g., "count DESC")
      limit: Maximum results
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)

  Returns:
      Query results as list of dictionaries
  """
  # Valid columns that can be selected
  VALID_SELECT_COLUMNS = {
      "type", "district", "province", "state", "organization",
      "timestamp", "last_activity", "ticket_id"
  }

  # Valid aggregation functions
  VALID_AGG_FUNCTIONS = {"count", "sum", "avg", "min", "max", "count_distinct"}

  # Validate select columns
  select_parts = []
  for col in select_columns:
    if col in VALID_DIMENSIONS:
      select_parts.append(f"{VALID_DIMENSIONS[col]} as {col}")
    elif col in VALID_SELECT_COLUMNS:
      select_parts.append(col)
    # Skip invalid columns silently

  # Process aggregations
  if aggregations:
    for agg in aggregations:
      func = agg.get("function", "count").lower()
      col = agg.get("column", "*")
      alias = agg.get("alias", f"{func}_{col}")

      if func not in VALID_AGG_FUNCTIONS:
        continue

      if func == "count":
        if col == "*":
          select_parts.append(f"COUNT(*) as {alias}")
        else:
          select_parts.append(f"COUNT({col}) as {alias}")
      elif func == "count_distinct":
        select_parts.append(f"COUNT(DISTINCT {col}) as {alias}")
      elif func in ("sum", "avg", "min", "max"):
        select_parts.append(f"{func.upper()}({col}) as {alias}")

  if not select_parts:
    raise ValueError("No valid columns to select")

  # Build conditions
  conditions = ["1=1"]
  params: List[Any] = []

  if filters:
    for dim, value in filters.items():
      if dim in VALID_DIMENSIONS:
        filter_col = VALID_DIMENSIONS[dim]
        if isinstance(value, list):
          placeholders = ",".join(["%s"] * len(value))
          conditions.append(f"{filter_col} IN ({placeholders})")
          params.extend(value)
        else:
          conditions.append(f"{filter_col} = %s")
          params.append(value)
      elif dim == "date_from":
        conditions.append("timestamp >= %s")
        params.append(value)
      elif dim == "date_to":
        conditions.append("timestamp <= %s")
        params.append(value)

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  where_clause = " AND ".join(conditions)

  # Build group by
  group_clause = ""
  if group_by:
    valid_groups = []
    for g in group_by:
      if g in VALID_DIMENSIONS:
        valid_groups.append(VALID_DIMENSIONS[g])
      elif g in VALID_SELECT_COLUMNS:
        valid_groups.append(g)
    if valid_groups:
      group_clause = f"GROUP BY {', '.join(valid_groups)}"

  # Build order by (with validation)
  order_clause = ""
  if order_by:
    # Simple validation - only allow basic order expressions
    order_by_clean = order_by.replace("'", "").replace('"', "").replace(";", "")
    if any(word in order_by_clean.lower() for word in ["count", "sum", "avg", "desc", "asc"]):
      order_clause = f"ORDER BY {order_by_clean}"

  query = f"""
        SELECT {', '.join(select_parts)}
        FROM traffy_tickets
        WHERE {where_clause}
        {group_clause}
        {order_clause}
        LIMIT %s
    """
  params.append(limit)

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      results = []
      for row in cur.fetchall():
        item = dict(row)
        # Convert any datetime objects to strings
        for k, v in item.items():
          if hasattr(v, 'isoformat'):
            item[k] = v.isoformat()
        results.append(item)
      return results
  except Exception as e:
    raise RuntimeError(f"Flexible query failed: {e}")


def query_by_bbox(
    bbox: Tuple[float, float, float, float],
    categories: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
  """
  Query tickets within a bounding box area.

  Args:
      bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat)
      categories: Optional list of categories to filter
      date_from: Start date filter
      date_to: End date filter
      limit: Maximum results

  Returns:
      List of ticket dictionaries with coordinates
  """
  conditions = ["location IS NOT NULL"]
  params: List[Any] = []

  # Bounding box filter
  bbox_cond, bbox_params = _build_bbox_condition(bbox)
  if bbox_cond:
    conditions.append(bbox_cond)
    params.extend(bbox_params)

  # Category filter
  if categories:
    placeholders = ",".join(["%s"] * len(categories))
    conditions.append(f"type IN ({placeholders})")
    params.extend(categories)

  # Date filters
  if date_from:
    conditions.append("timestamp >= %s")
    params.append(date_from)
  if date_to:
    conditions.append("timestamp <= %s")
    params.append(date_to)

  where_clause = " AND ".join(conditions)

  query = f"""
        SELECT
            ticket_id,
            type,
            state as status,
            district,
            ST_Y(location::geometry) as lat,
            ST_X(location::geometry) as lon,
            timestamp,
            comment as description
        FROM traffy_tickets
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s
    """
  params.append(limit)

  try:
    with get_cursor() as cur:
      cur.execute(query, params)
      results = []
      for row in cur.fetchall():
        item = dict(row)
        if item.get("timestamp"):
          item["timestamp"] = str(item["timestamp"])
        results.append(item)
      return results
  except Exception as e:
    raise RuntimeError(f"Bounding box query failed: {e}")


def get_available_districts() -> List[Dict[str, Any]]:
  """
  Get list of all available districts with their counts.

  Returns:
      List of {district, count} dictionaries
  """
  query = """
        SELECT district, COUNT(*) as count
        FROM traffy_tickets
        WHERE district IS NOT NULL
        GROUP BY district
        ORDER BY count DESC
    """

  try:
    with get_cursor() as cur:
      cur.execute(query)
      return [dict(row) for row in cur.fetchall()]
  except Exception as e:
    raise RuntimeError(f"District list query failed: {e}")
