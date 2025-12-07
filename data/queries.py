"""Optimized SQL queries for Traffy Fondue data with spatial filtering."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config.settings import settings
from .database import get_cursor


@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_reports(
    categories: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 10000,
) -> pd.DataFrame:
    """
    Fetch report data with spatial and temporal filtering.

    Args:
        categories: List of category names to filter
        date_from: Start date filter
        date_to: End date filter
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        limit: Maximum number of records to return

    Returns:
        DataFrame with columns: id, lat, lon, category, created_at, description, status
    """
    conditions = ["1=1"]
    params: List[Any] = []

    # Category filter
    if categories:
        placeholders = ",".join(["%s"] * len(categories))
        conditions.append(f"category IN ({placeholders})")
        params.extend(categories)

    # Date filters
    if date_from:
        conditions.append("created_at >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("created_at <= %s")
        params.append(date_to)

    # Bounding box spatial filter (min_lon, min_lat, max_lon, max_lat)
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        conditions.append("lon BETWEEN %s AND %s")
        conditions.append("lat BETWEEN %s AND %s")
        params.extend([min_lon, max_lon, min_lat, max_lat])

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT 
            id,
            lat,
            lon,
            category,
            created_at,
            description,
            status
        FROM reports
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)

    try:
        with get_cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                return pd.DataFrame(
                    columns=["id", "lat", "lon", "category", "created_at", "description", "status"]
                )

            return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame(
            columns=["id", "lat", "lon", "category", "created_at", "description", "status"]
        )


@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_reports_aggregated(
    categories: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    grid_size: float = 0.005,
) -> pd.DataFrame:
    """
    Fetch aggregated report data grouped by grid cells.
    Used for heatmap and hexagon visualizations.

    Args:
        categories: List of category names to filter
        date_from: Start date filter
        date_to: End date filter
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        grid_size: Size of grid cells in degrees (~500m at Bangkok latitude)

    Returns:
        DataFrame with columns: lat, lon, count, weight
    """
    conditions = ["1=1"]
    params: List[Any] = []

    if categories:
        placeholders = ",".join(["%s"] * len(categories))
        conditions.append(f"category IN ({placeholders})")
        params.extend(categories)

    if date_from:
        conditions.append("created_at >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("created_at <= %s")
        params.append(date_to)

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        conditions.append("lon BETWEEN %s AND %s")
        conditions.append("lat BETWEEN %s AND %s")
        params.extend([min_lon, max_lon, min_lat, max_lat])

    where_clause = " AND ".join(conditions)

    # Grid-based aggregation using floor division
    query = f"""
        SELECT 
            (FLOOR(lat / %s) * %s + %s / 2) as lat,
            (FLOOR(lon / %s) * %s + %s / 2) as lon,
            COUNT(*) as count,
            COUNT(*)::float as weight
        FROM reports
        WHERE {where_clause}
        GROUP BY FLOOR(lat / %s), FLOOR(lon / %s)
        HAVING COUNT(*) > 0
        ORDER BY count DESC
    """
    # Add grid_size params (used 8 times in query)
    grid_params = [grid_size] * 8
    params = grid_params + params

    try:
        with get_cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                return pd.DataFrame(columns=["lat", "lon", "count", "weight"])

            return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame(columns=["lat", "lon", "count", "weight"])


@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_category_counts(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Dict[str, int]:
    """
    Fetch count of reports per category.

    Args:
        date_from: Start date filter
        date_to: End date filter

    Returns:
        Dictionary mapping category name to count
    """
    conditions = ["1=1"]
    params: List[Any] = []

    if date_from:
        conditions.append("created_at >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("created_at <= %s")
        params.append(date_to)

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT category, COUNT(*) as count
        FROM reports
        WHERE {where_clause}
        GROUP BY category
        ORDER BY count DESC
    """

    try:
        with get_cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return {row["category"]: row["count"] for row in rows}
    except Exception as e:
        st.error(f"Database error: {e}")
        return {}


def get_date_range() -> Tuple[Optional[date], Optional[date]]:
    """Get the min and max dates in the dataset."""
    query = """
        SELECT MIN(created_at)::date as min_date, MAX(created_at)::date as max_date
        FROM reports
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


@st.cache_data(ttl=settings.cache_ttl_seconds)
def fetch_traffy_data(
    categories: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 100000,
) -> pd.DataFrame:
    """
    Fetch traffy_data from the database with PostGIS geometry extraction.

    Args:
        categories: List of category types to filter
        date_from: Start date filter
        date_to: End date filter
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

