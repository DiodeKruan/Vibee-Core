# Data module
from .database import get_connection, close_pool
from .queries import fetch_reports, fetch_reports_aggregated, fetch_category_counts

__all__ = [
    "get_connection",
    "close_pool",
    "fetch_reports",
    "fetch_reports_aggregated",
    "fetch_category_counts",
]

