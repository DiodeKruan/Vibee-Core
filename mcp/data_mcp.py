"""Data MCP interface for chatbot database queries."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings


@dataclass
class DataQuery:
    """Represents a data query from the chatbot."""

    query_type: str  # 'schema', 'aggregation', 'statistics', 'time_series', 'crosstab', 'flexible', 'ticket_detail', 'reports'
    
    # Filter parameters
    categories: Optional[List[str]] = None
    districts: Optional[List[str]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    filters: Optional[Dict[str, Any]] = None
    
    # Aggregation parameters
    group_by: Optional[str] = None
    dimension: Optional[str] = None
    dimension_value: Optional[str] = None
    
    # Time series parameters
    granularity: str = "day"
    metric: str = "count"
    
    # Crosstab parameters
    row_dimension: Optional[str] = None
    col_dimension: Optional[str] = None
    
    # Flexible query parameters
    select_columns: Optional[List[str]] = None
    aggregations: Optional[List[Dict[str, str]]] = None
    order_by: Optional[str] = None
    
    # Detail query parameters
    ticket_id: Optional[str] = None
    
    # General parameters
    limit: int = 1000
    days_back: int = 30


@dataclass
class DataResponse:
    """Response from data query."""

    success: bool
    data: Optional[Any] = None  # Can be DataFrame, List, or Dict
    summary: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None


class DataMCP:
    """
    Data MCP interface for programmatic database queries.
    
    This provides a structured interface for the chatbot to:
    - Query aggregated ticket data
    - Get statistics and summaries
    - Fetch time series data
    - Execute flexible analytical queries
    - Filter by district, category, date, and bounding box
    """

    def __init__(self):
        """Initialize Data MCP interface."""
        self._connected = False
        self._query_handlers: Dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up query handlers."""
        self._query_handlers = {
            "schema": self._handle_schema,
            "aggregation": self._handle_aggregation,
            "statistics": self._handle_statistics,
            "time_series": self._handle_time_series,
            "crosstab": self._handle_crosstab,
            "flexible": self._handle_flexible,
            "ticket_detail": self._handle_ticket_detail,
            "reports": self._handle_reports,
        }

    def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful
        """
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected

    def execute_query(self, query: DataQuery) -> DataResponse:
        """
        Execute a data query.

        Args:
            query: DataQuery to execute

        Returns:
            DataResponse with results
        """
        if not self._connected:
            return DataResponse(
                success=False,
                error="Not connected to MCP server",
            )

        handler = self._query_handlers.get(query.query_type)
        if handler:
            return handler(query)

        return DataResponse(
            success=False,
            error=f"Unknown query type: {query.query_type}",
        )

    def _build_filters(self, query: DataQuery) -> Dict[str, Any]:
        """Build filter dictionary from query parameters."""
        filters = query.filters.copy() if query.filters else {}
        
        if query.categories:
            filters["type"] = query.categories
        if query.districts:
            filters["district"] = query.districts
        if query.date_from:
            filters["date_from"] = query.date_from
        if query.date_to:
            filters["date_to"] = query.date_to
            
        return filters

    def _handle_schema(self, query: DataQuery) -> DataResponse:
        """Handle schema query."""
        try:
            from data.queries import get_schema_info
            
            schema = get_schema_info()
            return DataResponse(
                success=True,
                data=schema,
                message="Schema retrieved successfully",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_aggregation(self, query: DataQuery) -> DataResponse:
        """Handle aggregation query."""
        try:
            from data.queries import query_aggregation
            
            if not query.group_by:
                return DataResponse(
                    success=False,
                    error="group_by is required for aggregation queries",
                )
            
            filters = self._build_filters(query)
            
            results = query_aggregation(
                group_by=query.group_by,
                filters=filters if filters else None,
                order_by=query.order_by or "count_desc",
                limit=query.limit,
                bbox=query.bbox,
            )
            
            return DataResponse(
                success=True,
                data=results,
                message=f"Aggregation by {query.group_by} completed",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_statistics(self, query: DataQuery) -> DataResponse:
        """Handle statistics query."""
        try:
            from data.queries import query_statistics
            
            filters = self._build_filters(query)
            
            stats = query_statistics(
                dimension=query.dimension,
                dimension_value=query.dimension_value,
                filters=filters if filters else None,
                bbox=query.bbox,
            )
            
            return DataResponse(
                success=True,
                data=stats,
                summary=stats,
                message="Statistics retrieved successfully",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_time_series(self, query: DataQuery) -> DataResponse:
        """Handle time series query."""
        try:
            from data.queries import query_time_series
            from datetime import timedelta
            
            filters = self._build_filters(query)
            
            # Calculate date range
            date_from = query.date_from or (date.today() - timedelta(days=query.days_back))
            date_to = query.date_to or date.today()
            
            results = query_time_series(
                granularity=query.granularity,
                metric=query.metric,
                group_by=query.group_by,
                filters=filters if filters else None,
                date_from=date_from,
                date_to=date_to,
                limit=query.limit,
                bbox=query.bbox,
            )
            
            return DataResponse(
                success=True,
                data=results,
                message=f"Time series ({query.granularity}) retrieved",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_crosstab(self, query: DataQuery) -> DataResponse:
        """Handle crosstab query."""
        try:
            from data.queries import query_crosstab
            
            if not query.row_dimension or not query.col_dimension:
                return DataResponse(
                    success=False,
                    error="row_dimension and col_dimension are required for crosstab queries",
                )
            
            filters = self._build_filters(query)
            
            results = query_crosstab(
                row_dimension=query.row_dimension,
                col_dimension=query.col_dimension,
                metric=query.metric,
                filters=filters if filters else None,
                bbox=query.bbox,
            )
            
            return DataResponse(
                success=True,
                data=results,
                message=f"Crosstab {query.row_dimension} x {query.col_dimension} completed",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_flexible(self, query: DataQuery) -> DataResponse:
        """Handle flexible analytical query."""
        try:
            from data.queries import execute_flexible_query
            
            if not query.select_columns:
                return DataResponse(
                    success=False,
                    error="select_columns is required for flexible queries",
                )
            
            filters = self._build_filters(query)
            
            # Parse group_by if it's a string
            group_by_list = None
            if query.group_by:
                group_by_list = [g.strip() for g in query.group_by.split(",")]
            
            results = execute_flexible_query(
                select_columns=query.select_columns,
                aggregations=query.aggregations,
                filters=filters if filters else None,
                group_by=group_by_list,
                order_by=query.order_by,
                limit=query.limit,
                bbox=query.bbox,
            )
            
            return DataResponse(
                success=True,
                data=results,
                message=f"Query returned {len(results)} rows",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_ticket_detail(self, query: DataQuery) -> DataResponse:
        """Handle ticket detail query."""
        try:
            from data.queries import query_ticket_details
            
            if not query.ticket_id:
                return DataResponse(
                    success=False,
                    error="ticket_id is required for detail queries",
                )
            
            details = query_ticket_details(query.ticket_id)
            
            if not details:
                return DataResponse(
                    success=False,
                    error=f"No ticket found with ID: {query.ticket_id}",
                )
            
            return DataResponse(
                success=True,
                data=details,
                message=f"Ticket {query.ticket_id} details retrieved",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def _handle_reports(self, query: DataQuery) -> DataResponse:
        """Handle reports fetch query."""
        try:
            from data.queries import fetch_traffy_data
            
            df = fetch_traffy_data(
                categories=query.categories,
                districts=query.districts,
                date_from=query.date_from,
                date_to=query.date_to,
                bbox=query.bbox,
                limit=query.limit,
            )
            
            return DataResponse(
                success=True,
                data=df,
                summary={
                    "total_reports": len(df),
                    "categories": df["type"].nunique() if "type" in df.columns and not df.empty else 0,
                },
                message=f"Fetched {len(df)} reports",
            )
        except Exception as e:
            return DataResponse(success=False, error=str(e))

    def parse_natural_query(self, text: str) -> DataQuery:
        """
        Parse natural language into a DataQuery.

        Args:
            text: Natural language query from user

        Returns:
            DataQuery object
        """
        query = DataQuery(query_type="reports")
        text_lower = text.lower()

        # Check for category keywords
        matched_categories = []
        for category in settings.categories.categories:
            if category.lower() in text_lower:
                matched_categories.append(category)

        if matched_categories:
            query.categories = matched_categories

        # Check for query type keywords
        if any(word in text_lower for word in ["schema", "columns", "structure"]):
            query.query_type = "schema"
        elif any(word in text_lower for word in ["how many", "count", "total", "summary", "statistics"]):
            query.query_type = "statistics"
        elif any(word in text_lower for word in ["trend", "over time", "timeline", "history", "daily", "monthly"]):
            query.query_type = "time_series"
        elif any(word in text_lower for word in ["group by", "breakdown", "by district", "by type"]):
            query.query_type = "aggregation"

        return query

    def natural_query(self, text: str) -> DataResponse:
        """
        Process a natural language query end-to-end.

        Args:
            text: Natural language query

        Returns:
            DataResponse with results
        """
        query = self.parse_natural_query(text)
        return self.execute_query(query)

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """
        Get list of available data actions for the chatbot.

        Returns:
            List of action descriptions
        """
        return [
            {
                "action": "schema",
                "description": "Get database schema and available dimensions",
                "params": [],
            },
            {
                "action": "aggregation",
                "description": "Get counts grouped by a dimension",
                "params": ["group_by", "filters", "districts", "bbox", "order_by", "limit"],
            },
            {
                "action": "statistics",
                "description": "Get statistical summary",
                "params": ["dimension", "dimension_value", "filters", "districts", "bbox"],
            },
            {
                "action": "time_series",
                "description": "Get time series data",
                "params": ["granularity", "metric", "group_by", "filters", "districts", "bbox", "days_back"],
            },
            {
                "action": "crosstab",
                "description": "Get cross-tabulation of two dimensions",
                "params": ["row_dimension", "col_dimension", "metric", "filters", "bbox"],
            },
            {
                "action": "flexible",
                "description": "Execute flexible analytical query",
                "params": ["select_columns", "aggregations", "filters", "group_by", "order_by", "limit", "bbox"],
            },
            {
                "action": "ticket_detail",
                "description": "Get details of a specific ticket",
                "params": ["ticket_id"],
            },
            {
                "action": "reports",
                "description": "Fetch report data with filters",
                "params": ["categories", "districts", "date_from", "date_to", "bbox", "limit"],
            },
        ]


# Singleton instance
data_mcp = DataMCP()
