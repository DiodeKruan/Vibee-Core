"""Data MCP interface for chatbot database queries."""

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import settings


@dataclass
class DataQuery:
    """Represents a data query from the chatbot."""

    query_type: str  # 'reports', 'summary', 'categories', 'time_series'
    categories: Optional[List[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    bbox: Optional[tuple] = None
    limit: int = 1000


@dataclass
class DataResponse:
    """Response from data query."""

    success: bool
    data: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None


class DataMCP:
    """
    Data MCP interface for natural language database queries.
    
    This stub provides the interface that will be implemented
    when connecting to the actual MCP server.
    """

    def __init__(self):
        """Initialize Data MCP interface."""
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful
        """
        # Stub: Always return True for now
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected

    def parse_natural_query(self, text: str) -> DataQuery:
        """
        Parse natural language into a DataQuery.

        Args:
            text: Natural language query from user

        Returns:
            DataQuery object

        Examples:
            "Show reports from last week" -> DataQuery with date filter
            "Filter by roads and flooding" -> DataQuery with category filter
            "How many reports this month?" -> DataQuery for summary
        """
        # Stub implementation - returns default query
        # Real implementation would use NLP/LLM to parse

        query = DataQuery(query_type="reports")

        # Simple keyword matching for demo
        text_lower = text.lower()

        # Check for category keywords
        matched_categories = []
        for category in settings.categories.categories:
            if category.lower() in text_lower:
                matched_categories.append(category)

        if matched_categories:
            query.categories = matched_categories

        # Check for summary keywords
        if any(word in text_lower for word in ["how many", "count", "total", "summary"]):
            query.query_type = "summary"

        # Check for time series keywords
        if any(word in text_lower for word in ["trend", "over time", "timeline", "history"]):
            query.query_type = "time_series"

        return query

    def execute_query(self, query: DataQuery) -> DataResponse:
        """
        Execute a data query.

        Args:
            query: DataQuery to execute

        Returns:
            DataResponse with results
        """
        # Stub implementation - returns mock data
        # Real implementation would query the database via MCP

        if not self._connected:
            return DataResponse(
                success=False,
                error="Not connected to MCP server",
            )

        # Mock response based on query type
        if query.query_type == "summary":
            return DataResponse(
                success=True,
                summary={
                    "total_reports": 15000,
                    "categories": len(query.categories or settings.categories.categories),
                    "date_range": "Last 30 days",
                },
                message="Summary retrieved successfully",
            )

        elif query.query_type == "reports":
            # Return empty DataFrame with correct schema
            return DataResponse(
                success=True,
                data=pd.DataFrame(
                    columns=["id", "lat", "lon", "category", "created_at", "description", "status"]
                ),
                message=f"Query would fetch reports with filters: {query}",
            )

        return DataResponse(
            success=False,
            error=f"Unknown query type: {query.query_type}",
        )

    def get_available_actions(self) -> List[Dict[str, str]]:
        """
        Get list of available data actions for the chatbot.

        Returns:
            List of action descriptions
        """
        return [
            {
                "action": "fetch_reports",
                "description": "Fetch report data with optional filters",
                "params": ["categories", "date_from", "date_to", "bbox", "limit"],
            },
            {
                "action": "get_summary",
                "description": "Get summary statistics",
                "params": ["categories", "date_from", "date_to"],
            },
            {
                "action": "get_categories",
                "description": "List available categories with counts",
                "params": [],
            },
            {
                "action": "get_time_series",
                "description": "Get report counts over time",
                "params": ["categories", "date_from", "date_to", "granularity"],
            },
        ]

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


# Singleton instance
data_mcp = DataMCP()

