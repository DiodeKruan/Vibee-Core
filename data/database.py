"""PostgreSQL database connection management with connection pooling."""

from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from config.settings import settings

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def init_pool() -> pool.ThreadedConnectionPool:
    """Initialize the connection pool."""
    global _connection_pool

    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=settings.db.min_connections,
            maxconn=settings.db.max_connections,
            host=settings.db.host,
            port=settings.db.port,
            database=settings.db.database,
            user=settings.db.user,
            password=settings.db.password,
        )

    return _connection_pool


def get_pool() -> pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    global _connection_pool

    if _connection_pool is None:
        return init_pool()

    return _connection_pool


@contextmanager
def get_connection() -> Generator:
    """
    Context manager for database connections.
    Automatically returns connection to pool after use.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM reports")
    """
    conn = None
    try:
        conn = get_pool().getconn()
        yield conn
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            get_pool().putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = True) -> Generator:
    """
    Context manager for database cursors.
    Automatically handles connection and cursor lifecycle.

    Args:
        dict_cursor: If True, returns results as dictionaries

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM reports")
            rows = cur.fetchall()
    """
    with get_connection() as conn:
        cursor_factory = RealDictCursor if dict_cursor else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur
            conn.commit()


def close_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool

    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


def test_connection() -> bool:
    """Test database connectivity."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception:
        return False

