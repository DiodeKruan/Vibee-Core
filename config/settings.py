"""Application configuration settings."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", "vibee"))
    user: str = field(default_factory=lambda: os.getenv("DB_USERNAME", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgres"))
    min_connections: int = 1
    max_connections: int = 10


@dataclass
class MapConfig:
    """PyDeck map configuration."""

    # Bangkok center coordinates
    center_lat: float = 13.7563
    center_lon: float = 100.5018
    default_zoom: int = 11
    default_pitch: int = 30
    default_bearing: int = 0

    # Layer defaults
    default_layer: str = "scatter"
    default_point_radius: int = 100
    default_opacity: float = 0.8
    default_elevation_scale: int = 4

    # Performance limits
    max_points_scatter: int = 10000
    max_points_heatmap: int = 50000
    hexagon_radius: int = 200


@dataclass
class CategoryConfig:
    """Traffy Fondue report categories."""

    categories: List[str] = field(
        default_factory=lambda: [
            "ถนน",
            "ทางเท้า",
            "ไฟฟ้า",
            "น้ำท่วม",
            "ขยะ",
            "ต้นไม้",
            "ท่อระบายน้ำ",
            "ป้าย",
            "สะพาน",
            "อื่นๆ",
        ]
    )

    # Color mapping for categories (RGBA)
    colors: Dict[str, Tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "ถนน": (255, 99, 71, 200),
            "ทางเท้า": (50, 205, 50, 200),
            "ไฟฟ้า": (255, 215, 0, 200),
            "น้ำท่วม": (30, 144, 255, 200),
            "ขยะ": (139, 69, 19, 200),
            "ต้นไม้": (34, 139, 34, 200),
            "ท่อระบายน้ำ": (70, 130, 180, 200),
            "ป้าย": (255, 140, 0, 200),
            "สะพาน": (128, 128, 128, 200),
            "อื่นๆ": (148, 0, 211, 200),
        }
    )


@dataclass
class Settings:
    """Main application settings."""

    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    map: MapConfig = field(default_factory=MapConfig)
    categories: CategoryConfig = field(default_factory=CategoryConfig)

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes

    # Layer types available
    layer_types: List[str] = field(
        default_factory=lambda: ["scatter", "heatmap", "hexagon", "icon", "cluster"]
    )


# Singleton settings instance
settings = Settings()

