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
class GeminiConfig:
  """Gemini API configuration."""
  model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
  api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))


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
  default_point_radius: int = 15
  default_opacity: float = 0.8
  default_elevation_scale: int = 4

  # Performance limits
  max_points_scatter: int = 10000
  max_points_heatmap: int = 50000
  hexagon_radius: int = 200


@dataclass
class LongdoConfig:
  """Longdo Events configuration."""

  event_types: List[str] = field(
      default_factory=lambda: [
          "roadclosed",
          "fire",
          "information",
          "trafficjam",
          "event",
          "rain",
          "complaint",
          "flood",
          "carbreakdown",
          "warning",
          "accident",
      ]
  )

  # Display names for event types
  event_type_labels: Dict[str, str] = field(
      default_factory=lambda: {
          "roadclosed": "🚧 Road Closed",
          "fire": "🔥 Fire",
          "information": "ℹ️ Information",
          "trafficjam": "🚗 Traffic Jam",
          "event": "📅 Event",
          "rain": "🌧️ Rain",
          "complaint": "📢 Complaint",
          "flood": "🌊 Flood",
          "carbreakdown": "🚘 Car Breakdown",
          "warning": "⚠️ Warning",
          "accident": "💥 Accident",
      }
  )

  # Color mapping for event types (RGBA)
  colors: Dict[str, Tuple[int, int, int, int]] = field(
      default_factory=lambda: {
          "roadclosed": (255, 87, 34, 200),      # Deep orange
          "fire": (244, 67, 54, 200),            # Red
          "information": (33, 150, 243, 200),   # Blue
          "trafficjam": (255, 152, 0, 200),     # Orange
          "event": (156, 39, 176, 200),         # Purple
          "rain": (3, 169, 244, 200),           # Light blue
          "complaint": (233, 30, 99, 200),      # Pink
          "flood": (0, 188, 212, 200),          # Cyan
          "carbreakdown": (121, 85, 72, 200),   # Brown
          "warning": (255, 235, 59, 200),       # Yellow
          "accident": (183, 28, 28, 200),       # Dark red
      }
  )


@dataclass
class CategoryConfig:
  """Traffy Fondue report categories (ticket types)."""

  # Special category for tickets without type
  unspecified_type: str = "ไม่ระบุ"

  categories: List[str] = field(
      default_factory=lambda: [
          "PM2.5",
          "กีดขวาง",
          "การเดินทาง",
          "คนจรจัด",
          "คลอง",
          "ความปลอดภัย",
          "ความสะอาด",
          "จราจร",
          "ต้นไม้",
          "ถนน",
          "ทางเท้า",
          "ท่อระบายน้ำ",
          "น้ำท่วม",
          "ป้าย",
          "ป้ายจราจร",
          "ร้องเรียน",
          "สอบถาม",
          "สะพาน",
          "สัตว์จรจัด",
          "สายไฟ",
          "ห้องน้ำ",
          "เสนอแนะ",
          "เสียงรบกวน",
          "แสงสว่าง",
          "ไม่ระบุ",  # For tickets without type
      ]
  )

  # Color mapping for categories (RGBA)
  colors: Dict[str, Tuple[int, int, int, int]] = field(
      default_factory=lambda: {
          "PM2.5": (128, 128, 128, 200),
          "กีดขวาง": (255, 140, 0, 200),
          "การเดินทาง": (100, 149, 237, 200),
          "คนจรจัด": (255, 182, 193, 200),
          "คลอง": (0, 191, 255, 200),
          "ความปลอดภัย": (220, 20, 60, 200),
          "ความสะอาด": (139, 69, 19, 200),
          "จราจร": (255, 69, 0, 200),
          "ต้นไม้": (34, 139, 34, 200),
          "ถนน": (255, 99, 71, 200),
          "ทางเท้า": (50, 205, 50, 200),
          "ท่อระบายน้ำ": (70, 130, 180, 200),
          "น้ำท่วม": (30, 144, 255, 200),
          "ป้าย": (255, 215, 0, 200),
          "ป้ายจราจร": (255, 165, 0, 200),
          "ร้องเรียน": (178, 34, 34, 200),
          "สอบถาม": (147, 112, 219, 200),
          "สะพาน": (128, 128, 128, 200),
          "สัตว์จรจัด": (210, 105, 30, 200),
          "สายไฟ": (255, 215, 0, 200),
          "ห้องน้ำ": (0, 206, 209, 200),
          "เสนอแนะ": (60, 179, 113, 200),
          "เสียงรบกวน": (148, 0, 211, 200),
          "แสงสว่าง": (255, 255, 0, 200),
          "ไม่ระบุ": (169, 169, 169, 200),  # Gray for unspecified
      }
  )


@dataclass
class Settings:
  """Main application settings."""

  db: DatabaseConfig = field(default_factory=DatabaseConfig)
  map: MapConfig = field(default_factory=MapConfig)
  categories: CategoryConfig = field(default_factory=CategoryConfig)
  longdo: LongdoConfig = field(default_factory=LongdoConfig)
  gemini: GeminiConfig = field(default_factory=GeminiConfig)

  # Cache settings
  cache_ttl_seconds: int = 300  # 5 minutes

  # Layer types available
  layer_types: List[str] = field(
      default_factory=lambda: ["scatter", "heatmap", "hexagon", "cluster"]
  )

  # Data sources available
  data_sources: List[str] = field(
      default_factory=lambda: ["traffy", "longdo"]
  )


# Singleton settings instance
settings = Settings()
