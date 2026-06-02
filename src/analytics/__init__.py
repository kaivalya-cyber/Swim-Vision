# This file exposes SwimVision analytics modules as a package.
"""Analytics package for SwimVision."""

from src.analytics.trend import analyze_trends, SessionRecord

__all__ = ["analyze_trends", "SessionRecord"]
