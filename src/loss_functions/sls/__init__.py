"""Least-Squares algorithm module."""

from .config import SLSConfig
from .loss import SLSLoss
from .method import SLS

__all__ = [
    "SLSConfig",
    "SLSLoss",
    "SLS",
]
