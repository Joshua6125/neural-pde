"""Least-Squares algorithm module."""

from .config import FOSLSConfig
from .loss import FOSLSLoss
from .method import FOSLS

__all__ = [
    "FOSLSConfig",
    "FOSLSLoss",
    "FOSLS",
]
