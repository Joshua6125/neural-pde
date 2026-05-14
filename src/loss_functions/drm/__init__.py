"""Deep Ritz Method algorithm module."""

from .config import DRMConfig
from .loss import DRMLoss
from .method import DRM

__all__ = [
    "DRMConfig",
    "DRMLoss",
    "DRM",
]
