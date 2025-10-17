"""Distillation helpers for DEIM."""

from .deim_distiller import DEIMFGDDistiller
from .fgd import FGDFeatureLoss

__all__ = ['DEIMFGDDistiller', 'FGDFeatureLoss']
