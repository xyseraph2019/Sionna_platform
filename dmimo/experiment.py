"""
Deprecated: use :mod:`dmimo.link` instead.

This module is kept for backward compatibility. The same functions now live in
``dmimo.link`` (``build_link``, ``evaluate_precoder``, ``generate_dataset``,
``save_dataset``, ``load_dataset``).
"""
from .link import (
    build_link,
    evaluate_precoder,
    generate_dataset,
    load_dataset,
    save_dataset,
)

__all__ = [
    "build_link",
    "evaluate_precoder",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]
