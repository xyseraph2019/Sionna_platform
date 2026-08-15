
"""Backward-compatibility shim for the old plugins package.

New code should import :mod:`examples.custom_components` directly.
"""
from examples.custom_components import *  # noqa: F401,F403
