"""Shared bootstrap helpers for example scripts.

Running ``python examples/xxx.py`` directly does not automatically put the
platform root on ``sys.path``; this tiny module does that once so every example
can avoid repeating the path-insert boilerplate.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
