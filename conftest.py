"""Ensures the repository root is importable so tests can ``import src``."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
