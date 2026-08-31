"""Pytest configuration for the Plant Health Recognition Engine tests."""
import sys
from pathlib import Path

# Add project root to sys.path so tests can import modules directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
