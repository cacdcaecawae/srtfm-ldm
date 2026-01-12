"""Pytest configuration file - ensures root directory is in Python path."""
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
