"""Pytest configuration for d2p tests."""

import sys
from pathlib import Path

# Add tests directory to path so reference.py can be imported
sys.path.insert(0, str(Path(__file__).parent))

# Add project root to path so d2p can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
