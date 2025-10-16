"""
CLI Startup Script

Quick startup script for the Multimodal-DB CLI.

Usage:
    python run_cli.py [command] [options]
    python run_cli.py agent list
    python run_cli.py query run "How many agents are there?"

Author: Multimodal-DB Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add multimodal-db to path
sys.path.insert(0, str(Path(__file__).parent / "multimodal-db"))

from cli.cli import cli

if __name__ == "__main__":
    cli()
