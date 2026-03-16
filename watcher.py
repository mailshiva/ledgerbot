#!/usr/bin/env python3
"""
Top-level entry point for the statement watcher.
Usage: python watcher.py [folder] [--dry-run]
See src/watcher.py for full documentation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.watcher import main

if __name__ == "__main__":
    main()
