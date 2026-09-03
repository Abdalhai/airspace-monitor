#!/usr/bin/env python3
"""Launcher, so the tool runs as `python fod.py ...` with no install step.

A committee laptop may not allow `pip install -e .`, and asking a supervisor to
configure a PYTHONPATH mid-demo is not a plan. This file makes the package
importable from wherever it happens to sit.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fod.cli import main

if __name__ == "__main__":
    sys.exit(main())
