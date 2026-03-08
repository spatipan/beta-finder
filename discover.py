#!/usr/bin/env python3
"""
Backward-compatible wrapper for src/discover.py
Maintains CLI compatibility with original location
"""

if __name__ == "__main__":
    from src.discover import main
    main()
