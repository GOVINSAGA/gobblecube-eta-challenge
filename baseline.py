#!/usr/bin/env python
"""Compatibility entrypoint.

The starter trained an XGBoost baseline from this file. The submission now uses
``train_model.py`` for the actual model, so this wrapper keeps the documented
``python baseline.py`` command working.
"""

from __future__ import annotations

from train_model import main


if __name__ == "__main__":
    main()
