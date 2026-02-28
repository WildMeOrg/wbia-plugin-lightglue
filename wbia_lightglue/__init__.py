# -*- coding: utf-8 -*-
__version__ = '0.1.0'

try:
    from wbia_lightglue import _plugin  # noqa: F401 — register depc tables and ibs methods
except ImportError:
    # WBIA not installed — core module still importable for testing
    pass
