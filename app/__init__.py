"""Your Assistant - Automated AI Job Search Backend Package."""
import json
import sys
import types
from typing import Any

# Ensure environments with DLL security policies can import ChromaDB cleanly
try:
    __import__("orjson")
except Exception:
    _shim: Any = types.ModuleType("orjson")
    setattr(_shim, "dumps", lambda obj, *a, **k: json.dumps(obj).encode("utf-8"))
    setattr(_shim, "loads", lambda s, *a, **k: json.loads(s))
    setattr(_shim, "OPT_NON_STR_KEYS", 0)
    sys.modules["orjson"] = _shim

