"""Your Assistant - Automated AI Job Search Backend Package."""
import json
import sys
import types

# Ensure environments with DLL security policies can import ChromaDB cleanly
try:
    import orjson  # noqa: F401
except ImportError:
    _shim = types.ModuleType("orjson")
    _shim.dumps = lambda obj, *a, **k: json.dumps(obj).encode("utf-8")
    _shim.loads = lambda s, *a, **k: json.loads(s)
    _shim.OPT_NON_STR_KEYS = 0
    sys.modules["orjson"] = _shim
