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

# Automatically load .env file into environment
from pathlib import Path
import os
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_path)
    except Exception:
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


