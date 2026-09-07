import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path so 'app' can be imported seamlessly on Vercel
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.main import app

# Universal Vercel Python runtime compatibility
handler = app
