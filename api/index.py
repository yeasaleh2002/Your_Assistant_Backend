import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path so 'app' can be imported seamlessly on Vercel
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from app.main import app
    handler = app
except Exception as e:
    import traceback
    traceback.print_exc()
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    fallback_app = FastAPI(title="Your Assistant - Recovery Mode")

    @fallback_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def vercel_startup_error_handler(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Serverless Function Startup Failed",
                "detail": str(e),
                "traceback": traceback.format_exc(),
            },
        )

    handler = fallback_app
