"""OpenWebUI Server Entry Point"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run openwebui server
from openwebui_server import server

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        server.app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8001))
    )
