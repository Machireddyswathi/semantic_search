# main.py
"""
Application entry point.

Run directly:
    python main.py

Or via uvicorn CLI (recommended for production):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

WHY --workers 1 (not 4):
    Our components (FAISS index, GMM model, semantic cache) live
    in memory. With multiple workers (separate processes), each
    worker would load its OWN copy — multiplying RAM usage by N
    with NO benefit for CPU-bound numpy operations.

    For true horizontal scaling: use a load balancer + multiple
    single-worker containers (Docker Compose / Kubernetes).
"""

import uvicorn
from utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Semantic Search System...")
    logger.info("API docs available at: http://localhost:8000/docs")
    logger.info("ReDoc available at:    http://localhost:8000/redoc")

    uvicorn.run(
        "api.app:app",       # module_path:app_variable
        host="0.0.0.0",      # bind all interfaces (needed for Docker)
        port=8000,
        reload=False,        # WHY False: reload re-initializes ALL models
                             # on every file change — too expensive.
                             # Use True only during schema/route development.
        log_level="info",
        access_log=True,
    )