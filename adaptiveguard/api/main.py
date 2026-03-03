"""API entrypoint for AdaptiveGuard."""

from fastapi import FastAPI

app = FastAPI(title="AdaptiveGuard API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness endpoint for service checks."""
    return {"status": "ok"}
