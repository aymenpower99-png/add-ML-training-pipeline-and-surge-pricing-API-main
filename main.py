
# main.py — Backward-compatible entry point
# Run:  uvicorn main:app --reload --port 8000
#
# This re-exports the full Moviroo Pricing API from api/app.py
# so ALL endpoints work regardless of which entry point you use:
#   /price/estimate, /price/quick, /health, /vehicles, /zones

from api.app import app  # noqa: F401