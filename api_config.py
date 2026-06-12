"""
api_config.py  —  API REST de configuration ML, stockée dans SQLite.

Base de données : priceconfig.db  (même dossier)
Lance avec      : python api_config.py

Endpoints :
  GET  /api/config            → configuration complète en JSON
  POST /api/config            → mise à jour partielle (merge)
  POST /api/config/reset      → remet les valeurs par défaut
  GET  /api/config/audit      → journal des 100 dernières modifications
  GET  /api/config/<key>      → valeur d'une seule clé
  GET  /api/health            → statut de la DB
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── S'assurer que src/ est dans le path ──────────────────────────────────────
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flask import Flask, jsonify, request, abort
from flask_cors import CORS

from db import ConfigDB             # ← couche DB
from db import DEFAULTS             # ← pour validation des clés

app = Flask(__name__)
CORS(app)

# Instance unique de la DB (thread-safe via WAL + contextmanager)
# _db = ConfigDB()


# ─── Sérialisation JSON-safe ──────────────────────────────────────────────────

def _jsonify_cfg(cfg: dict):
    """
    Prépare le dict config pour jsonify :
      - les dicts de multiplicateurs restent tels quels (clés déjà en str)
      - les booléens sont sérialisés nativement
    """
    return {k: v for k, v in cfg.items()}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Vérifie que la DB répond."""
    return jsonify({"status": "ok"})


@app.get("/api/config")
def get_config():
    """Retourne toute la configuration."""
    return jsonify(_jsonify_cfg(_db.load()))


@app.get("/api/config/<string:key>")
def get_config_key(key: str):
    """Retourne la valeur d'une seule clé."""
    cfg = _db.load()
    if key not in cfg:
        abort(404, description=f"Clé inconnue : {key}")
    return jsonify({key: cfg[key]})


@app.post("/api/config")
def update_config():
    """
    Mise à jour partielle.
    Body JSON : { "BASE_FARE": 7.5, "MULT_CAR": { "economy": 0.8 }, ... }
    """
    payload = request.get_json(force=True, silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Corps JSON requis"}), 400

    # Validation des clés
    known_keys = set(DEFAULTS.keys())
    unknown = set(payload.keys()) - known_keys
    if unknown:
        return jsonify({"error": f"Clés inconnues : {sorted(unknown)}"}), 400

    try:
        updated = _db.save(payload, changed_by="api")
        return jsonify({"status": "ok", "config": _jsonify_cfg(updated)})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"error": "Erreur interne", "detail": str(exc)}), 500


@app.post("/api/config/reset")
def reset_config():
    """Remet toutes les valeurs par défaut."""
    try:
        cfg = _db.reset()
        return jsonify({"status": "reset", "config": _jsonify_cfg(cfg)})
    except Exception as exc:
        return jsonify({"error": "Erreur lors du reset", "detail": str(exc)}), 500


@app.get("/api/config/audit")
def get_audit():
    """Retourne les 100 dernières modifications (paramètre ?limit=N pour ajuster)."""
    try:
        limit = min(int(request.args.get("limit", 100)), 1000)
    except ValueError:
        limit = 100
    return jsonify(_db.get_audit_log(limit=limit))


# ─── Gestion des erreurs ──────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Méthode non autorisée"}), 405


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅  API Config  →  http://localhost:5000/api/config")
    print("📋  Audit log   →  http://localhost:5000/api/config/audit")
    print("❤️   Health      →  http://localhost:5000/api/health")
    app.run(debug=True, port=5000)
