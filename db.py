"""
db.py  —  Couche d'accès PostgreSQL pour la base priceconfig.

Usage :
    from db import ConfigDB
    db = ConfigDB()          # ouvre / crée les tables
    cfg = db.load()          # → dict complet
    db.save(patch_dict)      # sauvegarde partielle ou complète
    db.reset()               # remet les valeurs par défaut

Variables d'environnement requises (ou fichier .env) :
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

from __future__ import annotations

import os
import json
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
def _dsn() -> str:
    return (
        f"host={os.environ['DB_HOST']} "
        f"port={os.environ.get('DB_PORT', 5432)} "
        f"dbname={os.environ['DB_NAME']} "
        f"user={os.environ['DB_USER']} "
        f"password={os.environ['DB_PASSWORD']}"
    )


# ── Valeurs par défaut (source unique de vérité) ──────────────────────────────
DEFAULTS: dict = {
    # ── Scalaires numériques ──────────────────────────────────────────────────
    "BASE_FARE":          6.0,
    "RATE_PER_KM":        0.65,
    "RATE_PER_MIN":       0.3,
    "MIN_FARE":           4.0,
    "W_XGB":              0.55,
    "W_LGBM":             0.45,
    "MULT_NIGHT":         2.2,
    "MULT_FRIDAY_JUMUAH": 1.4,
    # ── Booléens feature-flags ────────────────────────────────────────────────
    "ENABLE_TRAFFIC":       True,
    "ENABLE_WEATHER":       True,
    "ENABLE_DEMAND":        True,
    "ENABLE_NIGHT":         True,
    "ENABLE_FRIDAY_JUMUAH": True,
    "ENABLE_RAMADAN":       True,
    "ENABLE_BEACH":         True,
    "ENABLE_ZONE":          True,
    "ENABLE_SPECIAL_EVENT": True,
    "ENABLE_SEASON":        True,
    # ── Multiplicateurs (dicts) ───────────────────────────────────────────────
    "MULT_TRAFFIC":       {"1": 1.0,   "2": 1.2,   "3": 1.5},
    "MULT_WEATHER":       {"1": 1.2,   "2": 2.1,   "3": 1.3,  "4": 1.1},
    "MULT_DEMAND":        {"normal": 1.0, "rush": 1.25, "surge": 1.6},
    "MULT_CAR":           {"economy": 0.75, "standard": 0.9, "comfort": 1.0,
                           "first_class": 1.6, "van": 1.3, "mini_bus": 1.5},
    "MULT_RAMADAN":       {"ramadan_iftar": 2.1, "ramadan_tarawih": 1.3,
                           "ramadan_suhoor": 1.15, "ramadan_last_week": 1.6, "none": 1.0},
    "MULT_BEACH":         {"afflux_matin": 1.25, "après_midi": 1.3,
                           "coucher_soleil": 1.35, "none": 1.0},
    "MULT_ZONE":          {"capitale": 1.15, "banlieue": 1.05, "balnéaire": 1.1,
                           "intérieure": 1.0, "sud": 0.95},
    "MULT_SPECIAL_EVENT": {"aid_el_fitr": 2.0, "aid_el_adha_week": 1.8,
                           "new_year_eve": 1.9, "new_year_days": 1.4, "none": 1.0},
    # ── Hyperparamètres modèles ───────────────────────────────────────────────
    "XGB_PARAMS": {
        "objective": "reg:squarederror", "n_estimators": 800, "max_depth": 7,
        "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    },
    "LGBM_PARAMS": {
        "objective": "regression", "n_estimators": 800, "max_depth": 7,
        "learning_rate": 0.04, "num_leaves": 63, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_samples": 10, "reg_alpha": 0.1,
        "reg_lambda": 1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    },
}

# ── Ensembles de clés par type ────────────────────────────────────────────────
_SCALAR_REAL_KEYS = {
    "BASE_FARE", "RATE_PER_KM", "RATE_PER_MIN", "MIN_FARE",
    "W_XGB", "W_LGBM", "MULT_NIGHT", "MULT_FRIDAY_JUMUAH",
}
_SCALAR_BOOL_KEYS = {
    "ENABLE_TRAFFIC", "ENABLE_WEATHER", "ENABLE_DEMAND", "ENABLE_NIGHT",
    "ENABLE_FRIDAY_JUMUAH", "ENABLE_RAMADAN", "ENABLE_BEACH", "ENABLE_ZONE",
    "ENABLE_SPECIAL_EVENT", "ENABLE_SEASON",
}
_MULT_KEYS = {
    "MULT_TRAFFIC", "MULT_WEATHER", "MULT_DEMAND", "MULT_CAR",
    "MULT_RAMADAN", "MULT_BEACH", "MULT_ZONE", "MULT_SPECIAL_EVENT",
}
_MODEL_KEYS = {"XGB_PARAMS", "LGBM_PARAMS"}


# ─────────────────────────────────────────────────────────────────────────────
class ConfigDB:
    """Gestion complète de la configuration dans PostgreSQL."""

    def __init__(self):
        self._dsn = _dsn()
        self._init_db()

    # ── Connexion ─────────────────────────────────────────────────────────────
    @contextmanager
    def _connect(self):
        con = psycopg2.connect(self._dsn)
        con.autocommit = False
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _cursor(self, con):
        """Curseur retournant des dicts (équivalent de row_factory de sqlite3)."""
        return con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_db(self):
        """Crée les tables si elles n'existent pas, puis seed si vide."""
        with self._connect() as con:
            cur = self._cursor(con)
            cur.execute(_SCHEMA_SQL)

        # Seed uniquement si la table scalaire est vide
        with self._connect() as con:
            cur = self._cursor(con)
            cur.execute("SELECT COUNT(*) AS cnt FROM scalar_params")
            if cur.fetchone()["cnt"] == 0:
                self._seed(con)

    def _seed(self, con):
        """Insère toutes les valeurs par défaut."""
        now = _now()
        cur = self._cursor(con)

        # Scalaires réels
        for k in _SCALAR_REAL_KEYS:
            cur.execute(
                """INSERT INTO scalar_params(key, value_real, value_type, updated_at)
                   VALUES(%s, %s, 'real', %s)
                   ON CONFLICT(key) DO NOTHING""",
                (k, float(DEFAULTS[k]), now),
            )

        # Scalaires booléens
        for k in _SCALAR_BOOL_KEYS:
            cur.execute(
                """INSERT INTO scalar_params(key, value_bool, value_type, updated_at)
                   VALUES(%s, %s, 'bool', %s)
                   ON CONFLICT(key) DO NOTHING""",
                (k, bool(DEFAULTS[k]), now),
            )

        # Multiplicateurs
        for group, mapping in DEFAULTS.items():
            if group not in _MULT_KEYS:
                continue
            for mk, mv in mapping.items():
                cur.execute(
                    """INSERT INTO multipliers(group_name, map_key, map_value, updated_at)
                       VALUES(%s, %s, %s, %s)
                       ON CONFLICT(group_name, map_key) DO NOTHING""",
                    (group, str(mk), float(mv), now),
                )

        # Hyperparamètres
        for model_name, params in [("XGB", DEFAULTS["XGB_PARAMS"]), ("LGBM", DEFAULTS["LGBM_PARAMS"])]:
            for pk, pv in params.items():
                vtype, vstr = _encode_model_param(pv)
                cur.execute(
                    """INSERT INTO model_params(model_name, param_key, param_value, value_type, updated_at)
                       VALUES(%s, %s, %s, %s, %s)
                       ON CONFLICT(model_name, param_key) DO NOTHING""",
                    (model_name, pk, vstr, vtype, now),
                )

    # ── Lecture complète ──────────────────────────────────────────────────────
    def load(self) -> dict:
        """Retourne la configuration complète sous forme de dict Python."""
        cfg: dict = {}
        with self._connect() as con:
            cur = self._cursor(con)

            # Scalaires
            cur.execute("SELECT key, value_real, value_bool, value_type FROM scalar_params")
            for row in cur.fetchall():
                if row["value_type"] == "real":
                    cfg[row["key"]] = row["value_real"]
                else:
                    cfg[row["key"]] = bool(row["value_bool"])

            # Multiplicateurs
            cur.execute("SELECT group_name, map_key, map_value FROM multipliers ORDER BY id")
            for row in cur.fetchall():
                cfg.setdefault(row["group_name"], {})[row["map_key"]] = row["map_value"]

            # Hyperparamètres
            cur.execute("SELECT model_name, param_key, param_value, value_type FROM model_params")
            for row in cur.fetchall():
                cfg_key = "XGB_PARAMS" if row["model_name"] == "XGB" else "LGBM_PARAMS"
                cfg.setdefault(cfg_key, {})[row["param_key"]] = _decode_model_param(
                    row["param_value"], row["value_type"]
                )

        return cfg

    # ── Sauvegarde (merge partiel) ────────────────────────────────────────────
    def save(self, patch: dict, changed_by: str = "api") -> dict:
        """
        Met à jour uniquement les clés présentes dans `patch`.
        Retourne la configuration complète après mise à jour.
        """
        now = _now()
        with self._connect() as con:
            cur = self._cursor(con)
            for key, value in patch.items():

                # ── Scalaires réels ──────────────────────────────────────────
                if key in _SCALAR_REAL_KEYS:
                    cur.execute("SELECT value_real FROM scalar_params WHERE key=%s", (key,))
                    old = cur.fetchone()
                    old_val = str(old["value_real"]) if old else None
                    cur.execute(
                        "UPDATE scalar_params SET value_real=%s, updated_at=%s WHERE key=%s",
                        (float(value), now, key),
                    )
                    _audit(cur, "scalar_params", key, old_val, str(value), changed_by)

                # ── Scalaires booléens ───────────────────────────────────────
                elif key in _SCALAR_BOOL_KEYS:
                    cur.execute("SELECT value_bool FROM scalar_params WHERE key=%s", (key,))
                    old = cur.fetchone()
                    old_val = str(bool(old["value_bool"])) if old else None
                    cur.execute(
                        "UPDATE scalar_params SET value_bool=%s, updated_at=%s WHERE key=%s",
                        (bool(value), now, key),
                    )
                    _audit(cur, "scalar_params", key, old_val, str(value), changed_by)

                # ── Multiplicateurs (merge clé par clé) ──────────────────────
                elif key in _MULT_KEYS and isinstance(value, dict):
                    for mk, mv in value.items():
                        cur.execute(
                            "SELECT map_value FROM multipliers WHERE group_name=%s AND map_key=%s",
                            (key, str(mk)),
                        )
                        old = cur.fetchone()
                        old_val = str(old["map_value"]) if old else None
                        cur.execute(
                            """INSERT INTO multipliers(group_name, map_key, map_value, updated_at)
                               VALUES(%s, %s, %s, %s)
                               ON CONFLICT(group_name, map_key)
                               DO UPDATE SET map_value=EXCLUDED.map_value,
                                             updated_at=EXCLUDED.updated_at""",
                            (key, str(mk), float(mv), now),
                        )
                        _audit(cur, "multipliers", f"{key}.{mk}", old_val, str(mv), changed_by)

                # ── Hyperparamètres ──────────────────────────────────────────
                elif key in _MODEL_KEYS and isinstance(value, dict):
                    model_name = "XGB" if key == "XGB_PARAMS" else "LGBM"
                    for pk, pv in value.items():
                        cur.execute(
                            "SELECT param_value FROM model_params WHERE model_name=%s AND param_key=%s",
                            (model_name, pk),
                        )
                        old = cur.fetchone()
                        old_val = old["param_value"] if old else None
                        vtype, vstr = _encode_model_param(pv)
                        cur.execute(
                            """INSERT INTO model_params(model_name, param_key, param_value, value_type, updated_at)
                               VALUES(%s, %s, %s, %s, %s)
                               ON CONFLICT(model_name, param_key)
                               DO UPDATE SET param_value=EXCLUDED.param_value,
                                             value_type=EXCLUDED.value_type,
                                             updated_at=EXCLUDED.updated_at""",
                            (model_name, pk, vstr, vtype, now),
                        )
                        _audit(cur, "model_params", f"{model_name}.{pk}", old_val, vstr, changed_by)

        return self.load()

    # ── Reset vers les défauts ────────────────────────────────────────────────
    def reset(self) -> dict:
        """Remet toutes les valeurs par défaut."""
        with self._connect() as con:
            cur = self._cursor(con)
            cur.execute("DELETE FROM scalar_params")
            cur.execute("DELETE FROM multipliers")
            cur.execute("DELETE FROM model_params")
            self._seed(con)
            _audit(cur, "all", "reset", None, "defaults", "reset")
        return self.load()

    # ── Audit log ─────────────────────────────────────────────────────────────
    def get_audit_log(self, limit: int = 100) -> list[dict]:
        with self._connect() as con:
            cur = self._cursor(con)
            cur.execute(
                "SELECT * FROM config_audit ORDER BY changed_at DESC LIMIT %s", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]


# ─── Helpers privés ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _encode_model_param(value) -> tuple[str, str]:
    if isinstance(value, bool):
        return "int", str(int(value))
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "float", str(value)
    return "str", str(value)


def _decode_model_param(value_str: str, value_type: str):
    if value_type == "int":
        return int(value_str)
    if value_type == "float":
        return float(value_str)
    return value_str


def _audit(cur, table: str, key: str, old, new, by: str):
    cur.execute(
        """INSERT INTO config_audit(table_name, record_key, old_value, new_value, changed_by)
           VALUES(%s, %s, %s, %s, %s)""",
        (table, key, old, new, by),
    )


# ── Schéma PostgreSQL ─────────────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scalar_params (
    key        TEXT    PRIMARY KEY,
    value_real DOUBLE PRECISION,
    value_bool BOOLEAN,
    value_type TEXT    NOT NULL CHECK(value_type IN ('real', 'bool')),
    updated_at TEXT    NOT NULL DEFAULT (now()::text)
);

CREATE TABLE IF NOT EXISTS multipliers (
    id         SERIAL  PRIMARY KEY,
    group_name TEXT    NOT NULL,
    map_key    TEXT    NOT NULL,
    map_value  DOUBLE PRECISION NOT NULL,
    updated_at TEXT    NOT NULL DEFAULT (now()::text),
    UNIQUE(group_name, map_key)
);

CREATE TABLE IF NOT EXISTS model_params (
    id          SERIAL  PRIMARY KEY,
    model_name  TEXT    NOT NULL,
    param_key   TEXT    NOT NULL,
    param_value TEXT    NOT NULL,
    value_type  TEXT    NOT NULL CHECK(value_type IN ('int', 'float', 'str')),
    updated_at  TEXT    NOT NULL DEFAULT (now()::text),
    UNIQUE(model_name, param_key)
);

CREATE TABLE IF NOT EXISTS config_audit (
    id          SERIAL  PRIMARY KEY,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    changed_by  TEXT        NOT NULL DEFAULT 'api',
    table_name  TEXT        NOT NULL,
    record_key  TEXT        NOT NULL,
    old_value   TEXT,
    new_value   TEXT
);

CREATE INDEX IF NOT EXISTS idx_mult_group  ON multipliers(group_name);
CREATE INDEX IF NOT EXISTS idx_model_name  ON model_params(model_name);
CREATE INDEX IF NOT EXISTS idx_audit_date  ON config_audit(changed_at);
"""