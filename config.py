from __future__ import annotations

BASE_FARE     = 6
RATE_PER_KM   = 0.65
RATE_PER_MIN  = 0.3
MIN_FARE      = 4

W_XGB  = 0.55
W_LGBM = 0.45

MULT_TRAFFIC: dict[int, float] = {1: 1, 2: 1.2, 3: 1.5}

MULT_WEATHER: dict[int, float] = {1: 1.2, 2: 2.1, 3: 1.3, 4: 1.1}

MULT_DEMAND: dict[str, float] = {"normal": 1, "rush": 1.25, "surge": 1.6}

MULT_NIGHT: float = 2.2

MULT_CAR: dict[str, float] = {"economy": 0.75, "standard": 0.9, "comfort": 1, "first_class": 1.6, "van": 1.3, "mini_bus": 1.5}

MULT_FRIDAY_JUMUAH: float = 1.4

MULT_RAMADAN: dict[str, float] = {"ramadan_iftar": 2.1, "ramadan_tarawih": 1.3, "ramadan_suhoor": 1.15, "ramadan_last_week": 1.6, "none": 1}

MULT_BEACH: dict[str, float] = {"afflux_matin": 1.25, "après_midi": 1.3, "coucher_soleil": 1.35, "none": 1}

MULT_ZONE: dict[str, float] = {"capitale": 1.15, "banlieue": 1.05, "balnéaire": 1.1, "intérieure": 1, "sud": 0.95}

MULT_SPECIAL_EVENT: dict[str, float] = {"aid_el_fitr": 2, "aid_el_adha_week": 1.8, "new_year_eve": 1.9, "new_year_days": 1.4, "none": 1}

XGB_PARAMS: dict = {"objective": "reg:squarederror", "n_estimators": 800, "max_depth": 7, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1, "random_state": 42, "n_jobs": -1, "verbosity": 0}

LGBM_PARAMS: dict = {"objective": "regression", "n_estimators": 800, "max_depth": 7, "learning_rate": 0.04, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 10, "reg_alpha": 0.1, "reg_lambda": 1, "random_state": 42, "n_jobs": -1, "verbose": -1}

# ── Categorical encoding maps (used by models/features.py) ───────────────────
ZONE_MAP: dict[str, int] = {
    "capitale":    0,
    "banlieue":    1,
    "balnéaire":   2,
    "balneaire":   2,
    "intérieure":  3,
    "interieure":  3,
    "intérieur":   3,
    "interieur":   3,
    "sud":         4,
}

DEMAND_MAP: dict[str, int] = {
    "normal": 0,
    "rush":   1,
    "surge":  2,
}

PERIODE_MAP: dict[str, int] = {
    "circulation_normale": 0,
    "heure_de_pointe":     1,
    "nuit":                2,
    "week_end":            3,
    "weekend":             3,
    "ramadan":             4,
    "vacances":            5,
}

BEACH_REASON_MAP: dict[str, int] = {
    "afflux_matin":  0,
    "après_midi":    1,
    "apres_midi":    1,
    "coucher_soleil":2,
    "none":          3,
}

CAR_MAP: dict[str, int] = {
    "economy":     0,
    "standard":    1,
    "comfort":     2,
    "first_class": 3,
    "van":         4,
    "mini_bus":    5,
    "minibus":     5,
}

WEATHER_LABELS: dict[int, str] = {
    1: "clair",
    2: "pluie",
    3: "tempête",
    4: "sirocco",
}

ESTIMATED_WEATHER_BY_SEASON: dict[str, dict] = {
    "été": {
        "temperature_2m": 30.0,
        "precipitation":    0.0,
        "rain":             0.0,
        "windspeed_10m":    12.0,
        "weathercode_raw":  0,
        "visibility":   10_000.0,
        "weather_code":     1,
        "weather_label": "clair",
        "weather_mult":    1.00,
    },
    "printemps": {
        "temperature_2m": 22.0,
        "precipitation":    5.0,
        "rain":             5.0,
        "windspeed_10m":    15.0,
        "weathercode_raw":  61,
        "visibility":    8_000.0,
        "weather_code":     2,
        "weather_label": "pluie",
        "weather_mult":    1.10,
    },
    "automne": {
        "temperature_2m": 20.0,
        "precipitation":    10.0,
        "rain":             10.0,
        "windspeed_10m":    18.0,
        "weathercode_raw":  63,
        "visibility":    6_000.0,
        "weather_code":     2,
        "weather_label": "pluie",
        "weather_mult":    1.10,
    },
    "hiver": {
        "temperature_2m": 14.0,
        "precipitation":    20.0,
        "rain":             15.0,
        "windspeed_10m":    22.0,
        "weathercode_raw":  95,
        "visibility":    4_000.0,
        "weather_code":     3,
        "weather_label": "tempête",
        "weather_mult":    1.30,
    },
}

# ── Religious calendar tables (year → (start_date, end_date)) ──────────────────
RAMADAN_TABLE: dict[int, tuple[str, str]] = {
    2025: ("2025-03-01", "2025-03-30"),
    2026: ("2026-02-18", "2026-03-19"),
    2027: ("2027-02-08", "2027-03-09"),
}

AID_ADHA_TABLE: dict[int, tuple[str, str]] = {
    2025: ("2025-06-06", "2025-06-13"),
    2026: ("2026-05-27", "2026-06-03"),
    2027: ("2027-05-17", "2027-05-24"),
}

# ── Feature toggles (admin ON/OFF controls) ──────────────────────────────────
ENABLE_TRAFFIC:       bool = True
ENABLE_WEATHER:       bool = True
ENABLE_DEMAND:        bool = True
ENABLE_NIGHT:         bool = True
ENABLE_FRIDAY_JUMUAH: bool = True
ENABLE_RAMADAN:       bool = True
ENABLE_BEACH:         bool = True
ENABLE_ZONE:          bool = True
ENABLE_SPECIAL_EVENT: bool = True
ENABLE_SEASON:        bool = True

TARGET_COL   = "surge_multiplier"
RANDOM_STATE = 42
