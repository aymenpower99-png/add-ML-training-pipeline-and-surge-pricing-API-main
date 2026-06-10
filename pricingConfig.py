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

# ── Feature toggles (admin ON/OFF controls) ──────────────────────────────────
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
