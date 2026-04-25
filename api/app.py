"""
╔══════════════════════════════════════════════════════════════════╗
║   MOVIROO — api/app.py                                           ║
║   FastAPI REST API pour l'application mobile                     ║
║                                                                  ║
║   Endpoints :                                                    ║
║     GET  /health              → statut de l'API + ML             ║
║     POST /price/estimate      → estimation prix (coords + contexte) ║
║     POST /price/quick         → prix rapide (coords seules)      ║
║     GET  /vehicles            → liste des types de véhicules     ║
║     GET  /zones               → multiplicateurs par zone         ║
║                                                                  ║
║   Lancer :                                                       ║
║     uvicorn api.app:app --reload --port 8000                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from typing   import Optional

# ── Force UTF-8 stdout/stderr (Windows console fix for emoji prints) ─────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fastapi              import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic             import BaseModel, Field, field_validator

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config           import MULT_CAR, MULT_ZONE, MULT_SPECIAL_EVENT
from models.predictor import predictor
from pricing.engine   import (
    calculate_trip_price,
    calculate_trip_prices_batch,
    CarType,
)


# ══════════════════════════════════════════════════════════════════
# DATETIME NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def _normalize_datetime(dt: datetime | None) -> datetime:
    """
    Normalise tous les datetime en timezone-naive UTC.

    - Si dt est None → datetime.now() timezone-naive
    - Si dt est timezone-aware → convertit en UTC puis supprime tzinfo
    - Si dt est timezone-naive → garde tel quel

    Cela évite les erreurs "Cannot subtract tz-naive and tz-aware"
    dans weather.py et garantit une cohérence dans tout le pipeline.
    """
    if dt is None:
        return datetime.now()

    # Si timezone-aware, convertir en UTC puis supprimer tzinfo
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def _sanitize_json_response(data: dict) -> dict:
    """
    Remplace les valeurs NaN/inf par None pour éviter les erreurs JSON.
    Utilisé pour les réponses ML qui peuvent contenir NaN (ex: ml_surge_xgb).
    """
    import math

    def clean(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(item) for item in obj]
        return obj

    return clean(data)


def _validate_trip_coordinates(
    lat_origin: float,
    lon_origin: float,
    lat_dest:   float,
    lon_dest:   float,
) -> None:
    """
    Validate trip coordinates before any OSRM / pricing call.

    Raises HTTPException(400) on:
      - (0, 0) coordinates (typical fallback for missing data)
      - Identical pickup and dropoff (zero-distance trip)
    Out-of-range values are caught earlier by Pydantic field_validator.
    """
    if lat_origin == 0 and lon_origin == 0:
        raise HTTPException(
            status_code=400,
            detail="Pickup coordinates (0, 0) are invalid — likely a missing/fallback value.",
        )
    if lat_dest == 0 and lon_dest == 0:
        raise HTTPException(
            status_code=400,
            detail="Dropoff coordinates (0, 0) are invalid — likely a missing/fallback value.",
        )
    if lat_origin == lat_dest and lon_origin == lon_dest:
        raise HTTPException(
            status_code=400,
            detail="Pickup and dropoff coordinates cannot be identical.",
        )


# ══════════════════════════════════════════════════════════════════
# APP FASTAPI
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Moviroo Pricing API",
    description = "API de tarification dynamique pour l'application mobile Moviroo",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — autoriser l'app mobile (React Native, Flutter, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════
# STARTUP — Chargement des modèles ML
# ══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    print("🚀 Moviroo API démarrage...")
    predictor.load()
    print("✅ API prête")


# ══════════════════════════════════════════════════════════════════
# SCHEMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════

class PriceEstimateRequest(BaseModel):
    """Corps de la requête POST /price/estimate"""

    # Coordonnées (obligatoires)
    lat_origin: float = Field(..., description="Latitude départ",      example=36.8625)
    lon_origin: float = Field(..., description="Longitude départ",     example=10.1956)
    lat_dest:   float = Field(..., description="Latitude destination", example=35.8256)
    lon_dest:   float = Field(..., description="Longitude destination",example=10.6370)

    # Véhicule
    car_type: str = Field(
        default="comfort",
        description="Type de véhicule : economy | standard | comfort | first_class | van | mini_bus",
        example="comfort",
    )

    # Date/heure de réservation (ISO 8601 — None = maintenant)
    booking_dt: Optional[str] = Field(
        default=None,
        description="Date/heure ISO 8601 (ex: 2026-07-31T13:00:00). Défaut = maintenant.",
        example="2026-07-31T13:00:00",
    )

    # Contexte trafic (optionnel — le geo_lookup complète automatiquement)
    trafic_niveau:     int   = Field(default=1,    ge=1, le=3,  description="Niveau de trafic 1-3")
    demande:           str   = Field(default="normal",          description="Demande : normal | rush | surge")
    indice_congestion: int   = Field(default=30,   ge=0, le=100,description="Indice de congestion 0-100")
    retard_estime_min: int   = Field(default=5,    ge=0,        description="Retard estimé en minutes")
    vitesse_moy_kmh:   float = Field(default=40.0, gt=0,        description="Vitesse moyenne km/h")
    chauffeurs_actifs: int   = Field(default=30,   ge=1,        description="Nombre de chauffeurs actifs")

    # Contexte géographique (optionnel — rempli par geo_lookup si absent)
    zone_type:       str = Field(default="intérieure", description="Zone : capitale | banlieue | balnéaire | intérieure | sud")
    has_beach:       int = Field(default=0, ge=0, le=1,description="Zone balnéaire : 0 ou 1")
    population:      int = Field(default=300_000,      description="Population de la ville")
    intensite_ville: int = Field(default=3, ge=1, le=5,description="Intensité urbaine 1-5")

    # ML
    use_ml: bool = Field(default=True, description="Utiliser le modèle ML (True) ou règles métier seules (False)")

    @field_validator("car_type")
    @classmethod
    def validate_car_type(cls, v: str) -> str:
        normalized = CarType.normalize(v)
        return normalized

    @field_validator("demande")
    @classmethod
    def validate_demande(cls, v: str) -> str:
        allowed = ["normal", "rush", "surge"]
        if v.lower() not in allowed:
            raise ValueError(f"demande doit être parmi {allowed}")
        return v.lower()

    @field_validator("zone_type")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        allowed = ["capitale", "banlieue", "balnéaire", "intérieure", "sud"]
        if v.lower() not in allowed:
            raise ValueError(f"zone_type doit être parmi {allowed}")
        return v.lower()


class QuickPriceRequest(BaseModel):
    """Corps simplifié pour POST /price/quick — app mobile"""
    lat_origin: float = Field(..., example=36.8625)
    lon_origin: float = Field(..., example=10.1956)
    lat_dest:   float = Field(..., example=35.8256)
    lon_dest:   float = Field(..., example=10.6370)
    car_type:   str   = Field(default="comfort", example="comfort")
    booking_dt: Optional[str] = Field(default=None)

    @field_validator("lat_origin", "lat_dest")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if v is None or not isinstance(v, (int, float)):
            raise ValueError("Latitude must be a number")
        if v != v:  # NaN check
            raise ValueError("Latitude cannot be NaN")
        if v < -90 or v > 90:
            raise ValueError(f"Latitude {v} out of range (-90..90)")
        return float(v)

    @field_validator("lon_origin", "lon_dest")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if v is None or not isinstance(v, (int, float)):
            raise ValueError("Longitude must be a number")
        if v != v:  # NaN check
            raise ValueError("Longitude cannot be NaN")
        if v < -180 or v > 180:
            raise ValueError(f"Longitude {v} out of range (-180..180)")
        return float(v)


class BatchPriceRequest(BaseModel):
    """
    Corps pour POST /price/batch — calcul prix multi-véhicules.

    Permet de récupérer les prix de plusieurs classes de véhicules
    en un seul appel (optimisé : OSRM + météo + geo-lookup exécutés 1 fois).
    """
    lat_origin: float = Field(..., example=36.8002)
    lon_origin: float = Field(..., example=10.1858)
    lat_dest:   float = Field(..., example=36.4513)
    lon_dest:   float = Field(..., example=10.7356)
    car_types:  list[str] = Field(
        default=["economy", "standard", "comfort", "first_class", "van", "mini_bus"],
        description="Liste des types de véhicules à calculer",
        example=["economy", "comfort", "first_class", "van"],
    )
    booking_dt: Optional[str] = Field(default=None)

    @field_validator("lat_origin", "lat_dest")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if v is None or not isinstance(v, (int, float)):
            raise ValueError("Latitude must be a number")
        if v != v:  # NaN check
            raise ValueError("Latitude cannot be NaN")
        if v < -90 or v > 90:
            raise ValueError(f"Latitude {v} out of range (-90..90)")
        return float(v)

    @field_validator("lon_origin", "lon_dest")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if v is None or not isinstance(v, (int, float)):
            raise ValueError("Longitude must be a number")
        if v != v:  # NaN check
            raise ValueError("Longitude cannot be NaN")
        if v < -180 or v > 180:
            raise ValueError(f"Longitude {v} out of range (-180..180)")
        return float(v)

    @field_validator("car_types")
    @classmethod
    def validate_car_types(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("car_types ne peut pas être vide")
        return v


# ══════════════════════════════════════════════════════════════════
# SCHEMAS RÉPONSE
# ══════════════════════════════════════════════════════════════════

class MultipliersDetail(BaseModel):
    traffic:       float
    weather:       float
    demand:        float
    night:         float
    car:           float
    friday:        float
    ramadan:       float
    beach:         float
    zone:          float
    special_event: float

class MLDetail(BaseModel):
    used:       bool
    surge_xgb:  Optional[float]
    surge_lgbm: Optional[float]
    confidence: Optional[float]

class WeatherDetail(BaseModel):
    code:        int
    label:       str
    temperature: float
    wind_kmh:    float
    is_night:    int
    estimated:   bool

class GeoDetail(BaseModel):
    ville:          Optional[str]
    zone_type:      Optional[str]
    has_beach:      Optional[int]
    distance_km:    Optional[float]
    in_dataset:     bool

class PriceResponse(BaseModel):
    # Prix
    base_fare:        float
    distance_cost:    float
    duration_cost:    float
    raw_price:        float
    surge_multiplier: float
    final_price:      float
    final_price_exact: float
    loyalty_points:   int
    currency:         str
    min_applied:      bool

    # Trajet
    distance_km:  float
    duration_min: int
    car_type:     str
    car_type_label: str
    zone_type:    str
    booking_dt:   str

    # Détails
    multipliers: MultipliersDetail
    ml:          MLDetail
    weather:     WeatherDetail
    geo_origin:  Optional[GeoDetail]
    geo_dest:    Optional[GeoDetail]

    # Saison / période
    season:        str
    periode:       str
    special_event: str

    # Source
    source: str
    labels: dict


# ══════════════════════════════════════════════════════════════════
# HELPER — Convertir le dict interne → PriceResponse
# ══════════════════════════════════════════════════════════════════

def _to_response(r: dict) -> PriceResponse:
    wth = r.get("weather", {})
    tf  = r.get("time_flags", {})
    geo_o = r.get("geo_origin")
    geo_d = r.get("geo_dest")

    def _geo(g: dict | None) -> Optional[GeoDetail]:
        if not g:
            return None
        return GeoDetail(
            ville       = g.get("ville"),
            zone_type   = g.get("zone_type"),
            has_beach   = int(g.get("has_beach", 0)),
            distance_km = g.get("distance_km"),
            in_dataset  = g.get("in_dataset", False),
        )

    return PriceResponse(
        base_fare        = r["base_fare"],
        distance_cost    = r["distance_cost"],
        duration_cost    = r["duration_cost"],
        raw_price        = r["raw_price"],
        surge_multiplier = r["surge_multiplier"],
        final_price       = r["final_price_rounded"],
        final_price_exact = r["final_price"],
        loyalty_points   = r.get("loyalty_points", 0),
        currency         = r["currency"],
        min_applied      = r["min_applied"],
        distance_km      = r["distance_km"],
        duration_min     = math.ceil(r["duration_min"]),
        car_type         = r["car_type"],
        car_type_label   = r.get("car_type_label", r["car_type"]),
        zone_type        = r["zone_type"],
        booking_dt       = r["booking_dt"],
        multipliers = MultipliersDetail(
            traffic       = r.get("mult_traffic",       1.0),
            weather       = r.get("mult_weather",       1.0),
            demand        = r.get("mult_demand",        1.0),
            night         = r.get("mult_night",         1.0),
            car           = r.get("mult_car",           1.0),
            friday        = r.get("mult_friday",        1.0),
            ramadan       = r.get("mult_ramadan",       1.0),
            beach         = r.get("mult_beach",         1.0),
            zone          = r.get("mult_zone",          1.0),
            special_event = r.get("mult_special_event", 1.0),
        ),
        ml = MLDetail(
            used       = r.get("ml_used",       False),
            surge_xgb  = r.get("ml_surge_xgb"),
            surge_lgbm = r.get("ml_surge_lgbm"),
            confidence = r.get("ml_confidence"),
        ),
        weather = WeatherDetail(
            code        = wth.get("weather_code",  1),
            label       = wth.get("weather_label", "clair"),
            temperature = wth.get("temperature_2m", 22.0),
            wind_kmh    = wth.get("windspeed_10m",  10.0),
            is_night    = wth.get("is_night",        0),
            estimated   = wth.get("weather_estimated", False),
        ),
        geo_origin     = _geo(geo_o),
        geo_dest       = _geo(geo_d),
        season         = tf.get("season",        "été"),
        periode        = tf.get("periode",       "circulation_normale"),
        special_event  = tf.get("special_event", "none"),
        source         = r.get("source", ""),
        labels         = r.get("labels", {}),
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Système"])
def health():
    """Vérifie que l'API et le modèle ML sont opérationnels."""
    return {
        "status":    "ok",
        "ml_loaded": predictor.is_loaded,
        "version":   "2.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/vehicles", tags=["Référentiel"])
def get_vehicles():
    """
    Retourne la liste des types de véhicules disponibles
    avec leur multiplicateur de prix.
    """
    return {
        "vehicles": [
            {
                "id":           v,
                "label":        CarType.LABELS[v],
                "multiplier":   MULT_CAR[v],
            }
            for v in CarType.ALL
        ]
    }


@app.get("/zones", tags=["Référentiel"])
def get_zones():
    """Retourne les multiplicateurs par zone géographique."""
    return {
        "zones": [
            {"id": z, "multiplier": m}
            for z, m in MULT_ZONE.items()
        ]
    }


@app.post("/price/estimate", response_model=PriceResponse, tags=["Tarification"])
def price_estimate(req: PriceEstimateRequest):
    """
    Calcule le prix dynamique complet d'un trajet.

    - Appel OSRM pour distance/durée réelles
    - Météo Open-Meteo (ou estimée si date lointaine)
    - Geo-lookup dans le dataset (rayon 20 km)
    - Flags temporels : Ramadan, Aïd, Nouvel An, Vendredi…
    - ML ensemble XGB+LGBM (ou règles métier si ML désactivé)
    """
    try:
        raw_dt = datetime.fromisoformat(req.booking_dt) if req.booking_dt else None
        booking_dt = _normalize_datetime(raw_dt)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"booking_dt invalide : '{req.booking_dt}'. Format attendu : 2026-07-31T13:00:00"
        )

    try:
        result = calculate_trip_price(
            lat_origin        = req.lat_origin,
            lon_origin        = req.lon_origin,
            lat_dest          = req.lat_dest,
            lon_dest          = req.lon_dest,
            zone_type         = req.zone_type,
            has_beach         = req.has_beach,
            population        = req.population,
            intensite_ville   = req.intensite_ville,
            trafic_niveau     = req.trafic_niveau,
            demande           = req.demande,
            indice_congestion = req.indice_congestion,
            retard_estime_min = req.retard_estime_min,
            vitesse_moy_kmh   = req.vitesse_moy_kmh,
            chauffeurs_actifs = req.chauffeurs_actifs,
            car_type          = req.car_type,
            booking_dt        = booking_dt,
            use_ml            = req.use_ml,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _sanitize_json_response(_to_response(result))


@app.post("/price/quick", response_model=PriceResponse, tags=["Tarification"])
def price_quick(req: QuickPriceRequest):
    """
    Estimation rapide avec coordonnées seules.
    Tous les paramètres contextuels sont auto-détectés :
    - Zone et beach depuis le dataset (geo-lookup rayon 20 km)
    - Météo depuis Open-Meteo
    - Heure/date : maintenant (ou booking_dt si fourni)
    """
    # Validate coordinates BEFORE any OSRM/pricing call
    _validate_trip_coordinates(
        req.lat_origin, req.lon_origin, req.lat_dest, req.lon_dest
    )

    try:
        raw_dt = datetime.fromisoformat(req.booking_dt) if req.booking_dt else None
        booking_dt = _normalize_datetime(raw_dt)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"booking_dt invalide : '{req.booking_dt}'"
        )

    try:
        result = calculate_trip_price(
            lat_origin = req.lat_origin,
            lon_origin = req.lon_origin,
            lat_dest   = req.lat_dest,
            lon_dest   = req.lon_dest,
            car_type   = req.car_type,
            booking_dt = booking_dt,
            use_ml     = True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _sanitize_json_response(_to_response(result))


@app.post("/price/batch", tags=["Tarification"])
def price_batch(req: BatchPriceRequest):
    """
    Calcul prix BATCH pour plusieurs types de véhicules en un seul appel.

    Utilisé par l'app passagère pour afficher tous les prix (Economy, Comfort,
    First Class, Van, Mini Bus) en un seul round-trip.

    Optimisation : OSRM, météo, geo-lookup et flags sont exécutés UNE FOIS.
    Seul le calcul de surge + prix final est itéré par car_type.

    Retour :
    {
      "common": {
        "distance_km": 66.72,
        "duration_min": 62,
        "zone_type": "capitale",
        "booking_dt": "...",
        "weather": { ... },
        "time_flags": { ... },
        ...
      },
      "prices": [
        { "car_type": "economy", "final_price": 150.0, "surge_multiplier": 3.07, ... },
        { "car_type": "comfort", "final_price": 205.0, "surge_multiplier": 3.07, ... },
        ...
      ]
    }
    """
    # Validate coordinates BEFORE any OSRM/pricing call
    _validate_trip_coordinates(
        req.lat_origin, req.lon_origin, req.lat_dest, req.lon_dest
    )

    try:
        raw_dt = datetime.fromisoformat(req.booking_dt) if req.booking_dt else None
        booking_dt = _normalize_datetime(raw_dt)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"booking_dt invalide : '{req.booking_dt}'"
        )

    try:
        result = calculate_trip_prices_batch(
            lat_origin = req.lat_origin,
            lon_origin = req.lon_origin,
            lat_dest   = req.lat_dest,
            lon_dest   = req.lon_dest,
            car_types  = req.car_types,
            booking_dt = booking_dt,
            use_ml     = True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Normalise la réponse (math.ceil des durées)
    common = result["common"]
    common["duration_min"] = math.ceil(common["duration_min"])

    # Sanitize NaN/inf values for JSON serialization
    return _sanitize_json_response(result)