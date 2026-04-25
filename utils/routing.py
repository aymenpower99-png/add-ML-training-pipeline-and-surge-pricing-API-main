

# from __future__ import annotations

# import requests

# # ── URL de l'API OSRM publique (remplaçable par une instance privée) ──
# OSRM_BASE_URL = "http://router.project-osrm.org"


# def get_osrm_distance(
#     lat1: float,
#     lon1: float,
#     lat2: float,
#     lon2: float,
#     osrm_url: str = OSRM_BASE_URL,
#     timeout:  int = 10,
# ) -> tuple[float, float]:
   
#     endpoint = (
#         f"{osrm_url}/route/v1/driving/"
#         f"{lon1},{lat1};{lon2},{lat2}"
#         f"?overview=false"
#     )

#     try:
#         resp = requests.get(endpoint, timeout=timeout)
#         resp.raise_for_status()
#         data  = resp.json()
#         route = data["routes"][0]

#         distance_km  = round(route["distance"] / 1_000, 2)
#         duration_min = round(route["duration"] / 60,    2)

#         return distance_km, duration_min

#     except (KeyError, IndexError) as exc:
#         raise RuntimeError(
#             f"OSRM — réponse inattendue pour "
#             f"({lat1},{lon1}) → ({lat2},{lon2}) : {exc}"
#         ) from exc
#     except requests.RequestException as exc:
#         raise RuntimeError(f"OSRM — erreur réseau : {exc}") from exc
import requests

def get_osrm_distance(lat1, lon1, lat2, lon2):
    """
    Compute road distance/duration via OSRM.

    Validates coordinates BEFORE the HTTP call to avoid sending
    (0,0;0,0) which causes OSRM to return 400 Bad Request.

    Raises:
        ValueError: if any coordinate is invalid (None, NaN, 0/0, out-of-range)
                    or if OSRM cannot return a route.
    """
    # ── Validate coordinates ─────────────────────────────────────
    coords = {"lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2}
    for name, val in coords.items():
        if val is None:
            raise ValueError(f"OSRM: {name} is None")
        try:
            v = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"OSRM: {name}={val!r} is not numeric")
        if v != v:  # NaN check
            raise ValueError(f"OSRM: {name} is NaN")

    if lat1 == 0 and lon1 == 0:
        raise ValueError(
            f"OSRM: origin coordinates (0, 0) are invalid — likely a missing/fallback value"
        )
    if lat2 == 0 and lon2 == 0:
        raise ValueError(
            f"OSRM: destination coordinates (0, 0) are invalid — likely a missing/fallback value"
        )
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError(f"OSRM: latitude out of range ({lat1}, {lat2})")
    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError(f"OSRM: longitude out of range ({lon1}, {lon2})")
    if lat1 == lat2 and lon1 == lon2:
        raise ValueError(
            f"OSRM: origin and destination are identical ({lat1}, {lon1})"
        )

    # ── Call OSRM ────────────────────────────────────────────────
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=false"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        if not data.get("routes"):
            raise ValueError(
                f"OSRM: no route returned for ({lat1},{lon1}) -> ({lat2},{lon2})"
            )

        route = data["routes"][0]
        distance_km = round(route["distance"] / 1000, 2)
        duration_min = round(route["duration"] / 60, 2)

        return distance_km, duration_min

    except requests.RequestException as e:
        raise ValueError(f"OSRM network error: {e}") from e
    except (KeyError, IndexError) as e:
        raise ValueError(f"OSRM unexpected response: {e}") from e
# point1 = (36.848097, 10.217551)
# point2 = (36.420177, 10.553902)

# distance, duration = get_osrm_distance(*point1, *point2)

# if distance is not None:
#     print(f"Distance réelle: {distance:.2f} km")
#     print(f"Durée: {duration:.2f} min")
# else:
#     print("Impossible de calculer la distance ou la durée.")