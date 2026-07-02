# backend.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from functools import lru_cache
from pybaseball import statcast_pitcher
import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import List, Dict

app = FastAPI(title="Pitch3D API")

# CORS: allow your Vite origin(s) while developing. "*" is fine for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONE (and only one) generic exception handler
@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # Let FastAPI handle HTTPException normally (keeps status code)
    if isinstance(exc, HTTPException):
        raise exc
    # Everything else -> 500 JSON. CORS headers are added by CORSMiddleware.
    return JSONResponse(status_code=500, content={"detail": "Internal error", "error": str(exc)})

@app.get("/health")
def health():
    return {"ok": True}


# Columns we need
PITCH_COLS_ODE = [
    "release_pos_x","release_pos_y","release_pos_z",
    "vx0","vy0","vz0",
    "release_spin_rate","spin_axis",
    "ax","ay","az",
    "pitch_type","release_speed","plate_x","plate_z","sz_top","sz_bot",
]
PITCH_COLS_NICE = ["release_extension","effective_speed","pfx_x","pfx_z","game_date"]
PITCH_COLS = PITCH_COLS_ODE + PITCH_COLS_NICE

def _sanitize_value(v):
    # pandas NA check first (handles NaT too)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # unwrap numpy scalars
    if isinstance(v, (np.generic,)):
        v = v.item()

    # floats: kill NaN/Inf
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # ints
    if isinstance(v, (int, np.integer)):
        return int(v)

    # datetimes -> string
    if isinstance(v, (datetime, pd.Timestamp)):
        # convert tz-aware to naive ISO if needed
        try:
            if getattr(v, "tzinfo", None) is not None:
                v = v.tz_convert(None)
        except Exception:
            pass
        return v.strftime("%Y-%m-%d %H:%M:%S")

    # everything else as-is (str, bool, etc.)
    return v

def _df_jsonable(df: pd.DataFrame) -> list[dict]:
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    # stringify datetime columns first (keeps them human friendly)
    for c in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        df[c] = df[c].astype("datetime64[ns]")
    records = df.to_dict(orient="records")
    return [{k: _sanitize_value(v) for k, v in rec.items()} for rec in records]

def _jsonable_dict(d: dict) -> dict:
    return {k: _sanitize_value(v) for k, v in d.items()}



@lru_cache(maxsize=64)
def _fetch_pitcher_df(mlbam_id: int, start: str, end: str) -> pd.DataFrame:
    try:
        df = statcast_pitcher(start, end, mlbam_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Statcast upstream error: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No Statcast rows found for that range/id")

    need = [c for c in PITCH_COLS if c in df.columns]
    if not need:
        raise HTTPException(status_code=502, detail="Statcast returned unexpected columns")

    try:
        df = df[need].dropna(subset=[
            "release_pos_x","release_pos_y","release_pos_z",
            "vx0","vy0","vz0","ay"
        ])
    except KeyError as e:
        raise HTTPException(status_code=502, detail=f"Missing expected columns: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="No complete rows with kinematic fields")
    return df.copy()

@app.get("/pitcher/{mlbam_id}/rows")
def pitcher_rows(mlbam_id: int, start: str, end: str):
    df = _fetch_pitcher_df(mlbam_id, start, end)
    return _df_jsonable(df)

@app.get("/pitcher/{mlbam_id}/averages")
def pitcher_averages(mlbam_id: int, start: str, end: str):
    df = _fetch_pitcher_df(mlbam_id, start, end)
    usage = df["pitch_type"].value_counts(normalize=True) * 100.0
    agg = df.groupby("pitch_type").mean(numeric_only=True).reset_index()
    agg["usage_pct"] = agg["pitch_type"].map(usage).fillna(0.0)
    return _df_jsonable(agg)

@app.get("/pitcher/{mlbam_id}/nearest")
def pitcher_nearest(
    mlbam_id: int,
    pitch_type: str,
    x: float = Query(..., description="target plate_x in feet"),
    z: float = Query(..., description="target plate_z in feet"),
    k: int   = Query(100, ge=1, le=1000),
    radius_in: float = 6.0,
    expand_step_in: float = 3.0,
    max_radius_in: float = 24.0,
    start: str = "2024-03-01",
    end: str   = "2024-11-30",
    weighted: bool = True,
    return_points: bool = True,   # <— NEW
):
    df = _fetch_pitcher_df(mlbam_id, start, end)
    sub = df[df["pitch_type"] == pitch_type].copy()
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No rows for pitch_type={pitch_type}")

    dx = sub["plate_x"].to_numpy() - x
    dz = sub["plate_z"].to_numpy() - z
    dist = np.hypot(dx, dz)
    sub["dist_ft"] = dist

    r_ft = radius_in / 12.0
    r_max_ft = max_radius_in / 12.0
    step_ft = expand_step_in / 12.0

    picked = sub[sub["dist_ft"] <= r_ft]
    while len(picked) < k and r_ft < r_max_ft:
        r_ft = min(r_ft + step_ft, r_max_ft)
        picked = sub[sub["dist_ft"] <= r_ft]

    if picked.empty:
        raise HTTPException(status_code=404, detail="No pitches within radius")

    picked = picked.sort_values("dist_ft").head(k)

    if weighted:
        bw = max(1e-6, r_ft * 0.5)
        w = np.exp(-(picked["dist_ft"].to_numpy()**2) / (2*bw*bw))
        def wavg(col):
            arr = picked[col].to_numpy()
            return float(np.average(arr, weights=w))
        agg_vals = {c: wavg(c) for c in PITCH_COLS_ODE if c in picked.columns and pd.api.types.is_numeric_dtype(picked[c])}
    else:
        agg_vals = picked[PITCH_COLS_ODE].mean(numeric_only=True).to_dict()

    agg_vals["pitch_type"] = pitch_type
    agg_vals["target_plate_x"] = x
    agg_vals["target_plate_z"] = z
    agg_vals["neighbors_used"] = int(len(picked))
    agg_vals["mean_dist_in"]   = float(picked["dist_ft"].mean() * 12.0)
    agg_vals["max_dist_in"]    = float(picked["dist_ft"].max()  * 12.0)
    agg_vals["search_radius_in"] = float(r_ft * 12.0)

    # NEW: return neighbor terminal positions + weights for the cloud
    if "plate_x" in picked.columns and "plate_z" in picked.columns:
        pts = []
        # use same weights if 'weighted' is True, else uniform weights = 1.0
        if weighted:
            weights = w
        else:
            weights = np.ones(len(picked), dtype=float)
        for px, pz, ww in zip(picked["plate_x"].to_numpy(),
                              picked["plate_z"].to_numpy(),
                              weights):
            # coerce to plain floats for JSON-safety
            pts.append({"x": float(px), "z": float(pz), "w": float(ww)})
        agg_vals["points"] = pts
    else:
        agg_vals["points"] = []

    return _jsonable_dict(agg_vals)

