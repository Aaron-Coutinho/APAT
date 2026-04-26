from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import time

from backend.vector_store import PatentVectorStore
from backend.logic_whitespace import calculate_white_space_opportunities
from backend.topic_modeling import PatentTopicModeler
from backend.logic_market_intelligence import get_rd_signals, get_applicant_rd_breakdown
from backend.logic_tech_trends import get_filing_trends, extract_problem_statements
from backend.logic_future_trends import (
    forecast_filing_trends,
    classify_trajectory,
    generate_rd_recommendations,
)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.process_data import ensure_clean_data_exists, process_raw_csv

app = FastAPI(title="Patent Intelligence API")

# Ensure dataset is generated if it doesn't exist
ensure_clean_data_exists()

vector_store = PatentVectorStore(csv_path="data/patents_clean.csv")

# ─── Endpoint-level TTL Cache (1 hour) ────────────────────────────────────────
_TTL = 3600

_ws_cache          = {"data": None, "ts": 0.0}
_topic_cache       = {"data": None, "ts": 0.0}
_rd_cache          = {"data": None, "ts": 0.0}
_applicant_cache   = {"data": None, "ts": 0.0}
_filing_cache      = {"data": None, "ts": 0.0}
_problems_cache    = {"data": None, "ts": 0.0}
_forecast_cache    = {"data": None, "ts": 0.0}
_trajectory_cache  = {"data": None, "ts": 0.0}
_rec_cache         = {"data": None, "ts": 0.0}
_dashboard_cache   = {"data": None, "ts": 0.0}
# ──────────────────────────────────────────────────────────────────────────────


def _is_fresh(cache: dict) -> bool:
    return cache["data"] is not None and (time.time() - cache["ts"]) < _TTL


def _set(cache: dict, data):
    cache["data"] = data
    cache["ts"]   = time.time()
    return data


# ─── Request models ────────────────────────────────────────────────────────────

class IdeaRequest(BaseModel):
    idea: str


# ─── Core search endpoints ────────────────────────────────────────────────────

@app.post("/api/validate_idea")
async def validate_idea(request: IdeaRequest):
    return vector_store.search_idea(request.idea, top_k=5)

@app.post("/api/search_patents")
async def search_patents(request: IdeaRequest):
    return vector_store.search_idea(request.idea, top_k=20)["top_matches"]


# ─── System Admin ─────────────────────────────────────────────────────────────

@app.post("/api/reload_system")
async def reload_system():
    global vector_store
    
    # 1. Force reprocess the raw data
    print("[/api/reload_system] Forcibly running data pipeline...")
    success = process_raw_csv()
    if not success:
        return {"error": "Failed to process data/patents_raw.csv"}

    # 2. Delete stale FAISS index so vector store rebuilds from fresh CSV
    import shutil
    faiss_store_path = "data/faiss_store"
    if os.path.exists(faiss_store_path):
        shutil.rmtree(faiss_store_path)
        print(f"[/api/reload_system] Deleted stale FAISS store at '{faiss_store_path}'")

    # 3. Reload the vector store memory (will rebuild from fresh CSV)
    print("[/api/reload_system] Rebuilding vector store from fresh data...")
    vector_store = PatentVectorStore(csv_path="data/patents_clean.csv")

    # 4. Clear all backend caches
    for cache in [_ws_cache, _topic_cache, _rd_cache, _applicant_cache, 
                  _filing_cache, _problems_cache, _forecast_cache, 
                  _trajectory_cache, _rec_cache, _dashboard_cache]:
        cache["data"] = None
        cache["ts"] = 0.0

    return {"status": "success", "message": "System caches cleared and dataset reloaded."}



# ─── White-Space ──────────────────────────────────────────────────────────────

@app.get("/api/white_space")
async def get_white_space():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_ws_cache):
        print("[/api/white_space] Cache HIT")
        return _ws_cache["data"]

    print("[/api/white_space] Cache MISS — computing...")
    ws_df  = calculate_white_space_opportunities(vector_store.df)
    result = ws_df.to_dict(orient="records")
    return _set(_ws_cache, result)


# ─── Topic Clusters ───────────────────────────────────────────────────────────

@app.get("/api/topic_clusters")
async def get_topic_clusters():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_topic_cache):
        print("[/api/topic_clusters] Cache HIT")
        return _topic_cache["data"]

    print("[/api/topic_clusters] Cache MISS — loading or running BERTopic...")
    try:
        modeler = PatentTopicModeler()
        result  = modeler.get_topics_for_api(vector_store.df.copy())
        if "error" in result:
            return result
        return _set(_topic_cache, result)
    except Exception as e:
        return {"error": str(e)}


# ─── Market Intelligence ──────────────────────────────────────────────────────

@app.get("/api/market_intelligence/rd_signals")
async def get_market_rd_signals():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_rd_cache):
        print("[/api/market_intelligence/rd_signals] Cache HIT")
        return _rd_cache["data"]

    print("[/api/market_intelligence/rd_signals] Cache MISS — fetching...")
    df     = get_rd_signals(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_rd_cache, result)


@app.get("/api/market_intelligence/applicants")
async def get_market_applicants():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_applicant_cache):
        print("[/api/market_intelligence/applicants] Cache HIT")
        return _applicant_cache["data"]

    print("[/api/market_intelligence/applicants] Cache MISS — computing...")
    df     = get_applicant_rd_breakdown(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_applicant_cache, result)


# ─── Tech Trends ──────────────────────────────────────────────────────────────

@app.get("/api/tech_trends/filing_trends")
async def get_tech_filing_trends():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_filing_cache):
        print("[/api/tech_trends/filing_trends] Cache HIT")
        return _filing_cache["data"]

    print("[/api/tech_trends/filing_trends] Cache MISS — computing...")
    df     = get_filing_trends(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_filing_cache, result)


@app.get("/api/tech_trends/problems")
async def get_tech_problems():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_problems_cache):
        print("[/api/tech_trends/problems] Cache HIT")
        return _problems_cache["data"]

    print("[/api/tech_trends/problems] Cache MISS — scanning abstracts...")
    df     = extract_problem_statements(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_problems_cache, result)


# ─── Future Trends ────────────────────────────────────────────────────────────

@app.get("/api/future_trends/forecast")
async def get_future_forecast():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_forecast_cache):
        print("[/api/future_trends/forecast] Cache HIT")
        return _forecast_cache["data"]

    print("[/api/future_trends/forecast] Cache MISS — computing...")
    df     = forecast_filing_trends(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_forecast_cache, result)


@app.get("/api/future_trends/trajectory")
async def get_future_trajectory():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_trajectory_cache):
        print("[/api/future_trends/trajectory] Cache HIT")
        return _trajectory_cache["data"]

    print("[/api/future_trends/trajectory] Cache MISS — computing...")
    df     = classify_trajectory(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_trajectory_cache, result)


@app.get("/api/future_trends/recommendations")
async def get_future_recommendations():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_rec_cache):
        print("[/api/future_trends/recommendations] Cache HIT")
        return _rec_cache["data"]

    print("[/api/future_trends/recommendations] Cache MISS — generating...")
    df     = generate_rd_recommendations(vector_store.df)
    result = df.to_dict(orient="records")
    return _set(_rec_cache, result)


# ─── Policy Dashboard ─────────────────────────────────────────────────────────
# Reads exclusively from the shared in-memory cache objects populated above.
# It never re-runs any heavy computation itself.

@app.get("/api/policy_dashboard")
async def get_policy_dashboard():
    if vector_store.df is None:
        return {"error": "Dataset not loaded."}
    if _is_fresh(_dashboard_cache):
        print("[/api/policy_dashboard] Cache HIT")
        return _dashboard_cache["data"]

    print("[/api/policy_dashboard] Building dashboard from shared cache...")

    # ── Trajectory (reuse cache or compute once) ────────────────────
    if not _is_fresh(_trajectory_cache):
        df_traj = classify_trajectory(vector_store.df)
        _set(_trajectory_cache, df_traj.to_dict(orient="records"))
    traj_records = _trajectory_cache["data"]
    df_traj = pd.DataFrame(traj_records)

    # ── White-Space (reuse cache or compute once) ───────────────────
    if not _is_fresh(_ws_cache):
        ws_df = calculate_white_space_opportunities(vector_store.df)
        _set(_ws_cache, ws_df.to_dict(orient="records"))
    ws_records = _ws_cache["data"]
    df_ws = pd.DataFrame(ws_records)

    # ── Recommendations (reuse cache or compute once) ───────────────
    if not _is_fresh(_rec_cache):
        df_rec = generate_rd_recommendations(vector_store.df)
        _set(_rec_cache, df_rec.to_dict(orient="records"))
    rec_records = _rec_cache["data"]
    df_rec = pd.DataFrame(rec_records)

    # ── Aggregate KPIs ──────────────────────────────────────────────
    fastest_row = df_traj.loc[df_traj['slope'].idxmax()] if not df_traj.empty else None
    high_count  = int((df_traj['trajectory'] == "🚀 High Growth").sum())
    dec_count   = int((df_traj['trajectory'] == "📉 Declining").sum())
    top_ws      = df_ws.iloc[0] if not df_ws.empty else None
    top_recs    = df_rec[df_rec['priority'] == "HIGH"].to_dict(orient="records") if not df_rec.empty else []

    result = {
        "fastest_growing_field":      fastest_row['ipc_cpc'] if fastest_row is not None else "N/A",
        "fastest_slope":              round(float(fastest_row['slope']), 2) if fastest_row is not None else 0,
        "high_growth_count":          high_count,
        "declining_count":            dec_count,
        "top_whitespace_opportunity": top_ws['tech_keyword'] if top_ws is not None else "N/A",
        "top_whitespace_score":       round(float(top_ws['white_space_score']), 2) if top_ws is not None else 0,
        "priority_matrix":            traj_records,
        "top_recommendations":        top_recs,
    }

    return _set(_dashboard_cache, result)