"""
AgroMonitor UZ — FastAPI Backend
=================================
Запуск:
    pip install fastapi uvicorn python-multipart pillow sqlalchemy
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict          — предсказание болезни
    POST /feedback         — отзыв агронома (коррекция)
    GET  /observations     — все сохранённые наблюдения
    GET  /dataset/export   — экспорт реального датасета (CSV)
    GET  /stats            — статистика
"""

import os
import io
import uuid
import json
import csv
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "agromonitor.db"
UPLOAD_DIR  = BASE_DIR / "uploads"
MODEL_DIR   = BASE_DIR.parent.parent / "plant_disease_project"

UPLOAD_DIR.mkdir(exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base   = declarative_base()
Session = sessionmaker(bind=engine)


class Observation(Base):
    __tablename__ = "observations"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at       = Column(DateTime, default=datetime.utcnow)
    # Location context
    region           = Column(String)
    season           = Column(String)
    # Climate inputs
    avg_temperature_c = Column(Float)
    humidity_pct      = Column(Float)
    rainfall_mm       = Column(Float)
    soil_type         = Column(String)
    soil_ph           = Column(Float)
    # Image
    image_path        = Column(String)
    image_filename    = Column(String)
    # Prediction output
    predicted_class   = Column(String)
    predicted_label   = Column(Integer)   # disease_class_idx
    disease_probability = Column(Float)
    top3_json         = Column(Text)      # JSON list of top-3
    model_version     = Column(String, default="1.0")
    # Feedback from agronomist
    has_feedback      = Column(Boolean, default=False)
    true_class        = Column(String, nullable=True)
    feedback_notes    = Column(Text, nullable=True)
    feedback_at       = Column(DateTime, nullable=True)
    agronomist_name   = Column(String, nullable=True)


Base.metadata.create_all(engine)

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroMonitor UZ API",
    description="Plant disease prediction for Uzbekistan regions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loader ───────────────────────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        import sys
        sys.path.insert(0, str(MODEL_DIR))
        from utils.inference import DiseasePredictor
        _predictor = DiseasePredictor(device="cpu")
        print("[OK] Модель загружена")
    except Exception as e:
        print(f"[WARN] Модель не загружена: {e}. Используется mock.")
        _predictor = MockPredictor()
    return _predictor


class MockPredictor:
    """Заглушка если модель не обучена — для разработки UI."""
    DISEASES = [
        "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Bacterial_spot", "Tomato___Leaf_Mold",
        "Potato___Late_blight", "Potato___Early_blight",
        "Grape___Black_rot", "Apple___Apple_scab",
    ]

    def predict(self, image_path, temperature, humidity, rainfall,
                soil_ph, soil_type, season, region):
        rng = np.random.default_rng(abs(hash(image_path + region + season)) % (2**31))
        probs = rng.dirichlet(np.ones(8))
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.DISEASES[i], float(probs[i])) for i in top3_idx]
        return {
            "top3":       top3,
            "risk_score": float(probs[top3_idx[0]]),
            "raw_logits": probs.tolist(),
        }


# ── Helpers ────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato___Early_blight":   {"crop": "Tomato",   "uz": "Pomidor — erta yanib ketish",     "icon": "🍅"},
    "Tomato___Late_blight":    {"crop": "Tomato",   "uz": "Pomidor — kech yanib ketish",      "icon": "🍅"},
    "Tomato___Bacterial_spot": {"crop": "Tomato",   "uz": "Pomidor — bakterial dog'",         "icon": "🍅"},
    "Tomato___Leaf_Mold":      {"crop": "Tomato",   "uz": "Pomidor — barg mog'ori",           "icon": "🍅"},
    "Potato___Late_blight":    {"crop": "Potato",   "uz": "Kartoshka — kech yanib ketish",    "icon": "🥔"},
    "Potato___Early_blight":   {"crop": "Potato",   "uz": "Kartoshka — erta yanib ketish",    "icon": "🥔"},
    "Grape___Black_rot":       {"crop": "Grape",    "uz": "Uzum — qora chirish",              "icon": "🍇"},
    "Apple___Apple_scab":      {"crop": "Apple",    "uz": "Olma — qo'tir",                   "icon": "🍎"},
}

RISK_ADVICE = {
    "high":   "⚠️ Высокий риск — немедленная обработка фунгицидом рекомендована",
    "medium": "🟡 Средний риск — усиленный мониторинг, профилактическая обработка",
    "low":    "✅ Низкий риск — плановый мониторинг достаточен",
}


def risk_level(prob: float) -> str:
    if prob >= 0.65: return "high"
    if prob >= 0.35: return "medium"
    return "low"


# ── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "AgroMonitor UZ API v1.0"}


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_loaded": not isinstance(predictor, MockPredictor),
        "db_path": str(DB_PATH),
        "observations_count": Session().query(Observation).count(),
    }


@app.post("/predict")
async def predict(
    image:       UploadFile = File(...),
    region:      str  = Form(...),
    season:      str  = Form(...),
    temperature: float = Form(...),
    humidity:    float = Form(...),
    rainfall:    float = Form(10.0),
    soil_ph:     float = Form(7.5),
    soil_type:   str  = Form("loam_sierozem"),
):
    # Save image
    img_id   = str(uuid.uuid4())
    ext      = Path(image.filename).suffix or ".jpg"
    img_name = f"{img_id}{ext}"
    img_path = UPLOAD_DIR / img_name
    content  = await image.read()
    img_path.write_bytes(content)

    # Run prediction
    predictor = get_predictor()
    try:
        result = predictor.predict(
            image_path  = str(img_path),
            temperature = temperature,
            humidity    = humidity,
            rainfall    = rainfall,
            soil_ph     = soil_ph,
            soil_type   = soil_type,
            season      = season,
            region      = region,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    top3         = result["top3"]
    risk_score   = result["risk_score"]
    best_label, best_prob = top3[0]
    info         = DISEASE_INFO.get(best_label, {"crop": "Unknown", "uz": best_label, "icon": "🌿"})
    rl           = risk_level(best_prob)

    # Save to DB
    obs_id = str(uuid.uuid4())
    db     = Session()
    obs    = Observation(
        id                  = obs_id,
        region              = region,
        season              = season,
        avg_temperature_c   = temperature,
        humidity_pct        = humidity,
        rainfall_mm         = rainfall,
        soil_type           = soil_type,
        soil_ph             = soil_ph,
        image_path          = str(img_path),
        image_filename      = img_name,
        predicted_class     = best_label,
        predicted_label     = 0,
        disease_probability = best_prob,
        top3_json           = json.dumps(top3),
    )
    db.add(obs)
    db.commit()
    db.close()

    return {
        "observation_id":    obs_id,
        "predicted_disease": best_label,
        "disease_name_uz":   info["uz"],
        "crop":              info["crop"],
        "icon":              info["icon"],
        "probability":       round(best_prob, 4),
        "risk_score":        round(risk_score, 4),
        "risk_level":        rl,
        "advice":            RISK_ADVICE[rl],
        "top3": [
            {
                "disease": d,
                "name_uz": DISEASE_INFO.get(d, {}).get("uz", d),
                "prob":    round(p, 4),
                "icon":    DISEASE_INFO.get(d, {}).get("icon", "🌿"),
            }
            for d, p in top3
        ],
        "context": {
            "region": region, "season": season,
            "temperature": temperature, "humidity": humidity,
        },
    }


@app.post("/feedback/{observation_id}")
async def add_feedback(
    observation_id: str,
    true_class:      str  = Form(...),
    agronomist_name: str  = Form(""),
    notes:           str  = Form(""),
):
    db  = Session()
    obs = db.query(Observation).filter(Observation.id == observation_id).first()
    if not obs:
        db.close()
        raise HTTPException(status_code=404, detail="Observation not found")

    obs.has_feedback      = True
    obs.true_class        = true_class
    obs.agronomist_name   = agronomist_name
    obs.feedback_notes    = notes
    obs.feedback_at       = datetime.utcnow()
    db.commit()
    db.close()
    return {"status": "ok", "message": "Feedback saved", "observation_id": observation_id}


@app.get("/observations")
def list_observations(limit: int = 50, skip: int = 0):
    db   = Session()
    rows = db.query(Observation).order_by(
        Observation.created_at.desc()
    ).offset(skip).limit(limit).all()
    db.close()
    return [
        {
            "id":                 r.id,
            "created_at":         r.created_at.isoformat() if r.created_at else None,
            "region":             r.region,
            "season":             r.season,
            "predicted_disease":  r.predicted_class,
            "probability":        r.disease_probability,
            "has_feedback":       r.has_feedback,
            "true_class":         r.true_class,
            "agronomist":         r.agronomist_name,
        }
        for r in rows
    ]


@app.get("/dataset/export")
def export_dataset(only_with_feedback: bool = False):
    """Экспортирует CSV датасет для дообучения модели."""
    db  = Session()
    q   = db.query(Observation)
    if only_with_feedback:
        q = q.filter(Observation.has_feedback == True)
    rows = q.all()
    db.close()

    fields = [
        "id","created_at","region","season",
        "avg_temperature_c","humidity_pct","rainfall_mm",
        "soil_type","soil_ph","image_filename",
        "predicted_class","disease_probability",
        "has_feedback","true_class","agronomist_name","feedback_notes",
    ]

    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({f: getattr(r, f, None) for f in fields})

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=agromonitor_real_dataset.csv"},
    )


@app.get("/stats")
def stats():
    db    = Session()
    total = db.query(Observation).count()
    with_fb = db.query(Observation).filter(Observation.has_feedback == True).count()
    by_region = {}
    for obs in db.query(Observation).all():
        by_region[obs.region] = by_region.get(obs.region, 0) + 1
    db.close()
    return {
        "total_observations":    total,
        "with_feedback":         with_fb,
        "without_feedback":      total - with_fb,
        "by_region":             by_region,
        "dataset_readiness_pct": round(with_fb / total * 100, 1) if total else 0,
    }
