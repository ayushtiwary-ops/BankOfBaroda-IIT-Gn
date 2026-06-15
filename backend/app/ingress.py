"""Ingress API - produces events to Kafka, polls decisions from Postgres.

This is the stateless front door of the executable topology. It does NOT score
(the scoring pods do); it authenticates, produces the event to Kafka keyed by
identity_id, and lets the client poll the decision the pods wrote to Postgres.
The dashboard is served here.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .auth import make_auth
from .bus import TOPIC, make_producer
from .config import Settings
from .schemas import IdentityEvent
from .stores import PostgresDecisionStore

settings = Settings.from_env()
authenticate, require = make_auth(settings)
producer = make_producer(os.environ["PRAMAAN_KAFKA_BROKERS"])
decisions = PostgresDecisionStore(os.environ["PRAMAAN_DATABASE_URL"])
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

app = FastAPI(title="PRAMAAN - Ingress", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
    allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingress", "mode": settings.mode,
            "topic": TOPIC}


@app.post("/v1/events", status_code=202)
def ingest_event(event: IdentityEvent, ctx=Depends(require("events:write"))):
    # Produce keyed by identity_id (ordering + partitioned scaling); the pods score.
    event_id = uuid.uuid4().hex[:12]
    producer.send(TOPIC, key=event.identity_id,
                  value={"event_id": event_id, "event": event.model_dump(mode="json")})
    producer.flush()
    return JSONResponse(status_code=202,
                        content={"event_id": event_id, "status": "accepted"})


@app.get("/v1/decisions/{event_id}")
def get_decision(event_id: str, ctx=Depends(require("events:write"))):
    d = decisions.get(event_id)
    if d is None:
        raise HTTPException(status_code=404, detail="decision pending")
    return d  # client plane only (generic decision) holds end-to-end


@app.get("/")
def dashboard():
    return FileResponse(DEMO_DIR / "dashboard.html")
