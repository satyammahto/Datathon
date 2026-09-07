from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from database.session import engine, Base

# Import models to ensure they are registered with SQLAlchemy
from database import models

# Create all tables
Base.metadata.create_all(bind=engine)

from api.routes import router as api_router
from api.pipeline_routes import router as pipeline_router
try:
    from backend.app.api.routes import router as unified_aida_router
except ImportError:
    from app.api.routes import router as unified_aida_router

try:
    from aida.api import router as aida_trust_router
except ImportError:
    from backend.aida.api import router as aida_trust_router

app = FastAPI(
    title="DataMind AI API",
    description="Backend API for the Autonomous Data Analyst Agent",
    version="1.0.0"
)

# Setup CORS — allow all origins since we use Bearer token auth (not cookies)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(unified_aida_router)
app.include_router(aida_trust_router, prefix="/api")
app.include_router(aida_trust_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to DataMind AI API"}
# trigger reload
