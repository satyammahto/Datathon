"""
Tests for FastAPI Application Endpoints (Person 4 - Phase 4.0)
Verifies upload flow, stage progression, DashboardSpec retrieval,
error handling, and sample dataset catalog.
"""
import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify healthcheck endpoint returns healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_sample_datasets_catalog():
    """Verify sample datasets catalog returns available demo datasets."""
    res = client.get("/api/sample-datasets")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4
    sample_ids = [s["sample_id"] for s in samples]
    assert "customer_churn" in sample_ids
    assert "real_estate" in sample_ids


def test_analyze_sample_dataset_end_to_end():
    """Verify 1-click sample analysis executes full 15-stage pipeline."""
    res = client.post("/api/analyze-sample/customer_churn")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["current_stage"] == 15
    analysis_id = data["analysis_id"]

    # Verify progress endpoint
    prog_res = client.get(f"/api/analysis/{analysis_id}/progress")
    assert prog_res.status_code == 200
    prog_data = prog_res.json()
    assert prog_data["status"] == "completed"
    assert len(prog_data["stages"]) == 15

    # Verify dashboard endpoint
    dash_res = client.get(f"/api/analysis/{analysis_id}/dashboard")
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert dash_data["status"] == "SUCCESS"
    assert len(dash_data["kpis"]) >= 5
    assert len(dash_data["insights"]) > 0

    # Verify sub-endpoints
    insights_res = client.get(f"/api/analysis/{analysis_id}/insights")
    assert insights_res.status_code == 200

    models_res = client.get(f"/api/analysis/{analysis_id}/models")
    assert models_res.status_code == 200
    assert "leaderboard" in models_res.json()

    quality_res = client.get(f"/api/analysis/{analysis_id}/quality")
    assert quality_res.status_code == 200
    assert "row_count" in quality_res.json()

    errors_res = client.get(f"/api/analysis/{analysis_id}/errors")
    assert errors_res.status_code == 200

    report_res = client.get(f"/api/analysis/{analysis_id}/report")
    assert report_res.status_code == 200
    assert "dataset_overview" in report_res.json()


def test_file_upload_csv_analysis():
    """Verify real CSV file upload executes pipeline and returns task ID."""
    df = pd.DataFrame({
        "x1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    files = {"file": ("test_upload.csv", io.BytesIO(csv_bytes), "text/csv")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "analysis_id" in data
    assert data["filename"] == "test_upload.csv"


def test_invalid_file_upload_error_handling():
    """Verify that uploading invalid non-CSV/XLSX contents returns clean 400 error."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 400
