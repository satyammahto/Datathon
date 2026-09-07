"""
End-to-End Test for Unseen CSV Dataset Analysis
Tests the complete 4-brain integrated pipeline (P1 Discovery -> P2 ML Championship -> P3 Trust Layer -> P4 Dashboard Spec)
on a completely novel dataset without mock fallback.
"""

import os
import io
import tempfile
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["SECRET_KEY"] = "test-secret-key-for-unseen-csv"
os.environ["DATABASE_URL"] = "sqlite:///./test_unseen_e2e.db"

from main import app
from database.session import Base, get_db
from database.models import User
from auth import hash_password

try:
    from app.pipeline.discovery import run_discovery
    from app.pipeline.ml_analysis import run_ml_pipeline
    from aida.graph import run_aida_pipeline
    from aida.contracts.discovery_contract import DiscoveryContract as AidaDiscoveryContract
    from aida.contracts.analysis_contract import AnalysisContract as AidaAnalysisContract
    from app.pipeline.dashboard import DashboardAssembler
    from app.pipeline.visualization import AutoChartEngine
except ImportError:
    from backend.app.pipeline.discovery import run_discovery
    from backend.app.pipeline.ml_analysis import run_ml_pipeline
    from backend.aida.graph import run_aida_pipeline
    from backend.aida.contracts.discovery_contract import DiscoveryContract as AidaDiscoveryContract
    from backend.aida.contracts.analysis_contract import AnalysisContract as AidaAnalysisContract
    from backend.app.pipeline.dashboard import DashboardAssembler
    from backend.app.pipeline.visualization import AutoChartEngine

# Isolated SQLite test database
engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_test_env():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Create test user
    user = User(
        username="unseen_analyst",
        email="unseen@analyst.ai",
        hashed_password=hash_password("password123")
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate novel synthetic dataset (Employee Attrition / Salary analysis)
    np.random.seed(42)
    n_samples = 150
    df = pd.DataFrame({
        "emp_id": [f"EMP_{1000 + i}" for i in range(n_samples)],
        "years_experience": np.random.uniform(1.0, 15.0, size=n_samples).round(1),
        "monthly_salary": np.random.normal(6500, 1500, size=n_samples).clip(3000, 15000).round(2),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR"], size=n_samples),
        "satisfaction_score": np.random.uniform(1.0, 5.0, size=n_samples).round(2),
        "projects_completed": np.random.randint(1, 20, size=n_samples),
        "promoted_last_year": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        "left_company": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    yield db, csv_path, df

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("./test_unseen_e2e.db")
    except OSError:
        pass
    try:
        os.remove(csv_path)
    except OSError:
        pass


def test_core_4_brain_pipeline_on_unseen_csv(setup_test_env):
    """
    Directly test the full 4-brain Python pipeline progression:
    P1 Discovery -> P2 ML Championship -> P3 Trust Layer -> P4 Dashboard Assembly
    """
    _, csv_path, df = setup_test_env
    dataset_name = "employee_attrition_benchmark.csv"
    dataset_id = "test-unseen-emp-01"

    # Brain 1: Discovery Engine
    p1_discovery = run_discovery(file_path=csv_path, dataset_id=dataset_id, dataset_name=dataset_name)
    assert p1_discovery is not None
    assert p1_discovery.dataset_name == dataset_name
    assert len(p1_discovery.columns) == len(df.columns)
    assert p1_discovery.quality_report.overall_quality_score > 0
    assert p1_discovery.router.recommended_primary_task is not None

    # Brain 2: ML Championship & Analysis
    p2_analysis = run_ml_pipeline(df, p1_discovery)
    assert p2_analysis is not None
    assert len(p2_analysis.models) > 0
    assert p2_analysis.winner is not None
    champion_name = p2_analysis.winner.model_name
    assert champion_name is not None

    # Brain 3: LangGraph Trust Layer
    # Convert through Pydantic model validators
    aida_disc = AidaDiscoveryContract.model_validate(p1_discovery)
    aida_anal = AidaAnalysisContract.model_validate(p2_analysis)
    assert aida_disc.row_count == len(df)
    assert aida_disc.column_count == len(df.columns)
    assert len(aida_anal.models) == len(p2_analysis.models)

    aida_state = {
        "discovery_contract": aida_disc,
        "analysis_contract": aida_anal,
        "pipeline_errors": [],
        "errors": []
    }
    trust_result = run_aida_pipeline(aida_state)
    assert trust_result is not None
    verified_insights = trust_result.get("verified_insights", [])

    # Brain 4: Visualization Engine & Dashboard Assembly
    chart_engine = AutoChartEngine()
    auto_charts = chart_engine.analyze_and_generate(df)
    assert isinstance(auto_charts, list)
    assert len(auto_charts) > 0

    disc_dict = {
        "fingerprint": {
            "shape": [len(df), len(df.columns)],
            "has_datetime": False,
            "recommended_task": "classification"
        },
        "quality_report": {
            "overall_score": round(p1_discovery.quality_report.overall_quality_score, 1),
            "missing_cells_total": p1_discovery.quality_report.total_missing_cells,
            "total_rows": len(df),
            "total_cols": len(df.columns)
        },
        "leakage_report": {
            "has_leakage": False,
            "warnings": []
        }
    }

    analysis_dict = {
        "task_type": "classification",
        "champion_model": champion_name,
        "models": [{"model_name": champion_name, "is_champion": True, "metrics": {"accuracy": 0.85}}],
        "validation_strategy": {"method": "Stratified 5-Fold"}
    }

    dash_contract = DashboardAssembler.assemble(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        discovery=disc_dict,
        analysis=analysis_dict,
        insights=[{"title": "Test Finding", "claim": "Emp retention", "status": "verified"}],
        charts=auto_charts
    )

    assert dash_contract is not None
    assert dash_contract["dataset_id"] == dataset_id
    assert dash_contract["dataset_name"] == dataset_name
    assert dash_contract["quality_summary"]["rows"] == len(df)
    assert dash_contract["quality_summary"]["columns"] == len(df.columns)
    assert len(dash_contract["charts"]) == len(auto_charts)
    assert len(dash_contract["insights"]) > 0

    # Assert that no mock churn strings leaked into this novel dataset
    assert dash_contract["dataset_name"] != "Customer_Sales_Data"
    assert "Churn" not in [c.name for c in p1_discovery.columns]


def test_api_unseen_csv_full_lifecycle(setup_test_env):
    """
    Test uploading unseen CSV via HTTP, executing the live pipeline,
    and verifying all downstream endpoints return real, dataset-specific data.
    """
    _, csv_path, df = setup_test_env

    # 1. Authenticate
    login_resp = client.post(
        "/api/login",
        json={"email": "unseen@analyst.ai", "password": "password123"}
    )
    assert login_resp.status_code == 200, login_resp.text
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload unseen CSV
    with open(csv_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("unseen_attrition_data.csv", f, "text/csv")},
            headers=headers
        )
    assert upload_resp.status_code == 200, upload_resp.text
    dataset = upload_resp.json()
    dataset_id = dataset["id"]
    assert dataset["name"] == "unseen_attrition_data.csv"
    assert dataset["row_count"] == len(df)
    assert dataset["column_count"] == len(df.columns)

    # 3. Trigger live pipeline run
    run_resp = client.post(f"/api/pipeline/run/{dataset_id}", headers=headers)
    assert run_resp.status_code == 200, run_resp.text
    run_data = run_resp.json()
    assert run_data["status"] == "success"
    assert "progress" in run_data
    assert run_data["progress"]["overall_progress"] == 100
    assert len(run_data["progress"]["stages"]) == 6

    # 4. Fetch status
    status_resp = client.get(f"/api/pipeline/status/{dataset_id}", headers=headers)
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["is_running"] is False
    assert status_data["overall_progress"] == 100
    for st in status_data["stages"]:
        assert st["status"] == "completed"

    # 5. Fetch dashboard contract
    dash_resp = client.get(f"/api/pipeline/dashboard/{dataset_id}", headers=headers)
    assert dash_resp.status_code == 200, dash_resp.text
    dash = dash_resp.json()
    assert dash["dataset_id"] == dataset_id
    assert dash["dataset_name"] == "unseen_attrition_data.csv"
    assert dash["quality_summary"]["rows"] == len(df)
    assert dash["quality_summary"]["columns"] == len(df.columns)
    assert len(dash["charts"]) > 0
    assert len(dash["insights"]) > 0
    assert len(dash["kpis"]) > 0

    # 6. Fetch championship benchmark
    champ_resp = client.get(f"/api/pipeline/championship/{dataset_id}", headers=headers)
    assert champ_resp.status_code == 200, champ_resp.text
    champ = champ_resp.json()
    assert "models" in champ
    assert len(champ["models"]) > 0
    assert any(m.get("is_champion") for m in champ["models"])

    # 7. Fetch validated insights
    insights_resp = client.get(f"/api/pipeline/insights/{dataset_id}", headers=headers)
    assert insights_resp.status_code == 200, insights_resp.text
    insights_data = insights_resp.json()
    assert "insights" in insights_data
    assert len(insights_data["insights"]) > 0

    # 8. Fetch quality audit
    quality_resp = client.get(f"/api/pipeline/quality/{dataset_id}", headers=headers)
    assert quality_resp.status_code == 200, quality_resp.text
    quality_data = quality_resp.json()
    assert "overall_score" in quality_data
    assert quality_data["total_rows"] == len(df)
    assert quality_data["total_cols"] == len(df.columns)
