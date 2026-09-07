import os
import pytest

# Ensure environment variables are set for tests before loading any app modules
os.environ["SECRET_KEY"] = "test-secret-key-for-integration-tests"
os.environ["DATABASE_URL"] = "sqlite:///./test_integration.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from database.session import Base, get_db
from database.models import User
from auth import hash_password

# Use a separate test database
engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_database():
    # Setup
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Create test user
    user = User(
        username="integration_user",
        email="integration@test.com",
        hashed_password=hash_password("password123")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    yield db
    
    # Teardown
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    
    # Remove test db file
    try:
        os.remove("./test_integration.db")
    except OSError:
        pass

def test_upload_and_visualize(setup_database):
    db = setup_database
    
    # 1. Login to get token
    login_response = client.post(
        "/api/login",
        json={"email": "integration@test.com", "password": "password123"}
    )
    assert login_response.status_code == 200, login_response.text
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Upload dataset
    csv_content = "A,B,C\n1,cat,10.5\n2,dog,15.2\n3,cat,12.1"
    files = {"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
    
    upload_response = client.post(
        "/api/upload",
        headers=headers,
        files=files
    )
    assert upload_response.status_code == 200, upload_response.text
    dataset_id = upload_response.json()["id"]
    
    # 3. Request visualization
    viz_payload = {
        "dataset_id": dataset_id,
        "chart_type": "bar",
        "x_column": "B",
        "y_column": "C"
    }
    
    viz_response = client.post(
        "/api/visualize",
        headers=headers,
        json=viz_payload
    )
    assert viz_response.status_code == 200, viz_response.text
    
    viz_data = viz_response.json()
    assert viz_data["chart_type"] == "bar"
    assert "chart_data" in viz_data
    assert viz_data["chart_data"]["type"] == "bar"
