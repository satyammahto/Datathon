import os
import sys
import json
import pytest
from pathlib import Path

# Add project root and backend directory to sys.path so both 'backend.app' and 'app' are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app.schemas.discovery_contract import DiscoveryContract


@pytest.fixture
def mock_discovery_contract_data() -> dict:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "mock_discovery_contract.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_discovery_contract(mock_discovery_contract_data) -> DiscoveryContract:
    return DiscoveryContract.model_validate(mock_discovery_contract_data)
