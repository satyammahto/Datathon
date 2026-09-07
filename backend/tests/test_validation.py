import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import ValidationReport
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.pipeline.validation import select_validation_strategy, build_cv_splitter


def test_select_validation_classification():
    disc = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = select_validation_strategy(None, disc, MLTaskType.CLASSIFICATION)
    assert val.status == PipelineStatus.SUCCESS
    assert val.strategy == ValidationStrategy.STRATIFIED_K_FOLD.value
    assert val.folds == 5
    assert val.primary_metric in ["f1", "f1_macro"]
    assert val.split_details.get("shuffle") is True


def test_select_validation_regression():
    disc = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(regression=True),
    )
    val = select_validation_strategy(None, disc, MLTaskType.REGRESSION)
    assert val.status == PipelineStatus.SUCCESS
    assert val.strategy == ValidationStrategy.K_FOLD.value
    assert val.folds == 5
    assert val.primary_metric == "rmse"


def test_select_validation_time_aware():
    disc = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(regression=True, time_analysis=True),
    )
    val = select_validation_strategy(None, disc, MLTaskType.REGRESSION)
    assert val.status == PipelineStatus.SUCCESS
    assert val.strategy == ValidationStrategy.TIME_SERIES_SPLIT.value
    assert val.split_details.get("temporal_ordering") is True
    assert val.split_details.get("shuffle") is False


def test_build_cv_splitter_kfold():
    val = ValidationReport(
        strategy=ValidationStrategy.K_FOLD.value,
        folds=5,
        random_seed=42,
    )
    splitter, err = build_cv_splitter(val, n_samples=50)
    assert err is None
    assert isinstance(splitter, KFold)
    assert splitter.get_n_splits() == 5


def test_build_cv_splitter_stratified():
    val = ValidationReport(
        strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
        folds=5,
        random_seed=42,
    )
    y = np.array([0] * 25 + [1] * 25)
    splitter, err = build_cv_splitter(val, n_samples=50, y=y)
    assert err is None
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.get_n_splits() == 5


def test_build_cv_splitter_small_data_fold_reduction():
    # If smallest class has 3 samples, reduce folds to 3
    val = ValidationReport(
        strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
        folds=5,
        random_seed=42,
    )
    y = np.array([0] * 30 + [1] * 3)
    splitter, err = build_cv_splitter(val, n_samples=33, y=y)
    assert err is None
    assert splitter.get_n_splits() == 3


def test_build_cv_splitter_single_sample_class_fails_safely():
    # If smallest class has only 1 sample, stratified CV is mathematically impossible
    val = ValidationReport(
        strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
        folds=5,
        random_seed=42,
    )
    y = np.array([0] * 30 + [1] * 1)
    splitter, err = build_cv_splitter(val, n_samples=31, y=y)
    assert splitter is None
    assert "at least 2 samples per class" in err


def test_build_cv_splitter_insufficient_samples():
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=5)
    splitter, err = build_cv_splitter(val, n_samples=1)
    assert splitter is None
    assert "Insufficient observations" in err


def test_build_cv_splitter_timeseries():
    val = ValidationReport(
        strategy=ValidationStrategy.TIME_SERIES_SPLIT.value,
        folds=4,
    )
    splitter, err = build_cv_splitter(val, n_samples=20)
    assert err is None
    assert isinstance(splitter, TimeSeriesSplit)
    assert splitter.get_n_splits() == 4
