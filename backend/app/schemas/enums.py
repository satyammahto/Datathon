from enum import Enum


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class MLTaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_ANALYSIS = "time_analysis"
    NOT_APPLICABLE = "not_applicable"


class ValidationStrategy(str, Enum):
    STRATIFIED_K_FOLD = "StratifiedKFold"
    K_FOLD = "KFold"
    TIME_SERIES_SPLIT = "TimeSeriesSplit"
    GROUP_K_FOLD = "GroupKFold"
    TRAIN_TEST_SPLIT = "TrainTestSplit"


class ColumnDataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
