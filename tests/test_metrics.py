import numpy as np
from autogluon_agent.metrics import METRICS, score, oriented, default_metric

def test_defaults():
    assert default_metric("binary") == "roc_auc"
    assert default_metric("multiclass") == "log_loss"
    assert default_metric("regression") == "rmse"

def test_rmse_oriented_is_negative():
    v = score("rmse", np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert np.isclose(v, np.sqrt(2.0))
    assert oriented("rmse", v) == -v

def test_accuracy_from_binary_proba():
    assert score("accuracy", np.array([0, 1, 1]), np.array([0.2, 0.9, 0.4])) == 2 / 3

def test_multiclass_log_loss_with_missing_class_in_fold():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert score("log_loss", np.array(["a", "b"]), p, labels=["a", "b", "c"]) > 0

def test_ag_names():
    assert METRICS["rmse"].ag_name == "root_mean_squared_error"
    assert METRICS["pr_auc"].ag_name == "average_precision"
