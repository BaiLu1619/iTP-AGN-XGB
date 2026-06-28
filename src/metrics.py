from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    spe = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.5

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        pr_auc = 0.5

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "SEN": recall_score(y_true, y_pred, zero_division=0),
        "SPE": spe,
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }

