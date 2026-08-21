"""
Evaluation metrics for trajectory prediction and red-light classification.

Trajectory metrics:
    ADE  — Average Displacement Error (lower is better)
    FDE  — Final Displacement Error (lower is better)
    NLL  — Negative Log-Likelihood (lower is better, for probabilistic methods)

Classification metrics:
    Accuracy, Precision, Recall, F1-score, AUC-ROC
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ======================================================================
# Trajectory Prediction Metrics
# ======================================================================

def compute_ade(
    pred_trajectories: np.ndarray,    # (N, pred_len, 2) or (B, N, pred_len, 2)
    gt_trajectory: np.ndarray,        # (pred_len, 2) or (B, pred_len, 2)
) -> float:
    """
    Average Displacement Error — mean L2 distance over all prediction steps.

    For multi-modal (N samples): ADE = mean over samples and steps.
    """
    if pred_trajectories.ndim == 4:
        # (B, N, pred_len, 2) → average over samples first
        pred_trajectories = pred_trajectories.mean(axis=1)

    # (..., pred_len, 2)
    diff = pred_trajectories - gt_trajectory
    l2 = np.sqrt((diff ** 2).sum(axis=-1))  # (..., pred_len)
    return float(l2.mean())


def compute_fde(
    pred_trajectories: np.ndarray,
    gt_trajectory: np.ndarray,
) -> float:
    """
    Final Displacement Error — L2 distance at the last prediction step only.
    """
    if pred_trajectories.ndim == 4:
        pred_trajectories = pred_trajectories.mean(axis=1)

    final_pred = pred_trajectories[..., -1, :]   # (..., 2)
    final_gt = gt_trajectory[..., -1, :]          # (..., 2)
    diff = final_pred - final_gt
    l2 = np.sqrt((diff ** 2).sum(axis=-1))
    return float(l2.mean())


def compute_nll(
    log_probs: np.ndarray,  # (N,) log-probability per sample (higher = better)
) -> float:
    """
    Negative Log-Likelihood — average -log p(y|x) over samples.
    Lower is better.

    For flow-based models, NLL = -mean(log_probs).
    """
    return float(-log_probs.mean()) if len(log_probs) > 0 else float("inf")


def compute_trajectory_metrics(
    predictions: List[dict],     # each: {"pred_mean", "pred_samples", "log_probs"}
    ground_truth: np.ndarray,    # (B, pred_len, 2)
) -> Dict[str, float]:
    """
    Compute ADE, FDE, NLL for a batch of predictions.

    Parameters
    ----------
    predictions : list of dict
        Each element from model output (mean, samples, log_probs).
    ground_truth : np.ndarray (B, pred_len, 2)

    Returns
    -------
    dict with "ADE", "FDE", "NLL"
    """
    B = len(predictions)
    all_ade, all_fde, all_nll = [], [], []

    for b in range(B):
        pred = predictions[b]
        gt = ground_truth[b]

        # ADE / FDE from mean prediction
        mean_pred = pred.get("pred_mean", pred.get("mean"))
        if mean_pred is not None:
            all_ade.append(compute_ade(mean_pred, gt))
            all_fde.append(compute_fde(mean_pred, gt))

        # NLL from log_probs
        log_probs = pred.get("log_probs")
        if log_probs is not None:
            all_nll.append(compute_nll(np.array(log_probs)))

    return {
        "ADE": float(np.mean(all_ade)) if all_ade else float("inf"),
        "FDE": float(np.mean(all_fde)) if all_fde else float("inf"),
        "NLL": float(np.mean(all_nll)) if all_nll else float("inf"),
    }


# ======================================================================
# Red-Light Classification Metrics
# ======================================================================

def compute_classification_metrics(
    y_true: np.ndarray,       # (N,)  binary labels (0 or 1)
    y_pred: np.ndarray,       # (N,)  binary predictions
    y_prob: np.ndarray,       # (N,)  prediction probabilities
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1, AUC.

    Returns
    -------
    dict with metric_name → value
    """
    metrics = {}

    try:
        metrics["Accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["Accuracy"] = 0.0

    try:
        metrics["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["Precision"] = 0.0

    try:
        metrics["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["Recall"] = 0.0

    try:
        metrics["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["F1"] = 0.0

    try:
        if len(np.unique(y_true)) >= 2:
            metrics["AUC"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["AUC"] = 0.5
    except Exception:
        metrics["AUC"] = 0.5

    return metrics


# ======================================================================
# Result CSV Export
# ======================================================================

def export_trajectory_results_csv(
    results: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Export trajectory prediction results to CSV."""
    import pandas as pd

    rows = []
    for method, metrics in results.items():
        rows.append({
            "Method": method,
            "ADE": f"{metrics.get('ADE', float('nan')):.4f}",
            "FDE": f"{metrics.get('FDE', float('nan')):.4f}",
            "NLL": f"{metrics.get('NLL', float('nan')):.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Trajectory results saved to {output_path}")
    return df


def export_classification_results_csv(
    results: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Export red-light classification results to CSV."""
    import pandas as pd

    rows = []
    for method, metrics in results.items():
        rows.append({
            "Method": method,
            "Accuracy": f"{metrics.get('Accuracy', float('nan')):.4f}",
            "Precision": f"{metrics.get('Precision', float('nan')):.4f}",
            "Recall": f"{metrics.get('Recall', float('nan')):.4f}",
            "F1": f"{metrics.get('F1', float('nan')):.4f}",
            "AUC": f"{metrics.get('AUC', float('nan')):.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Classification results saved to {output_path}")
    return df
