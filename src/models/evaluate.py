"""
Evaluation utilities for Churn Prediction models.
Includes metrics calculation, optimal threshold tuning, and feature importance analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_top_k_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    k_percents: List[float] = [1, 2, 5, 10, 20, 50, 75, 100],
) -> pd.DataFrame:
    """Calculate Top-K% Decile / Lift Metrics for churn risk ranking.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_probs : np.ndarray
        Predicted churn probabilities.
    k_percents : list of float, default=[1, 2, 5, 10, 20]
        Top percentile cutoffs (e.g. 1%, 2%, 5%, 10%, 20%).

    Returns
    -------
    pd.DataFrame
        Table containing Top-K samples, churners captured, precision@k, recall@k, and lift@k.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_samples = len(y_true)
    total_positives = int(np.sum(y_true == 1))
    baseline_rate = (total_positives / n_samples) if n_samples > 0 else 0.0

    # Sort descending by predicted probability
    sorted_indices = np.argsort(-y_probs)
    sorted_y_true = y_true[sorted_indices]

    records = []
    for k_pct in k_percents:
        k_count = max(1, int(n_samples * (k_pct / 100.0)))
        top_k_labels = sorted_y_true[:k_count]
        captured_positives = int(np.sum(top_k_labels == 1))

        precision_at_k = captured_positives / k_count if k_count > 0 else 0.0
        recall_at_k = captured_positives / total_positives if total_positives > 0 else 0.0
        lift_at_k = precision_at_k / baseline_rate if baseline_rate > 0 else 1.0

        records.append({
            "k_percent": float(k_pct),
            "n_targeted": int(k_count),
            "captured_churns": int(captured_positives),
            "total_churns": int(total_positives),
            "precision_at_k": float(precision_at_k),
            "recall_at_k": float(recall_at_k),
            "lift_at_k": float(lift_at_k),
        })

    return pd.DataFrame(records)


def format_top_k_table(df_top_k: pd.DataFrame, title: str = "Top-K Risk Scoring & Cumulative Lift Analysis") -> str:
    """Format Top-K metrics DataFrame into a clean Markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, row in df_top_k.iterrows():
        k_str = f"**Top {row['k_percent']:.0f}%**"
        targeted = f"`{int(row['n_targeted']):,}`"
        captured = f"`{int(row['captured_churns']):,} / {int(row['total_churns']):,}`"
        prec = f"`{row['precision_at_k']*100:.2f}%`"
        rec = f"`{row['recall_at_k']*100:.2f}%`"
        lift = f"`{row['lift_at_k']:.2f}x`"
        lines.append(f"| {k_str} | {targeted} | {captured} | {prec} | {rec} | {lift} |")
    lines.append("")
    return "\n".join(lines)


def calculate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_probs : np.ndarray
        Predicted probabilities for the positive class (1).
    threshold : float, default=0.5
        Decision threshold to convert probabilities into binary predictions.

    Returns
    -------
    dict
        Dictionary containing calculated metrics, Top-K lift, and confusion matrix details.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    # Check if both classes are present in y_true
    has_both_classes = len(np.unique(y_true)) > 1

    roc_auc = float(roc_auc_score(y_true, y_probs)) if has_both_classes else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if has_both_classes else 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if only 1 class present
        tn = int(cm[0, 0]) if len(cm) > 0 else 0
        fp, fn, tp = 0, 0, 0

    # Calculate Top-K metrics
    df_top_k = calculate_top_k_metrics(y_true, y_probs, k_percents=[1, 2, 5, 10, 20])
    p5_row = df_top_k[df_top_k["k_percent"] == 5]
    r10_row = df_top_k[df_top_k["k_percent"] == 10]
    p5_val = float(p5_row["precision_at_k"].values[0]) if not p5_row.empty else 0.0
    r10_val = float(r10_row["recall_at_k"].values[0]) if not r10_row.empty else 0.0
    lift5_val = float(p5_row["lift_at_k"].values[0]) if not p5_row.empty else 1.0

    metrics = {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_top_5_pct": p5_val,
        "recall_top_10_pct": r10_val,
        "lift_top_5_pct": lift5_val,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probs)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "top_k_metrics": df_top_k.to_dict(orient="records"),
        "support_total": int(len(y_true)),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
    }
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1",
    num_steps: int = 100,
) -> Tuple[float, float]:
    """Find the optimal decision threshold that maximizes the specified metric.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_probs : np.ndarray
        Predicted probabilities for positive class.
    metric : str, default='f1'
        Metric to optimize ('f1', 'f1_macro', 'precision', 'recall').
    num_steps : int, default=100
        Number of candidate thresholds to evaluate between 0.01 and 0.99.

    Returns
    -------
    tuple
        (best_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    best_score = -1.0

    for th in thresholds:
        y_pred = (y_probs >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "f1_macro":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric for threshold tuning: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(th)

    return best_threshold, float(best_score)


def extract_feature_importances(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and aggregate feature importances from a trained XGBoost model.

    Parameters
    ----------
    model : Any
        Trained XGBoost model (XGBClassifier or Booster).
    feature_names : list of str
        List of feature names matching model training columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature names, gain, weight, and cover metrics.
    """
    try:
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        score_weight = booster.get_score(importance_type="weight")
        score_cover = booster.get_score(importance_type="cover")
    except Exception:
        # Fallback to feature_importances_ attribute if get_booster fails
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df
        return pd.DataFrame(columns=["feature", "gain", "weight", "cover"])

    # Map raw booster feature keys (e.g., 'f0', 'f1', or actual names)
    data = []
    for i, col in enumerate(feature_names):
        # Try both the column name and 'f{i}'
        gain = score_gain.get(col, score_gain.get(f"f{i}", 0.0))
        weight = score_weight.get(col, score_weight.get(f"f{i}", 0.0))
        cover = score_cover.get(col, score_cover.get(f"f{i}", 0.0))
        data.append({
            "feature": col,
            "gain": float(gain),
            "weight": float(weight),
            "cover": float(cover),
        })

    df_importance = pd.DataFrame(data)
    # Calculate percentage share of total gain
    total_gain = df_importance["gain"].sum()
    if total_gain > 0:
        df_importance["gain_ratio"] = df_importance["gain"] / total_gain
    else:
        df_importance["gain_ratio"] = 0.0

    df_importance = df_importance.sort_values(by="gain", ascending=False).reset_index(drop=True)
    return df_importance


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Summary") -> str:
    """Format metrics dictionary into a readable markdown table string."""
    cm = metrics["confusion_matrix"]
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Decision Threshold** | `{metrics['threshold']:.4f}` |",
        f"| **ROC-AUC** | `{metrics['roc_auc']:.4f}` |",
        f"| **PR-AUC (Average Precision)** | `{metrics['pr_auc']:.4f}` |",
        f"| **Precision@Top 5%** | `{metrics.get('precision_top_5_pct', 0.0)*100:.2f}%` (Lift: `{metrics.get('lift_top_5_pct', 1.0):.2f}x`) |",
        f"| **Recall@Top 10%** | `{metrics.get('recall_top_10_pct', 0.0)*100:.2f}%` |",
        f"| **Accuracy** | `{metrics['accuracy']:.4f}` |",
        f"| **Precision (Churn=1)** | `{metrics['precision']:.4f}` |",
        f"| **Recall (Churn=1)** | `{metrics['recall']:.4f}` |",
        f"| **F1-Score (Churn=1)** | `{metrics['f1']:.4f}` |",
        f"| **Macro F1-Score** | `{metrics['f1_macro']:.4f}` |",
        f"| **Brier Score** | `{metrics['brier_score']:.4f}` |",
        "",
        "**Confusion Matrix:**",
        f"- True Negative (TN): `{cm['tn']:,}` | False Positive (FP): `{cm['fp']:,}`",
        f"- False Negative (FN): `{cm['fn']:,}` | True Positive (TP): `{cm['tp']:,}`",
        f"- Total Samples: `{metrics['support_total']:,}` (Positive: `{metrics['support_positive']:,}`, Negative: `{metrics['support_negative']:,}`)",
        "",
    ]
    if "top_k_metrics" in metrics and metrics["top_k_metrics"]:
        df_top_k = pd.DataFrame(metrics["top_k_metrics"])
        lines.append(format_top_k_table(df_top_k, f"Top-K Ranking & Lift Breakdown ({title})"))
    return "\n".join(lines)


def plot_feature_importances(
    df_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Plot and save a horizontal bar chart of the top feature importances.

    Parameters
    ----------
    df_importance : pd.DataFrame
        DataFrame with 'feature' and 'gain' (or 'importance') columns.
    top_n : int, default=20
        Number of top features to display.
    output_path : str, optional
        File path to save the chart (.png).
    title : str, optional
        Custom title for the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df_importance.empty:
        print("[WARNING] Feature importance DataFrame is empty. Skipping plot.")
        return None

    score_col = "gain" if "gain" in df_importance.columns else "importance"
    ratio_col = "gain_ratio" if "gain_ratio" in df_importance.columns else None

    # Filter top_n and sort ascending for horizontal bar plot (highest on top)
    df_top = df_importance.head(top_n).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_top) * 0.35)))

    # Color gradient from teal to dark blue
    n_bars = len(df_top)
    colors = plt.cm.viridis(np.linspace(0.4, 0.85, n_bars))

    bars = ax.barh(
        df_top["feature"],
        df_top[score_col],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    max_val = df_top[score_col].max()
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1.0)
    ax.set_xlabel(f"Importance Score ({score_col.capitalize()})", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Add score and percentage annotations
    for idx, (bar, (_, row)) in enumerate(zip(bars, df_top.iterrows())):
        width = bar.get_width()
        ratio_str = f" ({row[ratio_col]*100:.1f}%)" if ratio_col and pd.notnull(row.get(ratio_col)) else ""
        label_text = f"{width:,.2f}{ratio_str}"
        ax.text(
            width + (max_val * 0.012),
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    chart_title = title or f"Top {len(df_top)} Most Important Features (Feature Importance)"
    ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if output_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] Saved Feature Importance Plot to: '{output_path}'")

    plt.close(fig)
    return fig

