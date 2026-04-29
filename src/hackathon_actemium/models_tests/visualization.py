from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def _prediction_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "y_true" and pd.api.types.is_numeric_dtype(df[c])]


def _sample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("-", "_")
    )


def _model_order_from_metrics(metrics: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    columns = list(columns)
    if metrics.empty or "model" not in metrics.columns or "rmse" not in metrics.columns:
        return columns
    ordered = metrics.sort_values("rmse")["model"].astype(str).tolist()
    return [m for m in ordered if m in columns] + [m for m in columns if m not in ordered]


def plot_metric_bars(metrics: pd.DataFrame, plots_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    paths: list[Path] = []
    if metrics.empty or "model" not in metrics.columns:
        return paths

    metrics = metrics.copy().sort_values("rmse")
    models = metrics["model"].astype(str)
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.38
    ax.bar(x - width / 2, metrics["rmse"], width, label="RMSE", color="#3b82f6")
    ax.bar(x + width / 2, metrics["mae"], width, label="MAE", color="#10b981")
    ax.set_title("Comparaison des erreurs par modele")
    ax.set_ylabel("Erreur")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "metrics_rmse_mae.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    if "skill_vs_persistence" in metrics.columns:
        skill = metrics["skill_vs_persistence"].astype(float) * 100.0
        clipped = skill.clip(lower=-100, upper=100)
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in clipped]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(models, clipped, color=colors)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title("Gain ou perte vs baseline persistence")
        ax.set_ylabel("Skill vs persistence (%)")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.text(
            0.99,
            0.02,
            "Valeurs clippees entre -100% et +100% pour la lisibilite",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#6b7280",
        )
        fig.tight_layout()
        path = plots_dir / "skill_vs_persistence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_predictions(predictions: pd.DataFrame, plots_dir: Path, name: str, metrics: pd.DataFrame, max_points: int) -> list[Path]:
    plt = _setup_matplotlib()
    paths: list[Path] = []
    if predictions.empty or "y_true" not in predictions.columns:
        return paths

    model_cols = _model_order_from_metrics(metrics, _prediction_columns(predictions))
    if not model_cols:
        return paths

    sampled = _sample_frame(predictions[["y_true"] + model_cols], max_points)
    x = np.arange(len(sampled))
    prefix = _safe_filename(name)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, sampled["y_true"], label="y_true", color="#111827", linewidth=2.2)
    for col in model_cols:
        ax.plot(x, sampled[col], label=col, linewidth=1.3, alpha=0.8)
    ax.set_title(f"Prediction vs verite terrain - {name}")
    ax.set_xlabel("Index temporel echantillonne")
    ax.set_ylabel("Niveau predit / observe")
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    path = plots_dir / f"{prefix}_predictions_vs_true.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    residuals = sampled[model_cols].subtract(sampled["y_true"], axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for col in model_cols:
        axes[0].plot(x, residuals[col], label=col, linewidth=1.1, alpha=0.8)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_title(f"Erreurs residuelles dans le temps - {name}")
    axes[0].set_ylabel("Prediction - reel")
    axes[0].legend(ncols=2, fontsize=9)

    for col in model_cols:
        axes[1].hist(residuals[col].dropna(), bins=40, alpha=0.35, label=col)
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_title("Distribution des erreurs")
    axes[1].set_xlabel("Erreur residuelle")
    axes[1].set_ylabel("Frequence")
    axes[1].legend(ncols=2, fontsize=9)
    fig.tight_layout()
    path = plots_dir / f"{prefix}_residuals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    scatter_cols = model_cols[: min(4, len(model_cols))]
    fig, axes = plt.subplots(1, len(scatter_cols), figsize=(5 * len(scatter_cols), 4.8), squeeze=False)
    y_true = sampled["y_true"]
    lo = float(np.nanmin([y_true.min(), sampled[scatter_cols].min().min()]))
    hi = float(np.nanmax([y_true.max(), sampled[scatter_cols].max().max()]))
    for ax, col in zip(axes[0], scatter_cols):
        ax.scatter(y_true, sampled[col], s=10, alpha=0.45, color="#2563eb")
        ax.plot([lo, hi], [lo, hi], color="#dc2626", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("Reel")
        ax.set_ylabel("Predit")
    fig.suptitle(f"Alignement prediction / reel - {name}", y=1.02)
    fig.tight_layout()
    path = plots_dir / f"{prefix}_predicted_vs_actual.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    best_col = model_cols[0]
    abs_errors = (predictions[best_col] - predictions["y_true"]).abs().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(np.arange(len(abs_errors)), abs_errors.values, color="#f97316")
    ax.set_title(f"Top 20 erreurs absolues - {best_col} ({name})")
    ax.set_xlabel("Rang de l'erreur")
    ax.set_ylabel("Erreur absolue")
    fig.tight_layout()
    path = plots_dir / f"{prefix}_top_errors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def generate_diagnostic_plots(outdir: Path | str, max_points: int = 2000) -> list[Path]:
    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "model_comparison.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    paths: list[Path] = []
    paths.extend(plot_metric_bars(metrics, plots_dir))

    prediction_files = {
        "modeles_tabular": outdir / "predictions_tabular.csv",
        "kalman": outdir / "predictions_kalman.csv",
        "deep_learning": outdir / "predictions_dl.csv",
    }
    for name, path in prediction_files.items():
        if path.exists():
            predictions = pd.read_csv(path)
            paths.extend(plot_predictions(predictions, plots_dir, name, metrics, max_points=max_points))

    summary_path = plots_dir / "plots_created.txt"
    summary_path.write_text("\n".join(str(p.name) for p in paths), encoding="utf-8")
    paths.append(summary_path)
    return paths
