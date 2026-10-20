"""Empirical Benchmark: Comparing CompactTab & CompactTab-Hybrid against SOTA Baselines.

Evaluates performance across Binary Classification, Multiclass Classification, and Regression:
- RandomForest (Baseline)
- MLP (Classical Neural Network)
- LightGBM (Leading GBDT)
- CatBoost (Leading GBDT)
- CompactTab (Our BatchEnsemble + Quantile PLE Architecture)
- CompactTab-Hybrid (Our Stacking of CompactTab with GBDTs)
"""

from __future__ import annotations

import time

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from compacttab.models.compacttab import CompactTabClassifier, CompactTabRegressor
from compacttab.models.hybrid import (
    CompactTabHybridClassifier,
    CompactTabHybridRegressor,
    _get_tree_classifier,
    _get_tree_regressor,
)

console = Console()


def run_benchmark():
    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]       COMPACTTAB & COMPACTTAB-HYBRID SOTA BENCHMARK SUITE       [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # -------------------------------------------------------------
    # 1. Binary Classification: Breast Cancer
    # -------------------------------------------------------------
    console.print("[bold yellow]1. Evaluating Binary Classification (Breast Cancer)...[/bold yellow]")
    bc = load_breast_cancer(as_frame=True)
    X_bc = bc.data
    y_bc = bc.target
    X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = train_test_split(
        X_bc, y_bc, test_size=0.25, random_state=42, stratify=y_bc
    )

    models_bin = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Torch-Style MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
        "LightGBM / CatBoost": _get_tree_classifier(random_state=42),
        "CompactTab (Ours)": CompactTabClassifier(
            preset="small", max_epochs=40, mixup_alpha=0.1, random_state=42
        ),
        "CompactTab-Hybrid (Ours)": CompactTabHybridClassifier(
            preset="small", cv_folds=3, max_epochs=35, random_state=42
        ),
    }

    results_bin = []
    for name, model in models_bin.items():
        t0 = time.perf_counter()
        model.fit(X_tr_bc, y_tr_bc)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = model.predict_proba(X_te_bc)[:, 1]
        preds = model.predict(X_te_bc)
        inf_t = (time.perf_counter() - t1) * 1000.0 / len(X_te_bc)

        acc = accuracy_score(y_te_bc, preds)
        auc = roc_auc_score(y_te_bc, probs)
        results_bin.append((name, acc, auc, fit_t, inf_t))

    table_bin = Table(title="Binary Classification Results (Breast Cancer)", show_header=True)
    table_bin.add_column("Model Architecture", style="cyan", no_wrap=True)
    table_bin.add_column("Accuracy", justify="right", style="green")
    table_bin.add_column("ROC-AUC", justify="right", style="bold green")
    table_bin.add_column("Fit Time (s)", justify="right")
    table_bin.add_column("Latency (ms/sample)", justify="right")

    for name, acc, auc, fit_t, inf_t in results_bin:
        table_bin.add_row(name, f"{acc:.4f}", f"{auc:.4f}", f"{fit_t:.2f}s", f"{inf_t:.3f}ms")
    console.print(table_bin)
    console.print()

    # -------------------------------------------------------------
    # 2. Multiclass Classification: Wine (3 classes)
    # -------------------------------------------------------------
    console.print(
        "[bold yellow]2. Evaluating Multiclass Classification (Wine Dataset, 3 Classes)...[/bold yellow]"
    )
    wine = load_wine(as_frame=True)
    X_w = wine.data
    y_w = wine.target
    X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(X_w, y_w, test_size=0.25, random_state=42, stratify=y_w)

    models_multi = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Torch-Style MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
        "LightGBM / CatBoost": _get_tree_classifier(random_state=42),
        "CompactTab (Ours)": CompactTabClassifier(
            preset="small", max_epochs=40, mixup_alpha=0.1, random_state=42
        ),
        "CompactTab-Hybrid (Ours)": CompactTabHybridClassifier(
            preset="small", cv_folds=3, max_epochs=35, random_state=42
        ),
    }

    results_multi = []
    for name, model in models_multi.items():
        t0 = time.perf_counter()
        model.fit(X_tr_w, y_tr_w)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = model.predict_proba(X_te_w)
        preds = model.predict(X_te_w)
        inf_t = (time.perf_counter() - t1) * 1000.0 / len(X_te_w)

        acc = accuracy_score(y_te_w, preds)
        auc = roc_auc_score(y_te_w, probs, multi_class="ovr")
        results_multi.append((name, acc, auc, fit_t, inf_t))

    table_multi = Table(title="Multiclass Classification Results (Wine Dataset)", show_header=True)
    table_multi.add_column("Model Architecture", style="cyan", no_wrap=True)
    table_multi.add_column("Accuracy", justify="right", style="green")
    table_multi.add_column("OVR ROC-AUC", justify="right", style="bold green")
    table_multi.add_column("Fit Time (s)", justify="right")
    table_multi.add_column("Latency (ms/sample)", justify="right")

    for name, acc, auc, fit_t, inf_t in results_multi:
        table_multi.add_row(name, f"{acc:.4f}", f"{auc:.4f}", f"{fit_t:.2f}s", f"{inf_t:.3f}ms")
    console.print(table_multi)
    console.print()

    # -------------------------------------------------------------
    # 3. Continuous Regression: Diabetes
    # -------------------------------------------------------------
    console.print("[bold yellow]3. Evaluating Continuous Regression (Diabetes)...[/bold yellow]")
    diab = load_diabetes(as_frame=True)
    X_d = diab.data
    y_d = diab.target
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.25, random_state=42)

    models_reg = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Torch-Style MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42),
        "LightGBM / CatBoost": _get_tree_regressor(random_state=42),
        "CompactTab (Ours)": CompactTabRegressor(
            preset="small", max_epochs=40, mixup_alpha=0.1, random_state=42
        ),
        "CompactTab-Hybrid (Ours)": CompactTabHybridRegressor(
            preset="small", cv_folds=3, max_epochs=35, random_state=42
        ),
    }

    results_reg = []
    for name, model in models_reg.items():
        t0 = time.perf_counter()
        model.fit(X_tr_d, y_tr_d)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_te_d)
        inf_t = (time.perf_counter() - t1) * 1000.0 / len(X_te_d)

        r2 = r2_score(y_te_d, preds)
        rmse = np.sqrt(mean_squared_error(y_te_d, preds))
        results_reg.append((name, r2, rmse, fit_t, inf_t))

    table_reg = Table(title="Continuous Regression Results (Diabetes)", show_header=True)
    table_reg.add_column("Model Architecture", style="cyan", no_wrap=True)
    table_reg.add_column("R2 Score", justify="right", style="bold green")
    table_reg.add_column("RMSE", justify="right", style="green")
    table_reg.add_column("Fit Time (s)", justify="right")
    table_reg.add_column("Latency (ms/sample)", justify="right")

    for name, r2, rmse, fit_t, inf_t in results_reg:
        table_reg.add_row(name, f"{r2:.4f}", f"{rmse:.2f}", f"{fit_t:.2f}s", f"{inf_t:.3f}ms")
    console.print(table_reg)
    console.print("\n[bold green][OK] Benchmark Suite Complete.[/bold green]\n")


if __name__ == "__main__":
    run_benchmark()
