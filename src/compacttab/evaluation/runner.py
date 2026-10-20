"""Unified benchmark runner comparing CompactTab against tree and neural baselines."""

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from compacttab.baselines.tree_models import (
    CatBoostBaseline,
    DummyBaseline,
    ExtraTreesBaseline,
    HistGradientBoostingBaseline,
    LightGBMBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from compacttab.data.dataset import TabularDataset, create_dataloader
from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.evaluation.metrics import compute_metrics
from compacttab.evaluation.tracker import ResourceTracker
from compacttab.models.baselines_nn import TabularMLP, TabularResNet
from compacttab.models.compacttab import CompactTab, CompactTabConfig
from compacttab.training.trainer import TabularTrainer


class BenchmarkRunner:
    """Executes standardized evaluation of CompactTab models against strong baselines."""

    def __init__(
        self,
        output_dir: str = "experiments",
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

    def run_dataset_benchmark(
        self,
        dataset_name: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        schema: DatasetSchema,
        model_names: Optional[List[str]] = None,
        max_epochs: int = 40,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """Run all configured models on a dataset and return comparison dataframe."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if model_names is None:
            model_names = [
                "Dummy",
                "RandomForest",
                "HistGradientBoosting",
                "LightGBM",
                "XGBoost",
                "CatBoost",
                "TabularMLP",
                "TabularResNet",
                "CompactTab-Tiny",
                "CompactTab-Small",
            ]

        # 1. Fit preprocessor strictly on training data
        preprocessor = TabularPreprocessor(
            schema=schema, num_bins=16
        )
        preprocessor.fit(df_train, y_train)

        # 2. Transform train, val, and test splits
        num_tr, mask_tr, cat_tr, y_tr_trans = preprocessor.transform(df_train, y_train)
        num_val, mask_val, cat_val, y_val_trans = preprocessor.transform(df_val, y_val)
        num_te, mask_te, cat_te, y_te_trans = preprocessor.transform(df_test, y_test)

        # Flatten features for tree baselines
        X_tr_tree = np.hstack([num_tr, cat_tr]) if cat_tr.shape[1] > 0 else num_tr
        X_val_tree = np.hstack([num_val, cat_val]) if cat_val.shape[1] > 0 else num_val
        X_te_tree = np.hstack([num_te, cat_te]) if cat_te.shape[1] > 0 else num_te

        # PyTorch Datasets
        train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr_trans)
        val_dataset = TabularDataset(num_val, mask_val, cat_val, y_val_trans)
        test_dataset = TabularDataset(num_te, mask_te, cat_te, y_te_trans)

        test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        results = []

        for m_name in model_names:
            ResourceTracker.reset_vram_peak()
            ram_start = ResourceTracker.get_peak_ram_mb()
            fit_start = time.perf_counter()

            model_obj: Any = None
            trainer: Optional[TabularTrainer] = None
            params_count = 0
            model_disk_size_kb = 0.0

            try:
                # -----------------------------
                # 1. Tree & Sanity Baselines
                # -----------------------------
                if m_name == "Dummy":
                    model_obj = DummyBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_train.to_numpy())
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds = model_obj.predict(X_te_tree)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                elif m_name == "RandomForest":
                    model_obj = RandomForestBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_train.to_numpy())
                    fit_time = time.perf_counter() - fit_start
                    params_count = model_obj.count_parameters()

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds = model_obj.predict(X_te_tree)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                elif m_name == "HistGradientBoosting":
                    model_obj = HistGradientBoostingBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_train.to_numpy())
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds = model_obj.predict(X_te_tree)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                elif m_name == "LightGBM":
                    model_obj = LightGBMBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_train.to_numpy())
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds = model_obj.predict(X_te_tree)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                elif m_name == "XGBoost":
                    # Labels for XGBoost classification need to be 0..C-1
                    y_tr_xgb = y_tr_trans if schema.is_classification else y_train.to_numpy()
                    model_obj = XGBoostBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_tr_xgb)
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds_idx = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
                        preds = preprocessor.inverse_transform_target(preds_idx)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                elif m_name == "CatBoost":
                    model_obj = CatBoostBaseline(schema, random_state=self.random_seed)
                    model_obj.fit(X_tr_tree, y_train.to_numpy())
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: model_obj.predict_proba(X_te_tree)
                        if schema.is_classification
                        else model_obj.predict(X_te_tree)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = model_obj.predict_proba(X_te_tree)
                        preds = model_obj.predict(X_te_tree)
                    else:
                        probs = None
                        preds = model_obj.predict(X_te_tree)

                # -----------------------------
                # 2. Neural Baselines
                # -----------------------------
                elif m_name == "TabularMLP":
                    model_obj = TabularMLP(
                        schema=schema,
                        cat_cardinalities=preprocessor.cat_cardinalities_,
                        hidden_dims=[128, 128],
                    )
                    params_count = model_obj.count_parameters()
                    trainer = TabularTrainer(
                        model=model_obj,
                        schema=schema,
                        device=self.device,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                    )
                    trainer.fit(train_dataset, val_dataset)
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: trainer.predict_proba(test_loader)
                        if schema.is_classification
                        else trainer.predict(test_loader)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = trainer.predict_proba(test_loader)
                        preds_idx = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
                        preds = preprocessor.inverse_transform_target(preds_idx)
                    else:
                        probs = None
                        preds_norm = trainer.predict(test_loader)
                        preds = preprocessor.inverse_transform_target(preds_norm)

                elif m_name == "TabularResNet":
                    model_obj = TabularResNet(
                        schema=schema,
                        cat_cardinalities=preprocessor.cat_cardinalities_,
                        hidden_dim=128,
                        num_blocks=3,
                    )
                    params_count = model_obj.count_parameters()
                    trainer = TabularTrainer(
                        model=model_obj,
                        schema=schema,
                        device=self.device,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                    )
                    trainer.fit(train_dataset, val_dataset)
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: trainer.predict_proba(test_loader)
                        if schema.is_classification
                        else trainer.predict(test_loader)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = trainer.predict_proba(test_loader)
                        preds_idx = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
                        preds = preprocessor.inverse_transform_target(preds_idx)
                    else:
                        probs = None
                        preds_norm = trainer.predict(test_loader)
                        preds = preprocessor.inverse_transform_target(preds_norm)

                # -----------------------------
                # 3. CompactTab Variants
                # -----------------------------
                elif m_name.startswith("CompactTab"):
                    preset = "tiny" if "Tiny" in m_name else "small"
                    config = CompactTabConfig.from_preset(preset)
                    model_obj = CompactTab(
                        schema=schema,
                        bin_edges=preprocessor.num_bin_edges_,
                        cat_cardinalities=preprocessor.cat_cardinalities_,
                        config=config,
                    )
                    params_count = model_obj.count_parameters()
                    trainer = TabularTrainer(
                        model=model_obj,
                        schema=schema,
                        device=self.device,
                        learning_rate=config.from_preset(preset).learning_rate if hasattr(config, "learning_rate") else 1e-3,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                    )
                    trainer.fit(train_dataset, val_dataset)
                    fit_time = time.perf_counter() - fit_start

                    predict_fn = (
                        lambda: trainer.predict_proba(test_loader)
                        if schema.is_classification
                        else trainer.predict(test_loader)
                    )
                    lat, tp = ResourceTracker.measure_latency_and_throughput(
                        predict_fn, n_samples=len(df_test)
                    )

                    if schema.is_classification:
                        probs = trainer.predict_proba(test_loader)
                        preds_idx = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
                        preds = preprocessor.inverse_transform_target(preds_idx)
                    else:
                        probs = None
                        preds_norm = trainer.predict(test_loader)
                        preds = preprocessor.inverse_transform_target(preds_norm)

                else:
                    continue

                # Measure serialized model size if neural
                if trainer is not None and model_obj is not None:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        trainer.save_checkpoint(tmp_path)
                        model_disk_size_kb = os.path.getsize(tmp_path) / 1024.0
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                # Compute standard metrics against y_test
                if schema.is_classification:
                    y_eval_true = y_te_trans
                    if probs is not None:
                        y_eval_pred = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
                    else:
                        y_eval_pred = preprocessor.label_encoder_.transform(preds) if preprocessor.label_encoder_ is not None else preds
                else:
                    y_eval_true = y_test.to_numpy()
                    y_eval_pred = preds

                eval_metrics = compute_metrics(
                    y_true=y_eval_true,
                    y_pred=y_eval_pred,
                    y_prob=probs,
                    task_type=schema.task_type,
                )

                peak_ram = ResourceTracker.get_peak_ram_mb()
                peak_vram = ResourceTracker.get_peak_vram_mb()

                res_row = {
                    "dataset": dataset_name,
                    "task_type": schema.task_type.value,
                    "model": m_name,
                    "params": params_count,
                    "disk_size_kb": round(model_disk_size_kb, 1),
                    "fit_time_sec": round(fit_time, 3),
                    "latency_ms_per_sample": round(lat, 4),
                    "throughput_samples_per_sec": round(tp, 1),
                    "peak_ram_mb": round(peak_ram, 1),
                    "peak_vram_mb": round(peak_vram, 1),
                }
                res_row.update({k: round(v, 4) for k, v in eval_metrics.items()})
                results.append(res_row)

            except Exception as e:
                print(f"Error evaluating model {m_name} on {dataset_name}: {e}")
                results.append({
                    "dataset": dataset_name,
                    "task_type": schema.task_type.value,
                    "model": m_name,
                    "error": str(e),
                })

        res_df = pd.DataFrame(results)

        # Save to experiments/
        json_path = os.path.join(self.output_dir, f"{dataset_name}_results.json")
        csv_path = os.path.join(self.output_dir, f"{dataset_name}_results.csv")
        res_df.to_json(json_path, orient="records", indent=2)
        res_df.to_csv(csv_path, index=False)

        return res_df
