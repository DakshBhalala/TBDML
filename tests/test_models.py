"""Unit tests for CompactTab architecture, layers, and parameter counts."""

import numpy as np
import torch

from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.baselines_nn import TabularMLP, TabularResNet
from compacttab.models.compacttab import (
    CompactTab,
    CompactTabClassifier,
    CompactTabConfig,
    CompactTabRegressor,
)
from compacttab.models.embeddings import CategoricalEmbedding, PiecewiseLinearEncoding
from compacttab.models.hybrid import (
    CompactTabHybridClassifier,
    CompactTabHybridRegressor,
)
from compacttab.models.layers import BatchEnsembleLinear, GatedResidualBlock


def test_piecewise_linear_encoding():
    bin_edges = {
        "f1": np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32),
        "f2": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    }
    ple = PiecewiseLinearEncoding(bin_edges, num_cols=["f1", "f2"], embed_dim=8)

    x = torch.tensor([[5.0, 0.5], [1.0, -0.5]], dtype=torch.float32)
    out = ple(x)  # Expected: (2, 2, 8)
    assert out.shape == (2, 2, 8)
    assert not torch.isnan(out).any()


def test_categorical_embedding():
    cat_cards = {"c1": 5, "c2": 10}
    cat_emb = CategoricalEmbedding(cardinalities=cat_cards, cat_cols=["c1", "c2"], embed_dim=8)

    x_cat = torch.tensor([[0, 3], [4, 9]], dtype=torch.long)
    out = cat_emb(x_cat)
    assert out.shape == (2, 2, 8)
    assert not torch.isnan(out).any()


def test_batch_ensemble_linear():
    layer = BatchEnsembleLinear(in_features=16, out_features=32, num_heads=4)
    x = torch.randn(8, 16)  # (Batch=8, in_dim=16)
    out = layer(x)  # Expected: (k=4, Batch=8, out_dim=32)
    assert out.shape == (4, 8, 32)
    assert not torch.isnan(out).any()


def test_gated_residual_block():
    block = GatedResidualBlock(hidden_dim=32, expansion_factor=2.0, num_heads=4, dropout=0.1)
    x = torch.randn(4, 8, 32)  # (k=4, Batch=8, Dim=32)
    out = block(x)
    assert out.shape == (4, 8, 32)
    assert not torch.isnan(out).any()


def test_compacttab_parameter_counts_and_forward():
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.BINARY,
        numerical_cols=["num_1", "num_2", "num_3", "num_4"],
        categorical_cols=["cat_1", "cat_2"],
        num_classes=2,
    )
    bin_edges = {col: np.linspace(0, 10, 9, dtype=np.float32) for col in schema.numerical_cols}
    cat_cards = {"cat_1": 6, "cat_2": 12}

    # Test Tiny Preset (~100k params target)
    cfg_tiny = CompactTabConfig.from_preset("tiny")
    model_tiny = CompactTab(schema, bin_edges, cat_cards, cfg_tiny)
    params_tiny = model_tiny.count_parameters()
    assert 20_000 < params_tiny < 300_000, f"Expected ~100k params, got {params_tiny}"

    # Test Small Preset (~1M params target)
    cfg_small = CompactTabConfig.from_preset("small")
    model_small = CompactTab(schema, bin_edges, cat_cards, cfg_small)
    params_small = model_small.count_parameters()
    assert 300_000 < params_small < 3_000_000, f"Expected ~1M params, got {params_small}"

    # Forward pass in training mode (returns k heads)
    batch_size = 4
    num_vals = torch.randn(batch_size, 4)
    num_masks = torch.zeros(batch_size, 4)
    cat_indices = torch.randint(0, 5, (batch_size, 2))

    model_tiny.train()
    out_train = model_tiny(num_vals, num_masks, cat_indices)
    assert out_train.shape == (cfg_tiny.num_heads, batch_size, 1)

    # Forward pass in eval mode (returns ensemble-averaged prediction)
    model_tiny.eval()
    out_eval = model_tiny(num_vals, num_masks, cat_indices)
    assert out_eval.shape == (batch_size, 1)
    # Binary classification probabilities should be in [0, 1]
    assert (out_eval >= 0.0).all() and (out_eval <= 1.0).all()


def test_tabular_mlp_and_resnet():
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.REGRESSION,
        numerical_cols=["num_1", "num_2"],
        categorical_cols=["cat_1"],
    )
    cat_cards = {"cat_1": 4}

    mlp = TabularMLP(schema, cat_cards, hidden_dims=[32, 32])
    resnet = TabularResNet(schema, cat_cards, hidden_dim=32, num_blocks=2)

    batch_size = 3
    num_vals = torch.randn(batch_size, 2)
    num_masks = torch.zeros(batch_size, 2)
    cat_indices = torch.tensor([[1], [2], [0]])

    mlp.eval()
    out_mlp = mlp(num_vals, num_masks, cat_indices)
    assert out_mlp.shape == (batch_size, 1)

    resnet.eval()
    out_resnet = resnet(num_vals, num_masks, cat_indices)
    assert out_resnet.shape == (batch_size, 1)


def test_compacttab_classifier_and_regressor():
    import pandas as pd

    # Classification
    rng = np.random.default_rng(42)
    X_cls = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"num_{i}" for i in range(4)])
    X_cls["cat_0"] = rng.choice(["A", "B", "C"], size=50)
    y_cls = pd.Series((X_cls["num_0"] > 0).astype(int))

    clf = CompactTabClassifier(preset="tiny", max_epochs=3, batch_size=32)
    clf.fit(X_cls, y_cls)
    preds = clf.predict(X_cls)
    probs = clf.predict_proba(X_cls)

    assert len(preds) == len(X_cls)
    assert probs.shape == (len(X_cls), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)

    # Regression
    y_reg = pd.Series(X_cls["num_0"] * 2.0 + 1.0)
    reg = CompactTabRegressor(preset="tiny", max_epochs=3, batch_size=32)
    reg.fit(X_cls, y_reg)
    preds_reg = reg.predict(X_cls)

    assert len(preds_reg) == len(X_cls)
    assert not np.isnan(preds_reg).any()


def test_compacttab_hybrid_classifier_and_regressor():
    import pandas as pd

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"feat_{i}" for i in range(4)])
    y_cls = pd.Series((X["feat_0"] > 0).astype(int))

    # Test Hybrid Classifier with 2-fold CV for fast test execution
    hybrid_clf = CompactTabHybridClassifier(preset="tiny", cv_folds=2, max_epochs=2, batch_size=32)
    hybrid_clf.fit(X, y_cls)
    probs = hybrid_clf.predict_proba(X)
    preds = hybrid_clf.predict(X)

    assert probs.shape == (len(X), 2)
    assert len(preds) == len(X)
    assert 0.0 <= hybrid_clf.blend_weight_ <= 1.0

    # Test Hybrid Regressor
    y_reg = pd.Series(X["feat_0"] * 2.0 + X["feat_1"])
    hybrid_reg = CompactTabHybridRegressor(preset="tiny", cv_folds=2, max_epochs=2, batch_size=32)
    hybrid_reg.fit(X, y_reg)
    preds_reg = hybrid_reg.predict(X)

    assert len(preds_reg) == len(X)
    assert not np.isnan(preds_reg).any()
    assert 0.0 <= hybrid_reg.blend_weight_ <= 1.0
