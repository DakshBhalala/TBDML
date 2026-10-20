"""Unit tests for TabularTrainer, early stopping, and model checkpoint serialization."""

import os
import tempfile
import numpy as np
import pytest
import torch

from compacttab.data.loaders import generate_synthetic_tabular_dataset
from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.dataset import TabularDataset
from compacttab.models.compacttab import CompactTab, CompactTabConfig
from compacttab.training.trainer import TabularTrainer


def test_compacttab_training_and_checkpoint_reload():
    # 1. Generate synthetic dataset
    df_tr, df_val, df_te, y_tr, y_val, y_te, schema = generate_synthetic_tabular_dataset(
        n_samples=200, n_num=4, n_cat=1, random_state=42
    )

    # 2. Preprocessing
    preprocessor = TabularPreprocessor(schema=schema, num_bins=8)
    preprocessor.fit(df_tr, y_tr)

    num_tr, mask_tr, cat_tr, y_tr_trans = preprocessor.transform(df_tr, y_tr)
    num_val, mask_val, cat_val, y_val_trans = preprocessor.transform(df_val, y_val)
    num_te, mask_te, cat_te, y_te_trans = preprocessor.transform(df_te, y_te)

    train_ds = TabularDataset(num_tr, mask_tr, cat_tr, y_tr_trans)
    val_ds = TabularDataset(num_val, mask_val, cat_val, y_val_trans)
    test_ds = TabularDataset(num_te, mask_te, cat_te, y_te_trans)

    # 3. Model setup
    config = CompactTabConfig.from_preset("tiny")
    model = CompactTab(
        schema=schema,
        bin_edges=preprocessor.num_bin_edges_,
        cat_cardinalities=preprocessor.cat_cardinalities_,
        config=config,
    )

    # 4. Train
    trainer = TabularTrainer(
        model=model,
        schema=schema,
        max_epochs=5,
        patience=3,
        batch_size=32,
        learning_rate=1e-3,
    )
    trainer.fit(train_ds, val_ds)

    assert len(trainer.history["train_loss"]) > 0
    assert len(trainer.history["val_loss"]) > 0

    from compacttab.data.dataset import create_dataloader
    test_loader = create_dataloader(test_ds, batch_size=32, shuffle=False)
    preds_before = trainer.predict_proba(test_loader)
    assert preds_before.shape == (len(df_te), 2)

    # 5. Serialization and reload test
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 0

        # Create fresh model instance and reload
        model_new = CompactTab(
            schema=schema,
            bin_edges=preprocessor.num_bin_edges_,
            cat_cardinalities=preprocessor.cat_cardinalities_,
            config=config,
        )
        trainer_new = TabularTrainer(
            model=model_new,
            schema=schema,
            max_epochs=1,
            batch_size=32,
        )
        trainer_new.load_checkpoint(ckpt_path)

        preds_after = trainer_new.predict_proba(test_loader)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)

    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
