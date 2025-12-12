from __future__ import annotations

from pathlib import Path

import torch

from vla_training.data import (
    EpisodeDataset,
    Normalizer,
    SequenceConfig,
    build_augmentations,
    build_dataloader,
    load_norms,
    load_spec,
)


def test_dataloader_shapes(dummy_dataset, dummy_norms: Path) -> None:
    data_root, spec_path, _ = dummy_dataset
    spec = load_spec(spec_path)
    norms = load_norms(dummy_norms)
    normalizer = Normalizer(
        action_mean=norms.action_mean,
        action_std=norms.action_std,
        proprio_mean=norms.proprio_mean,
        proprio_std=norms.proprio_std,
    )
    dataset = EpisodeDataset(
        spec=spec,
        data_root=data_root,
        sequence=SequenceConfig(seq_len=8, stride=8),
        normalizer=normalizer,
        augment_fn=build_augmentations(None),
        seed=123,
    )
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0, seed=42)
    batch = next(iter(loader))
    assert batch["actions"].shape[0] <= 2  # last batch may be smaller
    assert batch["actions"].shape[2] == spec.action_dim
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["lengths"].ndim == 1

