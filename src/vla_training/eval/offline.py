from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from vla_training.data import EpisodeDataset, Normalizer, SequenceConfig, build_dataloader, load_norms, load_spec
from vla_training.train.runner import _build_model, _device, precision_context, set_seed
from vla_training.train.config import TrainingConfig, DataConfig, OptimizerConfig, SchedulerConfig, CheckpointConfig


@dataclass
class EvalResult:
    mse: float
    mse_per_dim: List[float]
    delta_magnitude: float
    clipping_fraction: float
    steps: int

    def to_dict(self) -> dict:
        return {
            "mse": self.mse,
            "mse_per_dim": self.mse_per_dim,
            "delta_magnitude": self.delta_magnitude,
            "clipping_fraction": self.clipping_fraction,
            "steps": self.steps,
        }


def _load_checkpoint_model(
    checkpoint_path: Path,
    spec_path: Path,
    norms_path: Path,
    device: torch.device,
    mock_model: bool,
) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    spec = load_spec(spec_path)
    sample_shapes = state.get("sample_shapes")
    if sample_shapes is None:
        raise ValueError("Checkpoint missing sample_shapes; cannot rebuild model.")

    cfg_dict = state.get("config")
    if not cfg_dict:
        raise ValueError("Checkpoint missing config.")
    cfg = TrainingConfig(
        total_steps=cfg_dict["total_steps"],
        data=DataConfig(
            data_root=Path(cfg_dict["data"]["data_root"]),
            spec_path=spec_path,
            norms_path=norms_path,
            seq_len=cfg_dict["data"]["seq_len"],
            stride=cfg_dict["data"]["stride"],
            batch_size=cfg_dict["data"]["batch_size"],
            num_workers=0,
        ),
        optimizer=OptimizerConfig(lr=cfg_dict["optimizer"]["lr"], weight_decay=cfg_dict["optimizer"]["weight_decay"]),
        scheduler=SchedulerConfig(name=cfg_dict["scheduler"]["name"], warmup_steps=cfg_dict["scheduler"]["warmup_steps"]),
        checkpoint=CheckpointConfig(dir=Path(cfg_dict["checkpoint"]["dir"]), save_every=0),
        precision=cfg_dict.get("precision", "bf16"),
        grad_checkpointing=cfg_dict.get("grad_checkpointing", True),
        compile_model=cfg_dict.get("compile_model", False),
        seed=cfg_dict.get("seed"),
        mock_model=mock_model,
    )

    model = _build_model(spec, sample_shapes, device, cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    action_min: Optional[List[float]],
    action_max: Optional[List[float]],
) -> Tuple[float, List[float], float, float]:
    with torch.no_grad():
        diff = preds - targets
        mse_per_dim = diff.pow(2).mean(dim=[0, 1]).cpu().tolist()
        mse = float(np.mean(mse_per_dim))
        deltas = torch.diff(preds, dim=1)
        delta_mag = deltas.abs().mean().item()
        clip_frac = 0.0
        if action_min or action_max:
            below = above = torch.zeros_like(preds)
            if action_min:
                below = preds < torch.tensor(action_min, device=preds.device)
            if action_max:
                above = preds > torch.tensor(action_max, device=preds.device)
            clip_frac = float((below | above).float().mean().item())
    return mse, mse_per_dim, delta_mag, clip_frac


def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    spec_path: Path,
    norms_path: Path,
    max_episodes: int = 2,
    batch_size: int = 1,
    mock_model: bool = False,
) -> EvalResult:
    device = _device()
    set_seed(0)
    spec = load_spec(spec_path)
    norms = load_norms(norms_path)
    normalizer = Normalizer(
        action_mean=norms.action_mean,
        action_std=norms.action_std,
        proprio_mean=norms.proprio_mean,
        proprio_std=norms.proprio_std,
    )
    dataset = EpisodeDataset(
        spec=spec,
        data_root=data_root,
        sequence=SequenceConfig(seq_len=spec.max_episode_steps or 32, stride=spec.max_episode_steps or 32),
        normalizer=normalizer,
        augment_fn=None,
        max_episodes=max_episodes,
        seed=0,
    )
    loader: DataLoader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, seed=0)

    model = _load_checkpoint_model(checkpoint_path, spec_path, norms_path, device, mock_model)

    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    for batch in loader:
        lengths = batch["lengths"]
        T = lengths.max().item()
        # Trim to T and only first action_dim
        actions = batch["actions"][:, :T, : spec.action_dim].to(device)

        if mock_model:
            preds = actions.clone() * 0  # dummy zero prediction
        else:
            # For PI05, use predict_action_chunk; fallback to forward
            with precision_context(device, "fp32"):
                if hasattr(model, "predict_action_chunk"):
                    mb = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    preds = model.predict_action_chunk(
                        {
                            "observation.images.front": mb.get("obs.camera_front", None),
                            "observation.state": mb.get("obs.joint_pos", None),
                            "observation.language.tokens": torch.zeros((actions.shape[0], 1), device=device, dtype=torch.long),
                            "observation.language.attention_mask": torch.ones((actions.shape[0], 1), device=device, dtype=torch.long),
                            "action": actions,
                        }
                    )
                else:
                    preds, _ = model({"actions": actions})
        preds = preds[:, :T, : spec.action_dim]
        preds_list.append(preds.detach().cpu())
        targets_list.append(actions.detach().cpu())

    preds_cat = torch.cat(preds_list, dim=0)
    targets_cat = torch.cat(targets_list, dim=0)

    mse, mse_per_dim, delta_mag, clip_frac = _compute_metrics(
        preds_cat, targets_cat, spec.action_min, spec.action_max
    )
    return EvalResult(mse=mse, mse_per_dim=mse_per_dim, delta_magnitude=delta_mag, clipping_fraction=clip_frac, steps=preds_cat.shape[0])

