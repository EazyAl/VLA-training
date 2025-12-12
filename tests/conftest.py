from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import yaml

from vla_training.data.spec import DatasetSpec
from vla_training.data.norms import compute_norms


def _make_episode(path: Path, action_dim: int, timesteps: int) -> None:
    actions = np.random.uniform(-1, 1, size=(timesteps, action_dim)).astype(np.float32)
    timestamps = np.cumsum(np.full((timesteps,), 0.05, dtype=np.float32))  # 20 Hz
    obs = {
        "obs.joint_pos": np.random.uniform(-1, 1, size=(timesteps, action_dim)).astype(np.float32),
        "obs.camera_front": np.random.randint(0, 255, size=(timesteps, 8, 8, 3), dtype=np.uint8),
    }
    np.savez(path, actions=actions, timestamps=timestamps, **obs)


def _write_spec(path: Path, action_dim: int) -> DatasetSpec:
    spec = DatasetSpec(
        name="dummy",
        version="1",
        control_freq_hz=20.0,
        control_freq_tolerance_pct=10.0,
        required_obs_keys=["camera_front"],
        required_proprio_keys=["joint_pos"],
        action_dim=action_dim,
        action_names=[f"j{i}" for i in range(action_dim)],
        action_type="delta",
        action_min=[-2.0] * action_dim,
        action_max=[2.0] * action_dim,
        max_episode_steps=256,
        allow_nan=False,
        clip_actions=False,
    )
    path.write_text(yaml.safe_dump(spec.model_dump()))
    return spec


@pytest.fixture()
def dummy_dataset(tmp_path: Path) -> Tuple[Path, Path, DatasetSpec]:
    data_root = tmp_path / "data"
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    action_dim = 7
    for i in range(3):
        _make_episode(episodes_dir / f"ep_{i}.npz", action_dim=action_dim, timesteps=20 + i * 5)

    spec_path = tmp_path / "spec.yaml"
    spec = _write_spec(spec_path, action_dim=action_dim)
    return data_root, spec_path, spec


@pytest.fixture()
def dummy_norms(tmp_path: Path, dummy_dataset: Tuple[Path, Path, DatasetSpec]) -> Path:
    data_root, spec_path, spec = dummy_dataset
    out_dir = tmp_path / "norms"
    compute_norms(
        spec=spec,
        dataset_root=data_root,
        out_dir=out_dir,
        max_episodes=None,
        proprio_keys=spec.required_proprio_keys,
    )
    return out_dir

