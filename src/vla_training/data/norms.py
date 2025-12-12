from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .spec import DatasetSpec


@dataclass
class NormsMetadata:
    dataset_name: str
    dataset_version: str
    spec_version: str
    action_dim: int
    computed_from: List[str]  # episode names

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "NormsMetadata":
        return cls(**json.loads(text))


@dataclass
class Norms:
    action_mean: np.ndarray
    action_std: np.ndarray
    proprio_mean: Dict[str, np.ndarray]
    proprio_std: Dict[str, np.ndarray]
    metadata: NormsMetadata

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "norms.npz",
            action_mean=self.action_mean,
            action_std=self.action_std,
            **{f"proprio_mean_{k}": v for k, v in self.proprio_mean.items()},
            **{f"proprio_std_{k}": v for k, v in self.proprio_std.items()},
        )
        (out_dir / "metadata.json").write_text(self.metadata.to_json())


def load_norms(path: Path) -> Norms:
    path = path if path.is_dir() else path.parent
    arrays = dict(np.load(path / "norms.npz", allow_pickle=False))
    action_mean = arrays.pop("action_mean")
    action_std = arrays.pop("action_std")
    proprio_mean = {}
    proprio_std = {}
    for key, arr in list(arrays.items()):
        if key.startswith("proprio_mean_"):
            proprio_mean[key.removeprefix("proprio_mean_")] = arr
        if key.startswith("proprio_std_"):
            proprio_std[key.removeprefix("proprio_std_")] = arr
    metadata = NormsMetadata.from_json((path / "metadata.json").read_text())
    return Norms(
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        metadata=metadata,
    )


def _running_mean_std(
    arrays: Iterable[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    count = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    for arr in arrays:
        if arr.size == 0:
            continue
        if mean is None:
            mean = np.zeros_like(arr[0], dtype=np.float64)
            m2 = np.zeros_like(arr[0], dtype=np.float64)
        for row in arr:
            count += 1
            delta = row - mean
            mean = mean + delta / count
            m2 = m2 + delta * (row - mean)
    if count < 2 or mean is None or m2 is None:
        raise ValueError("Insufficient data to compute statistics.")
    variance = m2 / (count - 1)
    std = np.sqrt(np.clip(variance, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


def _iter_actions_and_proprio(
    episodes: Sequence[Mapping[str, np.ndarray]],
    proprio_keys: Sequence[str],
) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
    actions_list: List[np.ndarray] = []
    proprio_lists: Dict[str, List[np.ndarray]] = {k: [] for k in proprio_keys}

    for ep in episodes:
        if "actions" not in ep:
            raise ValueError("Episode missing 'actions'.")
        actions = ep["actions"]
        if actions.ndim != 2:
            raise ValueError(f"'actions' must be 2D; got {actions.shape}.")
        actions_list.append(actions.astype(np.float32))
        for key in proprio_keys:
            obs_key = f"obs.{key}"
            if obs_key not in ep:
                raise ValueError(f"Episode missing proprio key '{obs_key}'.")
            arr = ep[obs_key]
            if arr.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"Proprio '{obs_key}' length mismatch vs actions: {arr.shape[0]} vs {actions.shape[0]}"
                )
            proprio_lists[key].append(arr.astype(np.float32))
    return actions_list, proprio_lists


def _load_episodes(episode_paths: Sequence[Path]) -> List[Dict[str, np.ndarray]]:
    return [dict(np.load(p, allow_pickle=False)) for p in episode_paths]


def compute_norms(
    spec: DatasetSpec,
    dataset_root: Path,
    out_dir: Path,
    max_episodes: int | None = None,
    proprio_keys: Optional[Sequence[str]] = None,
) -> Norms:
    episodes_dir = dataset_root / "episodes"
    paths = sorted(episodes_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episodes found in {episodes_dir}")
    if max_episodes is not None:
        paths = paths[:max_episodes]

    episodes = _load_episodes(paths)
    actions_list, proprio_lists = _iter_actions_and_proprio(
        episodes=episodes, proprio_keys=proprio_keys or []
    )
    action_mean, action_std = _running_mean_std(actions_list)

    proprio_mean: Dict[str, np.ndarray] = {}
    proprio_std: Dict[str, np.ndarray] = {}
    for key, arrs in proprio_lists.items():
        if arrs:
            m, s = _running_mean_std(arrs)
            proprio_mean[key] = m
            proprio_std[key] = s

    metadata = NormsMetadata(
        dataset_name=spec.name,
        dataset_version=spec.version,
        spec_version=spec.version,
        action_dim=spec.action_dim,
        computed_from=[p.stem for p in paths],
    )

    norms = Norms(
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        metadata=metadata,
    )
    norms.save(out_dir)
    return norms

