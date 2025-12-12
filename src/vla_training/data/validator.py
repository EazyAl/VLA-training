from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .spec import DatasetSpec


def _episode_name(path: Path) -> str:
    return path.stem


@dataclass
class EpisodeIssues:
    episode: str
    issues: List[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.issues.append(message)

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass
class ValidationReport:
    spec: DatasetSpec
    dataset_root: Path
    episodes_checked: int
    episodes_failed: int
    episode_issues: List[EpisodeIssues] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "spec": self.spec.model_dump(),
            "dataset_root": str(self.dataset_root),
            "episodes_checked": self.episodes_checked,
            "episodes_failed": self.episodes_failed,
            "failures": [
                {"episode": e.episode, "issues": e.issues}
                for e in self.episode_issues
                if e.issues
            ],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class Validator:
    """Validate episodes stored as .npz files in <dataset_root>/episodes."""

    def __init__(self, spec: DatasetSpec, dataset_root: str | Path) -> None:
        self.spec = spec
        self.dataset_root = Path(dataset_root)

    def _load_episode(self, path: Path) -> Dict[str, np.ndarray]:
        return dict(np.load(path, allow_pickle=False))

    def _check_shapes(
        self,
        episode: Dict[str, np.ndarray],
        ep_name: str,
        issues: EpisodeIssues,
    ) -> int:
        if "actions" not in episode:
            issues.add("Missing 'actions' array.")
            return 0
        actions = episode["actions"]
        if actions.ndim != 2:
            issues.add(f"'actions' must be 2D (T, action_dim); got {actions.shape}.")
            return 0
        if actions.shape[1] != self.spec.action_dim:
            issues.add(
                f"'actions' second dim mismatch: expected {self.spec.action_dim}, got {actions.shape[1]}."
            )
        timesteps = actions.shape[0]

        def expect_key(key: str) -> None:
            if key not in episode:
                issues.add(f"Missing observation key '{key}'.")
                return
            arr = episode[key]
            if arr.shape[0] != timesteps:
                issues.add(
                    f"Observation '{key}' length mismatch: expected {timesteps}, got {arr.shape[0]}."
                )

        for key in self.spec.required_obs_keys:
            expect_key(f"obs.{key}")
        for key in self.spec.required_proprio_keys:
            expect_key(f"obs.{key}")

        if self.spec.max_episode_steps and timesteps > self.spec.max_episode_steps:
            issues.add(
                f"Episode length {timesteps} exceeds max_episode_steps {self.spec.max_episode_steps}."
            )

        return timesteps

    def _check_nan_inf(self, arrays: Sequence[np.ndarray], issues: EpisodeIssues) -> None:
        for arr in arrays:
            if np.isnan(arr).any() or np.isinf(arr).any():
                issues.add("Found NaN or Inf values in episode.")
                return

    def _check_action_range(self, actions: np.ndarray, issues: EpisodeIssues) -> None:
        if self.spec.action_min:
            below = (actions < np.array(self.spec.action_min)).any()
            if below:
                issues.add("Actions below specified action_min.")
        if self.spec.action_max:
            above = (actions > np.array(self.spec.action_max)).any()
            if above:
                issues.add("Actions above specified action_max.")

    def _check_frequency(self, timestamps: np.ndarray, issues: EpisodeIssues) -> None:
        if timestamps.ndim != 1:
            issues.add(f"'timestamps' must be 1D; got shape {timestamps.shape}.")
            return
        if len(timestamps) < 2:
            return
        deltas = np.diff(timestamps)
        if np.any(deltas <= 0):
            issues.add("Timestamps are not strictly increasing.")
            return
        median_dt = float(np.median(deltas))
        actual_hz = 1.0 / median_dt if median_dt > 0 else 0.0
        target = self.spec.control_freq_hz
        tolerance = target * (self.spec.control_freq_tolerance_pct / 100.0)
        if abs(actual_hz - target) > tolerance:
            issues.add(
                f"Control frequency out of tolerance: actual {actual_hz:.2f}Hz, target {target}Hz, tol ±{tolerance:.2f}."
            )

    def validate(self, max_episodes: int | None = None) -> ValidationReport:
        episodes_dir = self.dataset_root / "episodes"
        paths = sorted(episodes_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No episodes found in {episodes_dir}")
        if max_episodes is not None:
            paths = paths[:max_episodes]

        issues_list: List[EpisodeIssues] = []
        failures = 0

        for path in paths:
            issues = EpisodeIssues(_episode_name(path))
            episode = self._load_episode(path)

            timesteps = self._check_shapes(episode, issues.episode, issues)
            if timesteps == 0:
                failures += 1
                issues_list.append(issues)
                continue

            actions = episode["actions"]
            obs_arrays = [
                episode[key]
                for key in episode.keys()
                if key.startswith("obs.") and episode[key].shape[0] == timesteps
            ]

            if not self.spec.allow_nan:
                self._check_nan_inf([actions, *obs_arrays], issues)

            self._check_action_range(actions, issues)

            if "timestamps" in episode:
                self._check_frequency(episode["timestamps"], issues)
            else:
                issues.add("Missing 'timestamps' array for frequency check.")

            if issues.issues:
                failures += 1
            issues_list.append(issues)

        return ValidationReport(
            spec=self.spec,
            dataset_root=self.dataset_root,
            episodes_checked=len(paths),
            episodes_failed=failures,
            episode_issues=issues_list,
        )

