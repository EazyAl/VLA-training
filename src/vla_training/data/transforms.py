from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class AugmentConfig:
    brightness: float = 0.0  # +/- fraction
    contrast: float = 0.0  # +/- fraction
    seed: Optional[int] = None


def _apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float, rng: np.random.Generator) -> np.ndarray:
    if brightness == 0 and contrast == 0:
        return img
    img_f = img.astype(np.float32)
    if brightness:
        delta = rng.uniform(-brightness, brightness)
        img_f = img_f * (1.0 + delta)
    if contrast:
        delta = rng.uniform(-contrast, contrast)
        mean = img_f.mean()
        img_f = (img_f - mean) * (1.0 + delta) + mean
    return np.clip(img_f, 0.0, 255.0).astype(img.dtype)


def build_augmentations(config: AugmentConfig | None) -> Callable[[Dict[str, np.ndarray], np.random.Generator], Dict[str, np.ndarray]]:
    if config is None or (config.brightness == 0 and config.contrast == 0):
        def noop(batch: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            return batch
        return noop

    def augment(batch: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
        out = dict(batch)
        for key, arr in batch.items():
            if not key.startswith("obs."):
                continue
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):  # (T, H, W, C)
                out[key] = np.stack(
                    [
                        _apply_brightness_contrast(frame, config.brightness, config.contrast, rng)
                        for frame in arr
                    ],
                    axis=0,
                )
        return out

    return augment


@dataclass
class Normalizer:
    action_mean: np.ndarray
    action_std: np.ndarray
    proprio_mean: Dict[str, np.ndarray]
    proprio_std: Dict[str, np.ndarray]
    eps: float = 1e-6

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(sample)
        if "actions" in sample:
            out["actions"] = (sample["actions"] - self.action_mean) / (self.action_std + self.eps)
        for key, mean in self.proprio_mean.items():
            obs_key = f"obs.{key}"
            if obs_key in sample:
                std = self.proprio_std.get(key, np.ones_like(mean))
                out[obs_key] = (sample[obs_key] - mean) / (std + self.eps)
        return out

