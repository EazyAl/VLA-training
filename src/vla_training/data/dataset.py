from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .spec import DatasetSpec
from .transforms import Normalizer

AugmentFn = Callable[[Dict[str, np.ndarray], np.random.Generator], Dict[str, np.ndarray]]


@dataclass
class SequenceConfig:
    seq_len: Optional[int] = None  # if None, use full episode
    stride: int = 1


def make_worker_init_fn(seed: int | None) -> Callable[[int], None]:
    def _init_fn(worker_id: int) -> None:
        if seed is None:
            return
        np.random.seed(seed + worker_id)

    return _init_fn


def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    pad_shape = (target_len - arr.shape[0], *arr.shape[1:])
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def pad_and_collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    max_len = max(sample["actions"].shape[0] for sample in batch)
    out: Dict[str, List[torch.Tensor]] = {}
    lengths: List[int] = []

    for sample in batch:
        lengths.append(sample["actions"].shape[0])
        for key, arr in sample.items():
            padded = _pad_to_length(arr, max_len)
            out.setdefault(key, []).append(torch.from_numpy(padded))

    collated: Dict[str, torch.Tensor] = {k: torch.stack(v, dim=0) for k, v in out.items()}
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    collated["attention_mask"] = mask
    collated["lengths"] = torch.tensor(lengths, dtype=torch.int32)
    return collated


class EpisodeDataset(Dataset[Dict[str, np.ndarray]]):
    """
    Dataset over episode windows stored as .npz in <root>/episodes.
    Each sample returns a dict of numpy arrays, optionally normalized and augmented.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        data_root: str | Path,
        sequence: SequenceConfig | None = None,
        normalizer: Optional[Normalizer] = None,
        augment_fn: Optional[AugmentFn] = None,
        max_episodes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.spec = spec
        self.data_root = Path(data_root)
        self.sequence = sequence or SequenceConfig()
        self.normalizer = normalizer
        self.augment_fn = augment_fn
        self.seed = seed

        episodes_dir = self.data_root / "episodes"
        paths = sorted(episodes_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No episodes found in {episodes_dir}")
        if max_episodes is not None:
            paths = paths[:max_episodes]

        self._windows: List[Tuple[Path, int, int]] = []
        rng = np.random.default_rng(seed)
        for path in paths:
            episode = dict(np.load(path, allow_pickle=False))
            if "actions" not in episode:
                continue
            T = episode["actions"].shape[0]
            seq_len = self.sequence.seq_len or T
            stride = self.sequence.stride
            if seq_len > T:
                continue
            starts = list(range(0, T - seq_len + 1, stride)) or [0]
            for s in starts:
                self._windows.append((path, s, s + seq_len))
        rng.shuffle(self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def _load_window(self, path: Path, start: int, end: int) -> Dict[str, np.ndarray]:
        ep = dict(np.load(path, allow_pickle=False))
        sample: Dict[str, np.ndarray] = {}
        for key, arr in ep.items():
            if arr.ndim == 0:
                continue
            sample[key] = arr[start:end]
        return sample

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        path, start, end = self._windows[idx]
        sample = self._load_window(path, start, end)

        if self.augment_fn:
            rng = np.random.default_rng(self.seed + idx if self.seed is not None else None)
            sample = self.augment_fn(sample, rng)
        if self.normalizer:
            sample = self.normalizer(sample)
        return sample


def build_dataloader(
    dataset: EpisodeDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    worker_init = make_worker_init_fn(seed)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_and_collate,
        worker_init_fn=worker_init,
        generator=generator,
        pin_memory=False,
    )

