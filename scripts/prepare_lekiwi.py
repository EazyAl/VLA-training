#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

"""
Prepares the PRFitz/lekiwi-dataset-pick-place-red123 dataset into .npz episodes.

What it does:
- Downloads parquet chunks (no videos) from HF.
- Filters to a limited number of episodes if requested.
- Extracts arm-only actions/proprio (first 6 dims), timestamps, and builds a dummy image
  placeholder (zeros) to satisfy the visual key expected downstream.
- Writes to <out_dir>/episodes/ep_<idx>.npz
"""


def load_parquet_files(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}/data")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def to_episodes(df: pd.DataFrame, max_episodes: int | None) -> List[pd.DataFrame]:
    episodes = []
    for _, ep_df in df.groupby("episode_index", sort=True):
        ep_df = ep_df.sort_values("frame_index")
        episodes.append(ep_df)
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes


def prepare_episode(ep_df: pd.DataFrame, image_shape=(64, 64, 3)) -> dict:
    actions_full = np.stack(ep_df["action"].to_list()).astype(np.float32)
    actions = actions_full[:, :6]  # arm + gripper

    state_full = np.stack(ep_df["observation.state"].to_list()).astype(np.float32)
    joint_pos = state_full[:, :6]

    timestamps = ep_df["timestamp"].to_numpy(dtype=np.float32)

    T = actions.shape[0]
    images = np.zeros((T, *image_shape), dtype=np.uint8)  # placeholder visual stream

    return {
        "actions": actions,
        "obs.joint_pos": joint_pos,
        "obs.camera_front": images,
        "timestamps": timestamps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/lekiwi_pickplace"))
    parser.add_argument("--repo-id", default="PRFitz/lekiwi-dataset-pick-place-red123")
    parser.add_argument("--max-episodes", type=int, default=5, help="Limit episodes for quick prep.")
    parser.add_argument("--allow-patterns", nargs="*", default=["data/**", "meta/**"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading parquet chunks...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.out_dir,
        allow_patterns=args.allow_patterns,
    )

    df = load_parquet_files(args.out_dir)
    episodes = to_episodes(df, args.max_episodes)
    ep_dir = args.out_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    for i, ep_df in enumerate(episodes):
        ep = prepare_episode(ep_df)
        np.savez(ep_dir / f"ep_{i}.npz", **ep)
    meta = {"episodes": len(episodes), "source": args.repo_id, "max_episodes": args.max_episodes}
    (args.out_dir / "prepared_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(episodes)} episodes to {ep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

