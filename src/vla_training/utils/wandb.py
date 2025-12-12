from __future__ import annotations

import os
from typing import Any, Dict, Optional

import wandb

def init_wandb(
    config: Dict[str, Any],
    project: str,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    mode: str = "disabled",
    group: Optional[str] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    if mode == "disabled":
        return None
    os.environ.setdefault("WANDB_SILENT", "true")
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        mode=mode,
        config=config,
        group=group,
    )


def log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb.run is None:
        return
    wandb.log(data, step=step)


def save_artifact(path: str, name: str, type_: str = "artifact") -> None:
    if wandb.run is None:
        return
    art = wandb.Artifact(name=name, type=type_)
    art.add_file(path)
    wandb.log_artifact(art)

