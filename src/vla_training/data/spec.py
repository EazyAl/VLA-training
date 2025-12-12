from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


ActionType = Literal["delta", "absolute"]


class DatasetSpec(BaseModel):
    """Schema describing the expected dataset contract."""

    name: str = Field(..., description="Human-readable dataset name or identifier.")
    version: str = Field("1", description="Spec version to track schema evolution.")
    control_freq_hz: float = Field(..., gt=0, description="Target control frequency (Hz).")
    control_freq_tolerance_pct: float = Field(
        5.0, ge=0, le=100, description="Allowed deviation (%) from target frequency."
    )
    required_obs_keys: List[str] = Field(
        default_factory=list,
        description="Observation keys that must exist for every timestep (e.g., camera_front).",
    )
    required_proprio_keys: List[str] = Field(
        default_factory=list,
        description="Proprioception keys that must exist for every timestep (e.g., joint_pos).",
    )
    action_dim: int = Field(..., gt=0, description="Number of action dimensions (DoF).")
    action_names: Optional[List[str]] = Field(
        None,
        description="Optional per-dimension names (length must equal action_dim if provided).",
    )
    action_type: ActionType = Field(
        "delta", description="Action semantics: delta (e.g., joint delta) or absolute targets."
    )
    action_min: Optional[List[float]] = Field(
        None, description="Optional per-dimension minimum allowed action values."
    )
    action_max: Optional[List[float]] = Field(
        None, description="Optional per-dimension maximum allowed action values."
    )
    max_episode_steps: Optional[int] = Field(
        None, gt=0, description="Optional hard cap on episode length."
    )
    allow_nan: bool = Field(
        False,
        description="If False, NaN/Inf in observations or actions cause validation failures.",
    )
    clip_actions: bool = Field(
        False,
        description="If True, actions outside min/max may be clipped downstream; validation still reports them.",
    )

    @model_validator(mode="after")
    def _check_action_lists(self) -> "DatasetSpec":
        if self.action_names and len(self.action_names) != self.action_dim:
            raise ValueError("action_names must match action_dim length.")
        if self.action_min and len(self.action_min) != self.action_dim:
            raise ValueError("action_min must match action_dim length.")
        if self.action_max and len(self.action_max) != self.action_dim:
            raise ValueError("action_max must match action_dim length.")
        return self


def load_spec(path: str | Path) -> DatasetSpec:
    """Load a DatasetSpec from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(text)
        return DatasetSpec.model_validate(data)
    return DatasetSpec.model_validate_json(text)

