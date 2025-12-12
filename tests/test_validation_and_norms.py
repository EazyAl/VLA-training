from __future__ import annotations

from pathlib import Path

from vla_training.data import Validator, compute_norms, load_norms, load_spec


def test_validator_passes(dummy_dataset) -> None:
    data_root, spec_path, _ = dummy_dataset
    spec = load_spec(spec_path)
    report = Validator(spec, data_root).validate()
    assert report.episodes_failed == 0
    assert report.episodes_checked > 0


def test_compute_norms_outputs_shapes(dummy_dataset, tmp_path: Path) -> None:
    data_root, spec_path, spec = dummy_dataset
    out_dir = tmp_path / "norms_out"
    norms = compute_norms(spec=spec, dataset_root=data_root, out_dir=out_dir)
    assert (out_dir / "norms.npz").exists()
    assert (out_dir / "metadata.json").exists()
    assert norms.action_mean.shape[0] == spec.action_dim
    assert norms.action_std.shape[0] == spec.action_dim
    loaded = load_norms(out_dir)
    assert loaded.metadata.action_dim == spec.action_dim

