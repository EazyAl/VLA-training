from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from vla_training.data import (
    DatasetSpec,
    EpisodeDataset,
    Normalizer,
    SequenceConfig,
    build_augmentations,
    build_dataloader,
    load_norms,
    load_spec,
)
from vla_training.data.transforms import AugmentConfig
from vla_training.utils import wandb as wandb_utils

from .config import CheckpointConfig, DataConfig, Precision, TrainingConfig

LOG = logging.getLogger(__name__)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def precision_context(device: torch.device, precision: Precision):
    if device.type == "cuda":
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[precision]
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def _maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and hasattr(torch, "compile"):
        LOG.info("Compiling model with torch.compile")
        return torch.compile(model)  # type: ignore[no-any-return]
    return model


def _maybe_enable_gc(model: nn.Module, enabled: bool) -> None:
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        LOG.info("Enabled gradient checkpointing on model")


def _build_model(
    spec: DatasetSpec,
    sample_shapes: Dict[str, Tuple[int, ...]],
    device: torch.device,
    training_cfg: TrainingConfig,
) -> PI05Policy:
    """
    Construct a PI05Policy with input/output feature shapes inferred from a sample.
    """
    if training_cfg.mock_model:
        class DummyPi05(nn.Module):
            def __init__(self, action_dim: int) -> None:
                super().__init__()
                self.linear = nn.Linear(action_dim, action_dim)

            def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
                if "actions" in batch:
                    actions = batch["actions"]
                elif ACTION in batch:
                    actions = batch[ACTION]
                else:
                    raise KeyError("Missing actions in batch for DummyPi05")
                proj = self.linear(actions)
                loss = (proj ** 2).mean()
                return loss, {"loss": float(loss.detach().cpu())}

        return DummyPi05(action_dim=spec.action_dim)  # type: ignore[return-value]

    # Work around LeRobot's transformers.siglip check for "replace" by supplying a stub if missing.
    try:
        import transformers.models.siglip as siglip  # type: ignore

        if not hasattr(siglip, "check"):
            class _Check:
                @staticmethod
                def check_whether_transformers_replace_is_installed_correctly() -> bool:
                    return True

            siglip.check = _Check()  # type: ignore[attr-defined]
    except Exception:
        pass
    
    # Work around LeRobot Pi05 PaliGemma model access issue
    # In newer transformers versions, PaliGemmaForConditionalGeneration API has changed
    # LeRobot 0.4.2 expects: self.paligemma.model.get_image_features(image)
    # But newer transformers: the model IS the object and methods are directly on it
    try:
        from transformers import PaliGemmaForConditionalGeneration
        
        # Add .model property that returns a wrapper (for compatibility with LeRobot 0.4.2)
        # LeRobot expects: self.paligemma.model.get_image_features(image)
        if not hasattr(PaliGemmaForConditionalGeneration, '_vla_model_wrapper_added'):
            class PaliGemmaModelWrapper:
                """Wrapper to provide LeRobot-compatible API for PaliGemma."""
                def __init__(self, paligemma_instance):
                    self._paligemma = paligemma_instance
                    # Try to get processor for proper image preprocessing
                    self._processor = None
                    try:
                        from transformers import AutoProcessor
                        if hasattr(paligemma_instance, 'config') and hasattr(paligemma_instance.config, 'name_or_path'):
                            model_name = paligemma_instance.config.name_or_path
                            try:
                                self._processor = AutoProcessor.from_pretrained(model_name)
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                def _preprocess_image(self, image):
                    """Preprocess image for vision_tower (SigLIP)."""
                    import torch.nn.functional as F
                    
                    # Ensure image is a tensor and on the right device
                    if not isinstance(image, torch.Tensor):
                        image = torch.tensor(image, device=self._paligemma.device)
                    else:
                        image = image.to(self._paligemma.device)
                    
                    # Handle different input shapes
                    if image.dim() == 3:  # (C, H, W) - add batch dim
                        image = image.unsqueeze(0)
                    
                    # Ensure channels-first format (B, C, H, W)
                    if image.dim() == 4 and image.shape[-1] in (1, 3):
                        image = image.permute(0, 3, 1, 2).contiguous()
                    
                    # If image is in [0, 1] range, convert to [0, 255] for resizing
                    if image.max() <= 1.0:
                        image = image * 255.0
                    
                    # Resize to 224x224 (SigLIP base-patch16-224 expects this)
                    if image.shape[-2:] != (224, 224):
                        image = F.interpolate(
                            image,
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Normalize using ImageNet stats (SigLIP expects this)
                    # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
                    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
                    
                    # Convert to float and normalize
                    image = image.to(torch.float32)
                    if image.shape[1] == 3:  # RGB
                        image = (image / 255.0 - mean) / std
                    else:
                        # Grayscale or other - just normalize to [0, 1] range
                        image = image / 255.0
                    
                    return image
                
                def get_image_features(self, image):
                    """Extract image features using PaliGemma vision model."""
                    # Try multiple approaches based on transformers version
                    # Approach 1: Direct method (transformers >= 4.52.3)
                    if hasattr(self._paligemma, 'get_image_features'):
                        return self._paligemma.get_image_features(image)
                    
                    # Approach 2: vision_tower attribute (newer transformers - this is what we have!)
                    if hasattr(self._paligemma, 'vision_tower'):
                        # Preprocess image for SigLIP
                        processed_image = self._preprocess_image(image)
                        
                        # vision_tower expects pixel_values keyword argument
                        vision_outputs = self._paligemma.vision_tower(pixel_values=processed_image)
                        return vision_outputs.last_hidden_state
                    
                    # Approach 3: vision_model attribute (older transformers)
                    if hasattr(self._paligemma, 'vision_model'):
                        processed_image = self._preprocess_image(image)
                        vision_outputs = self._paligemma.vision_model(pixel_values=processed_image)
                        return vision_outputs.last_hidden_state
                    
                    # Approach 4: Check for nested model structure (avoid recursion!)
                    # Use __dict__ to check without triggering property
                    if 'model' in self._paligemma.__dict__:
                        inner_model = self._paligemma.__dict__['model']
                        if hasattr(inner_model, 'get_image_features'):
                            return inner_model.get_image_features(image)
                    
                    # If we get here, log and raise error
                    attrs = [attr for attr in dir(self._paligemma) if not attr.startswith('_') and 'vision' in attr.lower()]
                    LOG.error(
                        "PaliGemma model structure - vision-related attrs: %s",
                        attrs
                    )
                    raise AttributeError(
                        f"PaliGemma model (type: {type(self._paligemma)}) has no vision_tower, "
                        "vision_model, get_image_features, or nested model.get_image_features method. "
                        "Available vision-related attributes: %s" % attrs
                    )
            
            def _model_property(self):
                return PaliGemmaModelWrapper(self)
            
            PaliGemmaForConditionalGeneration.model = property(_model_property)
            PaliGemmaForConditionalGeneration._vla_model_wrapper_added = True
    except (ImportError, AttributeError) as e:
        # If PaliGemma isn't available or patching fails, log and continue
        LOG.warning("Could not patch PaliGemma compatibility: %s", e)
        pass
    action_dim = spec.action_dim

    # Infer state dim from proprio keys if present; fallback to action_dim
    proprio_dims: List[int] = []
    for key, shape in sample_shapes.items():
        if key.startswith("obs.") and len(shape) >= 2 and key not in ("actions", "timestamps"):
            proprio_dims.append(int(np.prod(shape[1:])))
    state_dim = max(sum(proprio_dims), action_dim)

    # Pick first visual key if available
    visual_keys = [k for k, s in sample_shapes.items() if k.startswith("obs.") and len(s) == 4]
    if not visual_keys:
        raise RuntimeError("No visual observation found to feed PI05 image encoder.")
    image_key = visual_keys[0]
    _, img_h, img_w, _ = sample_shapes[image_key]

    input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        max_state_dim=max(state_dim, 32),
        max_action_dim=max(action_dim, 32),
        chunk_size=training_cfg.data.seq_len,
        n_action_steps=min(training_cfg.data.seq_len, training_cfg.data.seq_len),
        image_resolution=(img_h, img_w),
        gradient_checkpointing=training_cfg.grad_checkpointing,
        compile_model=training_cfg.compile_model,
        device=str(device),
        use_amp=training_cfg.precision in ("bf16", "fp16") and device.type == "cuda",
    )
    cfg.validate_features()
    model = PI05Policy(cfg)
    return model


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    config: TrainingConfig,
    spec: DatasetSpec,
    norms_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
            "config": asdict(config),
            "spec": spec.model_dump(),
            "norms_path": str(norms_path),
        },
        path,
    )
    LOG.info("Saved checkpoint at step %s -> %s", step, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    step = int(state.get("step", 0))
    LOG.info("Resumed checkpoint from %s @ step %s", path, step)
    return step


class TrainingRunner:
    def __init__(self, config: TrainingConfig) -> None:
        print("[train] TrainingRunner init begin", flush=True)
        self.config = config
        set_seed(config.seed)
        self.device = _device()
        LOG.info("Using device: %s", self.device)

        self.spec: DatasetSpec = load_spec(config.data.spec_path)
        self.norms_path = config.data.norms_path
        norms = load_norms(self.norms_path)
        self.normalizer = Normalizer(
            action_mean=norms.action_mean,
            action_std=norms.action_std,
            proprio_mean=norms.proprio_mean,
            proprio_std=norms.proprio_std,
        )

        augment = build_augmentations(
            AugmentConfig(
                brightness=config.data.augment_brightness,
                contrast=config.data.augment_contrast,
                seed=config.data.seed,
            )
        )
        self.train_dataset = EpisodeDataset(
            spec=self.spec,
            data_root=config.data.data_root,
            sequence=SequenceConfig(seq_len=config.data.seq_len, stride=config.data.stride),
            normalizer=self.normalizer,
            augment_fn=augment,
            max_episodes=config.data.max_episodes,
            seed=config.data.seed,
        )
        self.train_loader: DataLoader = build_dataloader(
            dataset=self.train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            seed=config.data.seed,
        )

        sample_shapes = {k: v.shape for k, v in self.train_dataset[0].items()}
        LOG.info("Sample shapes: %s", sample_shapes)
        print("[train] Building model...", flush=True)
        self.model = _maybe_compile(
            _build_model(self.spec, sample_shapes, self.device, config).to(self.device),
            config.compile_model,
        )
        print("[train] Model build complete.", flush=True)
        _maybe_enable_gc(self.model, config.grad_checkpointing)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = None
        if config.scheduler.name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.total_steps, eta_min=0
            )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.start_step = 0

        # W&B
        self.wandb_run = None
        if config.wandb:
            self.wandb_run = wandb_utils.init_wandb(
                config=asdict(config),
                project=config.wandb.project,
                entity=config.wandb.entity,
                run_name=config.wandb.run_name,
                tags=config.wandb.tags,
                mode=config.wandb.mode,
                group=config.wandb.group,
            )

        if config.checkpoint.resume:
            self.start_step = _load_checkpoint(
                config.checkpoint.resume, self.model, self.optimizer, self.scheduler, self.scaler
            )

    def _prepare_model_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Adapt dataloader batch to PI05 expected keys."""
        if self.config.mock_model:
            # Minimal batch for dummy model: just actions tensor
            # Ensure actions key present
            if "actions" in batch:
                return {"actions": batch["actions"].to(self.device)}
            if ACTION in batch:
                return {"actions": batch[ACTION].to(self.device)}
            raise KeyError("Batch missing 'actions' for mock model.")

        # Use last valid timestep per sequence
        lengths = batch.get("lengths")
        if lengths is None:
            raise ValueError("Batch missing 'lengths' key for sequence trimming.")
        lengths = lengths.to(self.device)

        def last_step(arr: torch.Tensor) -> torch.Tensor:
            idx = lengths - 1
            out = []
            for i in range(arr.shape[0]):
                out.append(arr[i, idx[i]])
            return torch.stack(out, dim=0)

        # Find visual key from model config
        image_keys = list(self.model.config.image_features.keys())
        if not image_keys:
            raise ValueError("Model config has no image feature keys.")
        image_key = image_keys[0]

        images = last_step(batch[image_key])
        # convert to float and channels-first
        if images.ndim == 4 and images.shape[-1] in (1, 3):
            images = images.permute(0, 3, 1, 2).contiguous()
        images = images.to(torch.float32) / 255.0

        # Build state from proprio keys in spec
        state_parts: List[torch.Tensor] = []
        for key in self.spec.required_proprio_keys:
            obs_key = f"obs.{key}"
            if obs_key not in batch:
                continue
            state_parts.append(last_step(batch[obs_key]).reshape(batch[obs_key].shape[0], -1))
        if state_parts:
            state = torch.cat(state_parts, dim=-1)
        else:
            # Fallback: zeros shaped to action dim
            state = torch.zeros(
                (images.shape[0], self.spec.action_dim), device=self.device, dtype=torch.float32
            )

        # Actions chunk (pad to chunk_size)
        actions = batch["actions"]  # (B, T, A)
        chunk_size = self.model.config.chunk_size
        if actions.shape[1] < chunk_size:
            pad_len = chunk_size - actions.shape[1]
            pad = torch.zeros((actions.shape[0], pad_len, actions.shape[2]), device=self.device)
            actions = torch.cat([actions, pad], dim=1)
        else:
            actions = actions[:, :chunk_size]

        # Dummy language tokens (zeros) and attention mask (ones)
        tokens = torch.zeros(
            (images.shape[0], self.model.config.tokenizer_max_length),
            dtype=torch.long,
            device=self.device,
        )
        attn_mask = torch.ones_like(tokens, dtype=torch.long)

        return {
            image_key: images,
            OBS_STATE: state,
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: attn_mask,
            ACTION: actions,
        }

    def train(self) -> None:
        LOG.info("Starting training for %s steps", self.config.total_steps)
        step = self.start_step
        for batch in self.train_loader:
            if step >= self.config.total_steps:
                break
            LOG.debug("Step %d: fetched batch", step)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            model_batch = self._prepare_model_batch(batch)

            with precision_context(self.device, self.config.precision):
                loss, loss_dict = self.model(model_batch)
            self.scaler.scale(loss).backward()

            if self.config.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler:
                self.scheduler.step()

            step += 1

            if (
                self.config.checkpoint.save_every > 0
                and step % self.config.checkpoint.save_every == 0
            ):
                ckpt_path = (
                    self.config.checkpoint.dir / f"step_{step}.pt"
                )
                _save_checkpoint(
                    ckpt_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    step,
                    self.config,
                    self.spec,
                    self.norms_path,
                )
            LOG.info(
                "step=%d loss=%.4f lr=%.2e",
                step,
                float(loss.detach().cpu()),
                self.optimizer.param_groups[0]["lr"],
            )
            if self.wandb_run:
                wandb_utils.log(
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/step": step,
                    },
                    step=step,
                )
        LOG.info("Training finished at step %s", step)

