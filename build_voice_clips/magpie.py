"""Thin wrapper around NVIDIA MagpieTTS: pinned checkpoint download, loading and synthesis."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "audio.yaml"
log = logging.getLogger(__name__)

# Speaker names published in the v2607 model card. The index count is verified
# against the loaded checkpoint in load_model().
SPEAKERS = {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_checkpoint(model_cfg: dict) -> Path:
    """Download (or reuse) the pinned .nemo file and verify its checksum."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=model_cfg["repo_id"],
            filename=model_cfg["filename"],
            revision=model_cfg["revision"],
        )
    )
    expected = model_cfg.get("sha256")
    if expected:
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint checksum mismatch for {path}: expected {expected}, got {actual}"
            )
    return path


def load_model(config: dict | None = None, device: str | None = None):
    """Download the pinned checkpoint if needed and load MagpieTTS once."""
    from nemo.collections.tts.models import MagpieTTSModel

    config = config or load_config()
    path = fetch_checkpoint(config["model"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    model = MagpieTTSModel.restore_from(str(path), map_location=device)
    model.eval()
    # Training uses a 0.8 phoneme probability so some words are randomly left as
    # graphemes. Force full phonemization for deterministic inference.
    for tok in model.tokenizer.tokenizers.values():
        g2p = getattr(tok, "g2p", None)
        if g2p is not None and hasattr(g2p, "phoneme_probability"):
            g2p.phoneme_probability = 1.0
    if model.num_baked_speakers != len(SPEAKERS):
        log.warning(
            "Checkpoint has %d baked speakers but SPEAKERS lists %d",
            model.num_baked_speakers,
            len(SPEAKERS),
        )
    return model


@dataclass
class Synth:
    """Holds a loaded model plus the fixed inference settings shared by every asset."""

    model: object
    speaker_index: int
    language: str = "en"
    use_cfg: bool = True
    seed: int | None = None
    # Overrides for the checkpoint's ModelInferenceParameters (temperature, topk,
    # cfg_scale, attention_sink_threshold, ...). Applied once, for every asset.
    model_params: dict | None = None

    def __post_init__(self):
        # NeMo splits these across two dataclasses; accept keys from either.
        targets = [
            self.model.inference_parameters,
            getattr(self.model, "chunked_inference_config", None),
        ]
        for key, value in (self.model_params or {}).items():
            target = next((t for t in targets if t is not None and hasattr(t, key)), None)
            if target is None:
                raise ValueError(f"unknown inference parameter {key!r}")
            setattr(target, key, type(getattr(target, key))(value))

    @property
    def sample_rate(self) -> int:
        return int(self.model.output_sample_rate)

    def speak(self, text: str) -> np.ndarray:
        """Synthesize `text` and return a float32 mono waveform at the model's native rate."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.inference_mode():
            audio, audio_len = self.model.do_tts(
                text,
                language=self.language,
                apply_TN=False,
                use_cfg=self.use_cfg,
                speaker_index=self.speaker_index,
            )
        n = int(audio_len[0])
        return audio[0, :n].float().cpu().numpy()


def resolve_speaker(speaker: str | int) -> int:
    """Accept a speaker name (any case) or index and return the index."""
    if isinstance(speaker, int) or str(speaker).isdigit():
        index = int(speaker)
        if index not in SPEAKERS.values():
            raise ValueError(f"Unknown speaker index {index}; choose from {list(SPEAKERS)}")
        return index
    for name, index in SPEAKERS.items():
        if name.lower() == str(speaker).lower():
            return index
    raise ValueError(f"Unknown speaker {speaker!r}; choose from {list(SPEAKERS)} or an index")


def speaker_name(index: int) -> str:
    return next(name for name, i in SPEAKERS.items() if i == index)
