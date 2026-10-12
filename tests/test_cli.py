from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from build_voice_clips.cli import parse_args, write_ogg
from build_voice_clips.magpie import load_config, resolve_speaker, speaker_name


def test_write_ogg_vorbis(tmp_path: Path):
    sr = 22050
    y = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, sr // 10, dtype=np.float32))
    path = tmp_path / "letters" / "a.ogg"
    write_ogg(path, y, sr)
    info = sf.info(path)
    assert (info.format, info.subtype, info.samplerate, info.channels) == ("OGG", "VORBIS", sr, 1)


def test_parse_args_defaults():
    args = parse_args(["sounds.yaml"])
    assert (args.sounds, args.output, args.overwrite, args.speaker) == (
        Path("sounds.yaml"),
        Path("output"),
        False,
        None,
    )


def test_packaged_config_loads():
    cfg = load_config()
    assert cfg["model"]["repo_id"].startswith("nvidia/") and "speaker" in cfg["inference"]


def test_speaker_round_trip():
    assert speaker_name(resolve_speaker("sofia")) == "Sofia"
    assert speaker_name(resolve_speaker("3")) == "Leo"
    with pytest.raises(ValueError):
        resolve_speaker("Nobody")
