"""Generate every clip in a sounds file as Ogg Vorbis, one subdirectory per group.

Examples:
  build-voice-clips sounds.yaml                      # missing clips, default speaker
  build-voice-clips sounds.yaml --speaker Leo        # another voice, into output/leo/
  build-voice-clips sounds.yaml --overwrite          # regenerate everything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .magpie import Synth, load_config, load_model, resolve_speaker, speaker_name
from .sounds import Clip, load_sounds, unknown_ipa_symbols


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build-voice-clips",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sounds", type=Path, help="YAML file describing the clips to generate")
    p.add_argument("--speaker", help="speaker name or index (default: Sofia)")
    p.add_argument("--output", type=Path, default=Path("output"), help="output root")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing files")
    return p.parse_args(argv)


def check_vocab(clips: list[Clip], model) -> bool:
    """Report IPA symbols the tokenizer would silently drop. Returns True when all are known."""
    vocab = set(model.tokenizer.tokenizers["english_phoneme"].tokens)
    ok = True
    for c in clips:
        bad = unknown_ipa_symbols(c.prompt, vocab)
        if bad:
            print(f"error: {c.name}: IPA symbols not in model vocabulary: {bad}", file=sys.stderr)
            ok = False
    return ok


def write_ogg(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, format="OGG", subtype="VORBIS")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    inf = cfg["inference"]
    speaker_index = resolve_speaker(args.speaker or inf["speaker"])
    out_dir = args.output / speaker_name(speaker_index).lower()

    clips = load_sounds(args.sounds)
    todo = [c for c in clips if args.overwrite or not (out_dir / f"{c.name}.ogg").exists()]
    if not todo:
        print(f"nothing to do: all {len(clips)} files exist in {out_dir} (use --overwrite)")
        return 0

    model = load_model(cfg)
    if not check_vocab(todo, model):
        return 2
    synth = Synth(
        model,
        speaker_index=speaker_index,
        language=inf.get("language", "en"),
        use_cfg=inf.get("use_cfg", True),
        seed=inf.get("seed"),
        model_params=inf.get("model_params"),
    )

    for clip in todo:
        t0 = time.time()
        y = synth.speak(clip.prompt)
        path = out_dir / f"{clip.name}.ogg"
        write_ogg(path, y, synth.sample_rate)
        secs = len(y) / synth.sample_rate
        print(f"{path}  {clip.prompt!r}  {secs:.2f}s  ({time.time() - t0:.1f}s)")
    print(f"generated {len(todo)}, skipped {len(clips) - len(todo)} existing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
