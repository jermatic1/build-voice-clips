"""Loads a sounds file: named groups of clips, each spoken from plain text or an IPA span."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

IPA_SPAN_RE = re.compile(r"\|([^|]*)\|")
KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Clip:
    group: str  # output subdirectory
    key: str  # output file stem
    prompt: str  # exact text passed to the model

    @property
    def name(self) -> str:
        return f"{self.group}/{self.key}"


def ipa_span(ipa: str) -> str:
    """Wrap space-separated IPA symbols in Magpie's `| ... |` override markers."""
    return f"| {' '.join(ipa.split())} |"


def unknown_ipa_symbols(prompt: str, vocab: set[str]) -> list[str]:
    """Return IPA symbols inside `| ... |` spans that the tokenizer would silently drop."""
    bad = []
    for span in IPA_SPAN_RE.findall(prompt):
        bad.extend(sym for sym in span.split() if sym not in vocab)
    return bad


def load_sounds(path: Path | str) -> list[Clip]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    groups = cfg.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"{path}: expected a non-empty 'groups' mapping")

    clips = []
    for group, spec in groups.items():
        group = str(group)
        if not KEY_RE.fullmatch(group):
            raise ValueError(f"group name {group!r} must be letters, digits, '_' or '-'")
        spec = spec or {}
        suffix = str(spec.get("suffix", ""))
        entries = spec.get("clips")
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"group {group!r}: expected a non-empty 'clips' mapping")
        for key, entry in entries.items():
            clips.append(_clip(group, str(key), entry or {}, suffix))
    return clips


def _clip(group: str, key: str, entry: dict, suffix: str) -> Clip:
    where = f"{group}/{key}"
    if not KEY_RE.fullmatch(key):
        raise ValueError(
            f"{where}: key must be letters, digits, '_' or '-' (quote numbers like '01')"
        )
    text, ipa = entry.get("text"), entry.get("ipa")
    if (text is None) == (ipa is None):
        raise ValueError(f"{where}: give exactly one of 'text' or 'ipa'")
    body = ipa_span(ipa) if ipa is not None else str(text)
    return Clip(group, key, body + suffix)
