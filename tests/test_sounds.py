from pathlib import Path

import pytest

from build_voice_clips.sounds import ipa_span, load_sounds, unknown_ipa_symbols

EXAMPLE = Path(__file__).resolve().parents[1] / "sounds.yaml"
# The english_phoneme vocabulary of magpie_tts_multilingual_357m v2607 (phoneme subset).
VOCAB = set("abdefhijklmnopstuvwz") | set("æðŋɑɔəɚɛɝɡɪɹʃʊʌʒθˈˌ'")


def test_example_file_loads_with_expected_names():
    clips = load_sounds(EXAMPLE)
    names = [c.name for c in clips]
    assert names[:26] == [f"letters/{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
    assert names[26:46] == [f"numbers/{n:02d}" for n in range(1, 21)]
    assert all(n.startswith("phrases/") for n in names[46:]) and len(names) > 46
    for c in clips:
        assert unknown_ipa_symbols(c.prompt, VOCAB) == [], c.name


def test_text_ipa_and_suffix(tmp_path: Path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "groups:\n"
        "  g:\n    suffix: '..'\n    clips:\n      a: {ipa: 'ˈ b i'}\n      b: {text: bee}\n"
        "  h:\n    clips:\n      c: {text: 'Hi!'}\n"
    )
    by_name = {c.name: c.prompt for c in load_sounds(f)}
    assert by_name == {"g/a": "| ˈ b i |..", "g/b": "bee..", "h/c": "Hi!"}


@pytest.mark.parametrize(
    "body, match",
    [
        ("groups: {}\n", "non-empty 'groups'"),
        ("groups:\n  g:\n    clips: {}\n", "non-empty 'clips'"),
        ("groups:\n  g:\n    clips:\n      a: {text: x, ipa: 'ˈ b i'}\n", "exactly one"),
        ("groups:\n  g:\n    clips:\n      a: {}\n", "exactly one"),
        ("groups:\n  g:\n    clips:\n      'a b': {text: x}\n", "key must be"),
        ("groups:\n  'bad group':\n    clips:\n      a: {text: x}\n", "group name"),
    ],
)
def test_invalid_files_rejected(tmp_path: Path, body: str, match: str):
    f = tmp_path / "s.yaml"
    f.write_text(body)
    with pytest.raises(ValueError, match=match):
        load_sounds(f)


def test_unknown_ipa_symbols_detected():
    assert unknown_ipa_symbols(ipa_span("b iː"), VOCAB) == ["iː"]
    assert unknown_ipa_symbols(ipa_span("w aɪ"), VOCAB) == ["aɪ"]
    assert unknown_ipa_symbols("plain text, no spans", VOCAB) == []
