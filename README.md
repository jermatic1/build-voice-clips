# build-voice-clips

Builds Ogg Vorbis voice clips from a YAML file with NVIDIA MagpieTTS, running
locally on your GPU. Clips are grouped, and every group becomes an output
directory with one file per clip, all in a single voice.

## Requirements

- Linux with an NVIDIA GPU and a current driver (CPU works, but slowly)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- About 10 GB of disk for the Python environment plus 1.5 GB for the model

The model checkpoint downloads into the Hugging Face cache on first run.

## Usage

Run it straight from GitHub without installing anything:

```bash
uvx --from git+https://github.com/jermatic1/build-voice-clips build-voice-clips sounds.yaml
```

Or from a clone of this repository:

```bash
uv sync
uv run build-voice-clips sounds.yaml                  # clips that are missing, default voice
uv run build-voice-clips sounds.yaml --speaker Leo    # another voice (Aria, Jason, John, Leo, Sofia)
uv run build-voice-clips sounds.yaml --overwrite      # rebuild everything
uv run build-voice-clips sounds.yaml --output DIR     # somewhere other than ./output
```

Files land under the speaker's name, then the group, then the clip key:

```
output/sofia/letters/a.ogg
output/sofia/numbers/01.ogg
output/sofia/phrases/welcome.ogg
```

## The sounds file

`sounds.yaml` in this repository is a complete example: the 26 English letter
names, the numbers 1 to 20 and a set of short phrases. The shape is:

```yaml
groups:
  letters:
    suffix: ".."              # optional, appended to every prompt in the group
    clips:
      a: {ipa: "ˈ ɛ ɪ"}       # spoken from IPA symbols
  phrases:
    clips:
      welcome: {text: "Welcome!"}   # spoken from plain text
```

Keys may contain letters, digits, `_` and `-`. Quote numeric keys such as
`"01"` so YAML keeps the leading zero. The header comment in `sounds.yaml`
explains how to write IPA for this model.

## Model

[`nvidia/magpie_tts_multilingual_357m`](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
under the [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
The exact revision, checksum, default speaker and sampling settings are in
`build_voice_clips/audio.yaml`.
