# Use Chatterbox TTS

This guide is the day-to-day companion to the [README](../README.md). Install first, using [INSTALL.md](INSTALL.md), and activate the virtual environment in every new terminal.

Only clone a voice you own or have permission to use. The built-in voice shipped with the checkpoints (`conds.pt`) is what you hear when you do not pass a sample.

Output audio is 24,000 Hz, one channel, WAV. Every file includes Resemble AI's Perth watermark. See [Watermark check](#watermark-check).

## Quick start

Save this as `quickstart.py` in the repository folder. It writes `greeting.wav` with the built-in English voice. If you pass a WAV you are allowed to clone, it also writes `cloned.wav`.

```python
"""Generate a greeting, then clone a voice from a short sample you have permission to use."""

import sys
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print("Loading the English Chatterbox model. The first run downloads a few gigabytes.")

model = ChatterboxTTS.from_pretrained(device=device)

text = (
    "Hello, thanks for calling. We are open today from nine until five, "
    "and we are happy to help you find the right service."
)

greeting = model.generate(text)
ta.save("greeting.wav", greeting, model.sr)
print(f"Wrote greeting.wav at {model.sr} Hz, using the built-in voice.")

sample = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("voice-sample.wav")
if not sample.is_file():
    print("No voice sample found, so no clone was written.")
    print("When you have a short WAV of a voice you own or may use, run:")
    print("  python quickstart.py path/to/your-sample.wav")
    sys.exit(0)

cloned = model.generate(text, audio_prompt_path=str(sample))
ta.save("cloned.wav", cloned, model.sr)
print(f"Wrote cloned.wav using the voice in {sample}.")
```

```bash
python quickstart.py
python quickstart.py path/to/your-sample.wav
```

What the script is doing:

1. It picks CUDA, then Apple MPS, then CPU. You can force the CPU by replacing the device block with `device = "cpu"`.
2. `ChatterboxTTS.from_pretrained` downloads `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, and `conds.pt` from the Hugging Face repository `ResembleAI/chatterbox`.
3. `generate(text)` speaks with the built-in voice stored in `conds.pt`.
4. `generate(text, audio_prompt_path=...)` reads your sample and uses it as the voice for that line.

A useful sample is one speaker, little noise, and about 5 to 10 seconds. The English model keeps up to about 10 seconds. Turbo and Nano refuse anything that is not longer than 5 seconds.

On CPU, the 500M English model can take longer to render a sentence than the sentence takes to play. Let the first job finish before you start a second one. A run of this model and of Nano, made on a 4-core CPU with no NVIDIA GPU, is written up in the README under [Recorded on a CPU](../README.md#recorded-on-a-cpu), with the MP3s and the log.

## Make the delivery calmer or more expressive

These knobs apply to the original English model and the multilingual model. They are the controls upstream documents:

| Goal | Settings |
| --- | --- |
| Neutral speech | `exaggeration=0.5`, `cfg_weight=0.5` (the defaults) |
| The reference speaker talks quickly | Lower `cfg_weight` to about `0.3` |
| Dramatic or commercial delivery | `exaggeration` around `0.7` or higher, and `cfg_weight` around `0.3` |
| Language transfer without the reference accent | `cfg_weight=0` |

Higher exaggeration tends to speed the line up. Lowering `cfg_weight` slows it back down. Very high exaggeration can become unstable. The browser app labels 0.5 as neutral and warns that extreme values can be unstable.

```python
wav = model.generate(
    text,
    audio_prompt_path="path/to/your-sample.wav",
    exaggeration=0.7,
    cfg_weight=0.3,
)
ta.save("vo-expressive.wav", wav, model.sr)
```

Turbo and Nano ignore `cfg_weight`, `exaggeration`, and `min_p`. The loader logs a warning when those arguments are set.

## Nano, for a CPU

Nano is the checkpoint upstream aimed at on-device and CPU use. It is English-only. The included example forces CUDA. For a CPU machine, pass `device="cpu"`:

```python
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)

text = "Thanks for holding. Your appointment is confirmed for Thursday at two. [chuckle] We will see you then."
wav = model.generate(text)
ta.save("nano-greeting.wav", wav, model.sr)

# A reference clip must be longer than 5 seconds.
# wav = model.generate(text, audio_prompt_path="path/to/your-sample.wav")
```

`from_pretrained(..., nano=True)` downloads `ResembleAI/chatterbox-nano`. Omit `nano=True` to download `ResembleAI/chatterbox-turbo` instead. Turbo is the 350M English model. Both accept tags in the text. The Turbo browser app offers these tags:

`[clear throat]` `[sigh]` `[shush]` `[cough]` `[groan]` `[sniff]` `[gasp]` `[chuckle]` `[laugh]`

`example_tts_nano.py` and `example_tts_turbo.py` are the same calls with `device="cuda"`.

## Multilingual speech

V3 is the multilingual checkpoint the upstream README recommends. The code defaults to V2 when `t3_model` is omitted, which is what `example_tts.py` does. Ask for V3 by name:

```python
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")

line = "Bonjour. Votre commande est prête. Vous pouvez passer la chercher aujourd'hui."
wav = model.generate(line, language_id="fr")
ta.save("bonjour.wav", wav, model.sr)
```

`language_id` is one of: `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`.

Match the reference clip to that language when you pass `audio_prompt_path`. A clip in another language can carry its accent into the result. Upstream's mitigation is `cfg_weight=0`.

The loader reads `ResembleAI/chatterbox` and selects `t3_mtl23ls_v3.safetensors` for V3. Dedicated single-language model cards are listed in [ORIGINAL.md](ORIGINAL.md#single-language-pack). This repository's loader does not take those repository names as an argument.

## Voice conversion

Voice conversion keeps the words of one recording and speaks them in a target voice. Both files should be recordings you have permission to use.

```python
import torch
import torchaudio as ta
from chatterbox.vc import ChatterboxVC

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio="statement.wav",
    target_voice_path="target-voice.wav",
)
ta.save("converted.wav", wav, model.sr)
```

`example_vc.py` is this example with placeholder filenames. The weights come from `ResembleAI/chatterbox` (`s3gen.safetensors` and `conds.pt`).

## Browser apps

From the repository folder, with the environment active:

| Command | Page |
| --- | --- |
| `python gradio_tts_app.py` | English speech. Sliders for exaggeration, pace, and sampling. |
| `python gradio_tts_turbo_app.py` | Turbo, with the paralinguistic tags as buttons. This file loads Turbo, not Nano. |
| `python gradio_vc_app.py` | Voice conversion. Stays on the local machine. |
| `python multilingual_app.py` | Multilingual page. The checkpoint follows `CHATTERBOX_MULTILINGUAL_T3_MODEL` (`v2` unless you set `v3`). |

`gradio_tts_app.py` and `gradio_tts_turbo_app.py` call Gradio with `share=True`. Starting them asks Gradio for a temporary public URL in addition to the local page. Run them on a computer you trust, do not put client recordings through a public link, and stop the program (Ctrl+C) when you are finished. `multilingual_app.py` starts its server as soon as you run the file.

The English and Turbo pages default to CUDA when it exists, and to CPU otherwise. They do not select MPS.

## Watermark check

Chatterbox runs each waveform through `perth.PerthImplicitWatermarker` before it returns audio. Upstream's description: the mark is imperceptible, survives MP3 compression and ordinary edits, and detects with nearly 100% accuracy.

To read the mark back:

```python
import perth
import librosa

AUDIO_PATH = "greeting.wav"

watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)
watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# 0.0 means no watermark. 1.0 means watermarked.
```

Plan on the mark being present in files this model writes. Removing it is not part of this guide.

## When a run fails

| What you see | What to do |
| --- | --- |
| Download hangs on the first `from_pretrained` | Leave it. The English files are a few gigabytes from `ResembleAI/chatterbox`. Later runs use `~/.cache/huggingface`. |
| `AssertionError` asking you to prepare conditionals or pass `audio_prompt_path` | The checkpoint's `conds.pt` was not loaded, so there is no built-in voice. Pass a sample path. |
| `Audio prompt must be longer than 5 seconds!` | Turbo or Nano received a short clip. Use a longer recording, or switch to `ChatterboxTTS` for a short sample. |
| `Unsupported language_id` | Use one of the codes listed above. |
| Process killed, or CUDA out of memory | Use Nano on CPU. Close other programs. The weight files plus PyTorch need more RAM than the file size alone. |
| Mac error while loading a checkpoint | Run `example_for_mac.py` with `AUDIO_PROMPT_PATH` pointed at a real file. |
| Cloned speech has the wrong accent | Use a reference in the target language, or set `cfg_weight=0` on the English or multilingual model. |
| `No module named chatterbox` | Activate `.venv` and run `pip install -e .` from this repository. |

Further install fixes are in [INSTALL.md](INSTALL.md). Upstream's own tips are preserved in [ORIGINAL.md](ORIGINAL.md#original-chatterbox-tips).
