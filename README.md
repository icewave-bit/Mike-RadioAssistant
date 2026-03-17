# RadioBuddy – VHF Radio ↔ Voice AI Pipeline

RadioBuddy turns a computer with a USB audio dongle into a VOX‑driven AI assistant over VHF radio (including Russian STT/TTS):

- **Audio in** from radio → **Whisper STT**
- **GPT‑5 Nano** for AI replies
- **TTS** for speech (macOS built‑in `say`, or **local Piper** on Linux)
- **Audio out** back to radio via the same dongle

All configuration is driven by environment variables (via `.env`).

## Prerequisites

- Python 3.10+ (recommended)
- USB audio dongle wired between radio and Mac (speaker out → Mac/dongle in, Mac/dongle out → radio mic/line in)
- `ffmpeg` installed (optional, recommended)
- Linux: PortAudio (`sounddevice` dependency). On Mint/Ubuntu:

```bash
sudo apt update
sudo apt install -y portaudio19-dev
```

## Setup (with `mise` + `uv` or plain `pip`)

```bash
cd RadioBuddy

# Option 1: use mise + uv (recommended)
mise install           # creates .venv using uv and installs tools from mise.toml
mise run start         # runs python run.py

# Option 2: plain venv + pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

Fill in your API keys and adjust audio/VOX settings as needed.

### Linux (Mint) TTS setup (Piper, local/offline, including Russian)

On Linux, RadioBuddy uses **Piper** for local TTS. You need to install Piper and provide a voice model path.

Install the Python package:

```bash
python -m pip install piper-tts
```

Download a voice model (example English: `en_US-lessac-medium`, example Russian: `ru_RU-irina-medium`):

```bash
python -m piper.download_voices en_US-lessac-medium
```

Set `PIPER_MODEL_PATH` in your `.env` to the downloaded `.onnx` model file path, for example:

```bash
# English
PIPER_MODEL_PATH=/absolute/path/to/en_US-lessac-medium.onnx
# or Russian
PIPER_MODEL_PATH=/absolute/path/to/ru_RU-irina-medium.onnx
```

## Configuration

Key variables (see `.env.example` for the full list):

- `GPT5_NANO_API_KEY`, `GPT5_NANO_BASE_URL`, `GPT5_NANO_MODEL`
- `AI_SYSTEM_PROMPT`
- `WHISPER_API_KEY`, `STT_PROVIDER`, `STT_MODEL`, `STT_LANGUAGE`
- `MACOS_TTS_VOICE` (e.g. a Russian-capable voice like `Milena` on macOS)
- `MACOS_TTS_RATE` (optional speech rate for macOS `say`, e.g. `180`)
- `PIPER_MODEL_PATH` (Linux only, e.g. a Russian voice model like `ru_RU-irina-medium`)
- `AUDIO_INPUT_DEVICE`, `AUDIO_OUTPUT_DEVICE`, `SAMPLE_RATE`, `CHUNK_SECONDS`
- `VOX_THRESHOLD_DB`, `VOX_MIN_DURATION_MS`

### DTMF control channel

RadioBuddy can listen for classic phone DTMF tones on the audio input and use them as a simple “remote control” over the radio.

- **Enable / basic config** (see `.env.example`):
  - `DTMF_ENABLED` – turn DTMF control on/off (default: on).
  - `DTMF_SECRET` – “open” prefix, default `0909`.
  - Timing / detection tuning: `DTMF_COMMAND_TIMEOUT_SEC`, `DTMF_DIGIT_GAP_TIMEOUT_SEC`, `DTMF_FRAME_MS`, `DTMF_HOP_MS`, `DTMF_MIN_TONE_MS`, `DTMF_ENERGY_GATE_DB`, `DTMF_PEAK_RATIO`, `DTMF_BANDPASS_ENABLED`.
- **Built-in commands** (after sending the secret `0909`):
  - `01` – switch to AI mode.
  - `02` – switch to repeater mode.
  - `03` – switch to dummy/offline mode.
  - `11` – Kira announces current local time.

Example sequence over the radio: dial `0909` to open, then `01` to switch to AI mode. The app confirms over TTS: “Кира передает – Код принят – команда 01” and then “Кира передает – Режим 01”.

## Running

Activate your virtualenv and then, for the simplest start:

```bash
python run.py
```

You can also use the CLI module directly:

```bash
python -m radiobuddy list-devices  # optional: list audio devices
python -m radiobuddy run --interactive-devices
```

### CLI help

```bash
python -m radiobuddy --help
python -m radiobuddy run --help
python -m radiobuddy list-devices --help
```

The script will:

1. Load configuration from `.env`.
2. Continuously monitor the configured input device.
3. When input energy exceeds the VOX threshold for long enough, record a segment.
4. Send the audio segment to Whisper for transcription (or a **dummy STT stub** if `WHISPER_API_KEY` is empty).
5. Send the text to GPT‑5 Nano and receive a reply (or a **dummy LLM stub** if `GPT5_NANO_API_KEY` is empty).
6. Use TTS to synthesize speech and play it to the configured output device (your radio).

Logs are printed to stdout; no API keys or secrets are logged. When keys are missing, no external API calls are made.

## Notes

- macOS uses the built-in `say` command for TTS. Linux uses local Piper (no network) when `PIPER_MODEL_PATH` is set.
- There is no PTT/CAT serial control in this version; VOX on the radio is expected to handle transmit.

