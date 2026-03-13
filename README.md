# RadioBuddy – VHF Radio ↔ Voice AI Pipeline

RadioBuddy turns a computer with a USB audio dongle into a VOX‑driven AI assistant over VHF radio:

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

## Setup

```bash
cd RadioBuddy

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

Fill in your API keys and adjust audio/VOX settings as needed.

### Linux (Mint) TTS setup (Piper, local/offline)

On Linux, RadioBuddy uses **Piper** for local TTS. You need to install Piper and provide a voice model path.

Install the Python package:

```bash
python -m pip install piper-tts
```

Download a voice model (example: `en_US-lessac-medium`):

```bash
python -m piper.download_voices en_US-lessac-medium
```

Set `PIPER_MODEL_PATH` in your `.env` to the downloaded `.onnx` model file path, for example:

```bash
PIPER_MODEL_PATH=/absolute/path/to/en_US-lessac-medium.onnx
```

## Configuration

Key variables (see `.env.example` for the full list):

- `GPT5_NANO_API_KEY`, `GPT5_NANO_BASE_URL`, `GPT5_NANO_MODEL`
- `AI_SYSTEM_PROMPT`
- `WHISPER_API_KEY`, `STT_PROVIDER`, `STT_MODEL`, `STT_LANGUAGE`
- `MACOS_TTS_VOICE`
- `PIPER_MODEL_PATH` (Linux only)
- `AUDIO_INPUT_DEVICE`, `AUDIO_OUTPUT_DEVICE`, `SAMPLE_RATE`, `CHUNK_SECONDS`
- `VOX_THRESHOLD_DB`, `VOX_MIN_DURATION_MS`

## Running

Activate your virtualenv and then:

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

