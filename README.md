# RealtimeSTT Browser Captions

Real-time speech-to-text transcription with browser-based captions designed for OBS Studio or live streaming overlay.

---

## Prerequisites

- [Homebrew](https://brew.sh/)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

---

## Installation

1. **Install system dependencies with Homebrew:**

   ```bash
   brew install uv portaudio direnv
   ```

2. **Set up the Python environment:**

   ```bash
   uv sync
   ```

3. **Configure environment variables:**

   Allow `direnv` to automatically load PortAudio compilation flags and CUDA library paths defined in `.envrc`:

   ```bash
   direnv allow
   ```

   *(Optional)* If not using `direnv`, export the variables manually:

   ```bash
   export PKG_CONFIG_PATH="$(brew --prefix portaudio)/lib/pkgconfig:$PKG_CONFIG_PATH"
   export CPPFLAGS="-I$(brew --prefix portaudio)/include"
   export LDFLAGS="-L$(brew --prefix portaudio)/lib"
   export LD_LIBRARY_PATH="$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -type d -name lib -printf '%p:' 2>/dev/null)$LD_LIBRARY_PATH"
   ```

---

## Running

1. **Start the transcription server:**

   ```bash
   uv run realtimeSTT.py
   ```

2. **Display Captions:**
   - **In OBS Studio:** Add a **Browser Source** pointing to `http://localhost:8765` (or load `captions.html`).
   - **In a Web Browser:** Navigate to `http://localhost:8765`.

---

## Configuration

- **`prompt.txt`**: Add initial context or custom prompts to guide Whisper transcription.
- **`blacklistwords.txt`**: List words (one per line) to filter out from live captions.