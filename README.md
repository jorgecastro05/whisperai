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



## CPU Support

uv pip uninstall torch torchaudio faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
uv pip install "RealtimeSTT[sherpa-onnx]"


## Configurations

### CPU FASTER WHISPER

```python
    recorder_config = {
    'spinner': False,
    'device': 'cpu',
    'compute_type': 'int8',              # big CPU win vs default float32
    #'cpu_threads': 4,                    # tune to (your cores - RVC's needs); try 2-4
    #'num_workers': 1,                    # don't parallelize beyond 1 stream
    'download_root': None,
    'realtime_model_type': 'tiny.en',    # lighter realtime pass; try distil-small.en if accuracy suffers
    'language': 'en',
    'silero_sensitivity': 0.05,
    'webrtc_sensitivity': 3,
    'post_speech_silence_duration': unknown_sentence_detection_pause,
    'min_length_of_recording': 1.5,
    'min_gap_between_recordings': 0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0.15,   # was 0.02 — much less frequent CPU bursts
    'on_realtime_transcription_stabilized': realtime_update,
    'silero_deactivity_detection': True,
    'early_transcription_on_silence': 0,
    'beam_size': 1,
    'beam_size_realtime': 1,
    'no_log_file': True,
    'initial_prompt_realtime': load_file(PROMPT_FILE),
    'silero_use_onnx': True,
    'faster_whisper_vad_filter': False,
    'initial_prompt': load_file(PROMPT_FILE),
    # if cpu_threads/num_workers above aren't recognized by your installed version:
    'transcription_engine_options': {'cpu_threads': 4, 'num_workers': 1},
    }
```

### GPU NVIDIA

```python
    GPU
    recorder_config = {
        'spinner': False,
        #'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        # 'input_device_index': 1,
        'realtime_model_type': 'small.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.5,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        #'on_realtime_transcription_update': realtime_update,
        #'on_realtime_transcription_update': text_detected,
        'on_realtime_transcription_stabilized': realtime_update,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 1,
        'beam_size_realtime': 1,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt_realtime': load_file(PROMPT_FILE),
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
        'initial_prompt': load_file(PROMPT_FILE)
    }
```    
