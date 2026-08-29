### Installing (homebrew)

```bash
brew install uv
brew install portaudio
brew install direnv
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate


export PKG_CONFIG_PATH="$(brew --prefix portaudio)/lib/pkgconfig:$PKG_CONFIG_PATH"
export CPPFLAGS="-I$(brew --prefix portaudio)/include"
export LDFLAGS="-L$(brew --prefix portaudio)/lib"

uv add "RealtimeSTT[faster-whisper]"
uv add nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

export LD_LIBRARY_PATH="$(
    find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" \
        -type d -name lib \
        -printf '%p:' 2>/dev/null
)$LD_LIBRARY_PATH"

uv run realtimeSTT.py
```

## Running

```bash
cd ~/whisperai
uv run realtimeSTT.py
```