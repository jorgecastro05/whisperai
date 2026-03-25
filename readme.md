## Different implementations for whisper AI Live Captions.
the latest implementation Realtime TTS
Exposes captions via http

## Realtime TTS

~~~bash
conda create -n realtimestt python=3.11 -y
conda activate realtimestt
sudo dnf install -y portaudio-devel ffmpeg
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/KoljaB/RealtimeSTT.git
cd RealtimeSTT

pip install -r requirements-gpu.txt
pip install -e .
python -c "import torch; print(torch.cuda.get_device_name(0))"

~~~

### Fasterwhisper implementation (Old):

```bash
conda create --name whisper-env python=3.9
pip install faster-whisper
pip install sounddevice
pip install python-pyaudio
sudo apt-get install python-pyaudio
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
pip install pyaudio numpy
conda install -c conda-forge libstdcxx-ng
 ```
## Linking cuda libraries

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux

```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
## Yaml Environment
Create with ```conda env create -f environment.yml```

```yaml
name: whisper-env
channels:
  - defaults
dependencies:
  - python=3.9
  - libstdcxx-ng
  - pip:
    - conda-env-export==0.6.1
    - faster-whisper==1.1.1
    - jaraco.collections==5.1.0
    - nvidia-cudnn-cu12==9.11.0.98
    - pip==25.1
    - platformdirs==4.2.2
    - pyaudio==0.2.14
    - sounddevice==0.5.2
    - tomli==2.0.1
    - wheel==0.45.1
```
