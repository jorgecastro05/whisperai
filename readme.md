## Different implementations for whisper AI Live Captions.

the latest implementation is fasterwhisper.

### Conda tutorial:

```bash
conda create --name whisper-env python=3.9
conda install -c conda-forge cudatoolkit=12.4 cudnn
pip install faster-whisper
pip install sounddevice
pip install python-pyaudio
sudo apt-get install python-pyaudio
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
pip install pyaudio numpy
conda install -c conda-forge libstdcxx-ng

export LD_LIBRARY_PATH=/home/$USER/.conda/envs/whisper-env/lib/python3.9/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
```