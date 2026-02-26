import sounddevice as sd
import numpy as np
import queue
import sys
import time
import torch
import os
from nemo.collections.asr.models import EncDecRNNTBPEModel

# Settings
MODEL_NAME = "stt_en_conformer_transducer_large"
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # seconds
OUTPUT_FILE = "transcript_nemo.txt"
PROMPT_FILE = "prompt.txt"

# Load model (pretrained from NGC)
print("Loading NeMo model...")
model = EncDecRNNTBPEModel.from_pretrained(model_name=MODEL_NAME)
model.eval()
print(f"Model '{MODEL_NAME}' loaded.")

# Load initial prompt (context)
def load_prompt(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

initial_prompt = load_prompt(PROMPT_FILE)
if initial_prompt:
    print(f"Loaded initial prompt from {PROMPT_FILE}: {initial_prompt}")

# Audio buffer
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)

# Callback for audio input
def audio_callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(indata.copy())

# Process audio in streaming fashion
def process_audio():
    global current_audio

    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))

        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_to_process = current_audio.copy()
            current_audio = np.array([], dtype=np.float32)

            # Transcribe chunk
            with torch.no_grad():
                results = model.transcribe([audio_to_process], batch_size=1)

            if results and len(results) > 0:
                text = results[0]
                if hasattr(text, "text"):
                    text = text.text
                text = str(text).strip()

                if text:
                    if initial_prompt:
                        text = f"{initial_prompt} {text}"

                    print("\r" + text + " " * 20, end="", flush=True)
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(text + "\n")

# Main loop
def main():
    print("Live transcription with NVIDIA NeMo started...")

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.1),
            dtype='float32'
        ):
            while True:
                process_audio()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
