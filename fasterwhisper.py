import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
import time
import os
import threading

# Settings
MODEL_SIZE = "medium.en"  # try "small.en" for faster performance with good quality
SAMPLE_RATE = 16000
CHUNK_DURATION = 3   # shorter chunks improve latency
SILENCE_THRESHOLD = 0.05
MIN_VOICE_DURATION = 0.5
SILENCE_TIMEOUT = 5
OUTPUT_FILE = "transcript.txt"
PROMPT_FILE = "prompt.txt"

# --- Model Initialization for faster-whisper ---
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16", cpu_threads=4, num_workers=2)
print(f"\nModel '{MODEL_SIZE}' loaded on cuda.")

# Audio processing state
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_voice_time = time.time()
last_write_time = time.time()
in_speech = False

# --- Function to load prompt from file ---
def load_prompt(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return ""

initial_context_prompt = load_prompt(PROMPT_FILE)
if initial_context_prompt:
    print(f"Loaded initial prompt from {PROMPT_FILE}:")

# --- Audio callback ---
def audio_callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(indata.copy())

# --- Voice activity detection ---
def is_voice_active(audio_chunk):
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms > SILENCE_THRESHOLD

# --- File handling ---
def clear_output_file():
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, "w") as f:
            f.write("")
        print("\n[File cleared due to prolonged silence]", end="", flush=True)

# --- Processing ---
def process_audio():
    global current_audio, last_voice_time, in_speech, last_write_time

    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))

        if time.time() - last_write_time > SILENCE_TIMEOUT:
            clear_output_file()
            last_write_time = time.time()

        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_to_process = current_audio.copy()
            current_audio = np.array([], dtype=np.float32)

            if np.max(np.abs(audio_to_process)) > 0:
                audio_to_process /= np.max(np.abs(audio_to_process))

            if is_voice_active(audio_to_process):
                last_voice_time = time.time()
                if not in_speech:
                    print("\n[Voice detected]", end="", flush=True)
                    in_speech = True
                threading.Thread(target=process_voice, args=(audio_to_process,), daemon=True).start()
            else:
                if in_speech:
                    print("\n[Silence detected]", end="", flush=True)
                    in_speech = False

# --- Voice transcription ---
def process_voice(audio):
    global last_write_time
    try:
        segments, _ = model.transcribe(
            audio,
            beam_size=2,              # smaller beam improves latency
            best_of=2,
            language="en",
            temperature=0.0,
            suppress_tokens=[-1],
            initial_prompt=initial_context_prompt
        )

        text = " ".join([s.text.strip() for s in segments]).strip()

        if text and len(text) > 2:
            print("\r" + text + " " * 20, end="", flush=True)
            with open(OUTPUT_FILE, "a") as f:
                f.write(text + "\n")
            last_write_time = time.time()

    except Exception as e:
        print(f"\nError during transcription: {str(e)}", file=sys.stderr)

# --- Main Execution ---
def main():
    clear_output_file()
    global last_write_time
    last_write_time = time.time()
    print(f"Live transcription started (clears after {SILENCE_TIMEOUT}s of silence)...")

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.05),  # smaller block for lower latency
            dtype='float32'
        ):
            while True:
                process_audio()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
