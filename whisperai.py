import sounddevice as sd
import numpy as np
import whisper
import queue
import sys
import time

# Settings
MODEL_SIZE = "small"
SAMPLE_RATE = 16000
CHUNK_DURATION = 3          # 2-second chunks for faster response
SILENCE_THRESHOLD = 0.05     # Higher threshold to ignore background noise
MIN_VOICE_DURATION = 0.5     # Minimum speech duration to process
SILENCE_TIMEOUT = 5          # Clear file after 5 seconds of silence
OUTPUT_FILE = "transcript.txt"

# Initialize Whisper
model = whisper.load_model(MODEL_SIZE)

# Audio processing
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_voice_time = time.time()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(indata.copy())

def is_voice_active(audio_chunk):
    """Improved voice detection using RMS energy"""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms > SILENCE_THRESHOLD

def clear_output_file():
    """Clears the output file"""
    open(OUTPUT_FILE, "w").close()
    print("\n[File cleared due to silence]", end="", flush=True)

def process_audio():
    global current_audio, last_voice_time
    
    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))
        
        # Check for silence timeout
        if time.time() - last_voice_time > SILENCE_TIMEOUT:
            clear_output_file()
            last_voice_time = time.time()  # Reset timer
        
        # Process when we have enough audio
        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio = current_audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio /= np.max(np.abs(audio))
            
            if is_voice_active(audio):
                last_voice_time = time.time()  # Update last voice time
                process_voice(audio)
            
            current_audio = np.array([], dtype=np.float32)

def process_voice(audio):
    """Process and save valid voice segments"""
    try:
        result = model.transcribe(
            audio,
            fp16=False,
            language="en",
            temperature=0.0,      # Reduce randomness
            suppress_tokens=[-1]  # Suppress filler words
        )
        
        text = result.get("text", "").strip()
        if text and len(text) > 2:  # Ignore very short outputs
            print("\r" + text + " " * 20, end="", flush=True)
            with open(OUTPUT_FILE, "a") as f:
                f.write(text + "\n")
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)

# Initialize
clear_output_file()  # Start with clean file
print(f"Live transcription (clears after {SILENCE_TIMEOUT}s silence)...")

try:
    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE//10,
        dtype='float32'
    ):
        while True:
            process_audio()
except KeyboardInterrupt:
    print("\nStopped transcription")
