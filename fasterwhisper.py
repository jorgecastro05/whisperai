import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
import time
import os

# Settings
MODEL_SIZE = "medium.en"
SAMPLE_RATE = 16000
CHUNK_DURATION = 4
SILENCE_THRESHOLD = 0.05
MIN_VOICE_DURATION = 0.5
SILENCE_TIMEOUT = 5
OUTPUT_FILE = "transcript.txt"
PROMPT_FILE = "prompt.txt"

# --- Model Initialization for faster-whisper ---
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"\nModel '{MODEL_SIZE}' loaded on cuda.")

# Audio processing state
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_voice_time = time.time() # This still tracks last detected voice activity
last_write_time = time.time() # NEW: Tracks when content was last written to file
in_speech = False # Track if we are currently in a speech segment

# --- Function to load prompt from file ---
def load_prompt(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        print(f"Warning: Prompt file '{file_path}' not found. No initial prompt will be used.", file=sys.stderr)
        return ""

# Load the initial prompt once when the script starts
initial_context_prompt = load_prompt(PROMPT_FILE)
if initial_context_prompt:
    print(f"Loaded initial prompt from {PROMPT_FILE}:\n'{initial_context_prompt}'")

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for each audio block."""
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(indata.copy())

def is_voice_active(audio_chunk):
    """Use RMS energy to determine if the chunk contains voice."""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms > SILENCE_THRESHOLD

def clear_output_file():
    """Clears the output file."""
    # Only clear if the file exists and has content to avoid unnecessary operations
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, "w") as f:
            f.write("") # Ensure file is empty
        print("\n[File cleared due to prolonged silence / no new transcription]", end="", flush=True)

def process_audio():
    """Main loop to process audio from the buffer."""
    global current_audio, last_voice_time, in_speech, last_write_time # Add last_write_time

    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))

        # NEW: Check for silence timeout based on last_write_time
        if time.time() - last_write_time > SILENCE_TIMEOUT:
            clear_output_file()
            # Reset last_write_time after clearing to prevent immediate re-clear
            last_write_time = time.time()

        # Process the buffer when it has enough audio data
        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_to_process = current_audio.copy()
            current_audio = np.array([], dtype=np.float32)

            # Normalize audio (optional but good practice)
            if np.max(np.abs(audio_to_process)) > 0:
                audio_to_process /= np.max(np.abs(audio_to_process))

            if is_voice_active(audio_to_process):
                last_voice_time = time.time() # Update last voice activity
                if not in_speech:
                    print("\n[Voice detected, starting transcription...]", end="", flush=True)
                    in_speech = True
                process_voice(audio_to_process)
            else:
                if in_speech:
                    print("\n[Silence detected, stopping transcription...]", end="", flush=True)
                    in_speech = False

def process_voice(audio):
    """
    Transcribe audio using faster-whisper and save the result.
    """
    global last_write_time # Declare as global to modify it

    try:
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language="en",
            temperature=0.0,
            suppress_tokens=[-1],
            initial_prompt=initial_context_prompt
        )

        full_text = ""
        for segment in segments:
            full_text += segment.text.strip() + " "

        text = full_text.strip()

        # Only write if text is meaningful and not a common "silence" output
        if text and len(text) > 2 and text.lower() not in ["thanks for watching!",
                                                           "thank you for watching",
                                                           "thank you for watching!",
                                                           "see you next time",
                                                           "thank you for watching now!",
                                                           "i'll see you guys in the next video. bye."]:
            print("\r" + text + " " * 20, end="", flush=True)
            with open(OUTPUT_FILE, "a") as f:
                f.write(text + "\n")
            last_write_time = time.time() # IMPORTANT: Update when content is written
        # else:
        #     # Optional: print a message if silence was processed but suppressed
        #     print("\r[Skipping silence transcription / filtering empty output]", end="", flush=True)

    except Exception as e:
        print(f"\nError during transcription: {str(e)}", file=sys.stderr)

# --- Main Execution ---
def main():
    clear_output_file() # Clear on startup
    # Initialize last_write_time to ensure it starts counting correctly from the beginning
    global last_write_time
    last_write_time = time.time()
    print(f"Live transcription started (clears after {SILENCE_TIMEOUT}s of no new content)...")

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
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()