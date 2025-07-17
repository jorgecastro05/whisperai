import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
import time
import os 

# Settings
MODEL_SIZE = "medium.en"         # Model size (e.g., "tiny", "base", "small", "medium", "large-v3")
# For multi-lingual models, append ".en" for English-only version (e.g., "small.en")
SAMPLE_RATE = 16000
CHUNK_DURATION = 4           # Process audio in 3-second chunks
SILENCE_THRESHOLD = 0.05     # RMS energy threshold to detect voice
MIN_VOICE_DURATION = 0.5     # Minimum speech duration to process
SILENCE_TIMEOUT = 5          # Clear transcript file after 5 seconds of silence
OUTPUT_FILE = "transcript.txt"
PROMPT_FILE = "prompt.txt" 

# --- Model Initialization for faster-whisper ---
# On first run, the model is downloaded automatically.
# device="cuda" for GPU (with float16) or "cpu" for CPU (with int8)
# For GPU: model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
# For CPU:
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
#model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"\nModel '{MODEL_SIZE}' loaded on cuda.")


# Audio processing state
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_voice_time = time.time()
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
    with open(OUTPUT_FILE, "w") as f:
        f.write("") # Ensure file is empty
    print("\n[File cleared due to silence]", end="", flush=True)

def process_audio():
    """Main loop to process audio from the buffer."""
    global current_audio, last_voice_time, in_speech
    
    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))
        
        # Check for silence timeout and clear the file if needed
        # This part is fine for clearing the file, but doesn't stop transcription.
        if time.time() - last_voice_time > SILENCE_TIMEOUT:
            # Only clear if the file actually has content
            if open(OUTPUT_FILE, "r").read().strip(): 
                clear_output_file()
            last_voice_time = time.time()  # Reset timer to prevent rapid clearing

        # Process the buffer when it has enough audio data
        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_to_process = current_audio.copy()
            current_audio = np.array([], dtype=np.float32) # Clear buffer after copying
            
            # Normalize audio (optional but good practice)
            if np.max(np.abs(audio_to_process)) > 0:
                audio_to_process /= np.max(np.abs(audio_to_process))
            
            # --- Key Change Here: Control transcription based on voice activity ---
            if is_voice_active(audio_to_process):
                if not in_speech:
                    print("\n[Voice detected, starting transcription...]", end="", flush=True)
                    in_speech = True # Mark as in speech
                last_voice_time = time.time() # Update time of last detected voice
                process_voice(audio_to_process)
            else:
                if in_speech:
                    print("\n[Silence detected, stopping transcription...]", end="", flush=True)
                    in_speech = False # Mark as out of speech
                # If not in speech, and current chunk is silent, do NOT call process_voice
                # This prevents the model from transcribing silence.
                
def process_voice(audio):
    """
    Transcribe audio using faster-whisper and save the result.
    This function is now adapted for faster-whisper's output.
    """
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
        
        # Additional check: only write if the text is not a common "silence" output
        # You can expand this list if you find other common outputs for silence
        if text and len(text) > 2 and text.lower() not in ["thanks for watching!", 
                                                           "Thank you for watching", 
                                                           "Thank you for watching!",
                                                           "See you next time",
                                                           "Thank you for watching now!"
                                                           "I'll see you guys in the next video. Bye."]: # Added condition
            print("\r" + text + " " * 20, end="", flush=True)
            with open(OUTPUT_FILE, "a") as f:
                f.write(text + "\n")
        # else:
        #     # Optional: print a message if silence was processed but suppressed
        #     print("\r[Skipping silence transcription]", end="", flush=True)

    except Exception as e:
        print(f"\nError during transcription: {str(e)}", file=sys.stderr)

# --- Main Execution ---
def main():
    clear_output_file()
    print(f"Live transcription started (clears after {SILENCE_TIMEOUT}s of silence)...")

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