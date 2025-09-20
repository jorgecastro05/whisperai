import sounddevice as sd
import numpy as np
import whisperx
import queue
import sys
import time
import os

# Settings
MODEL_SIZE = "medium.en"
LANGUAGE_CODE = "en"
SAMPLE_RATE = 16000
CHUNK_DURATION = 4  # Process audio in 4-second chunks
SILENCE_THRESHOLD = 0.05 # RMS energy threshold for detecting voice
MIN_VOICE_DURATION = 0.5 # Not directly used in this version but kept for context
SILENCE_TIMEOUT = 5 # Seconds of no new transcription before clearing the file
OUTPUT_FILE = "transcript.txt"
PROMPT_FILE = "prompt.txt"

# --- Model Initialization for whisperx ---
# 1. Load the base whisper model
# Using a GPU (cuda) with float16 compute type for speed and efficiency
model = whisperx.load_model(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"\nWhisperX model '{MODEL_SIZE}' loaded on cuda.")

# 2. Load the alignment model
# This is used to refine the timestamps of the transcribed words
align_model, align_metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device="cuda")
print("WhisperX alignment model loaded.")


# Audio processing state
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_write_time = time.time() # Tracks when content was last written to file
in_speech = False # Track if we are currently in a speech segment

# --- Function to load prompt from file ---
def load_prompt(file_path):
    """Loads an initial prompt from a text file."""
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
    """This function is called by sounddevice for each new audio block."""
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(indata.copy())

def is_voice_active(audio_chunk):
    """Use RMS energy to determine if the chunk contains voice."""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms > SILENCE_THRESHOLD

def clear_output_file():
    """Clears the output file if it exists and has content."""
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("") # Ensure file is empty
        print("\n[File cleared due to prolonged silence]", end="", flush=True)

def process_audio():
    """Main loop to process audio from the buffer."""
    global current_audio, in_speech, last_write_time

    while not audio_buffer.empty():
        chunk = audio_buffer.get()
        current_audio = np.concatenate((current_audio, chunk.flatten()))

        # Check for silence timeout based on when we last wrote a transcription
        if time.time() - last_write_time > SILENCE_TIMEOUT:
            clear_output_file()
            last_write_time = time.time() # Reset timer after clearing

        # Process the buffer when it has enough audio data
        if len(current_audio) >= SAMPLE_RATE * CHUNK_DURATION:
            audio_to_process = current_audio.copy()
            current_audio = np.array([], dtype=np.float32)

            # Optional: Normalize audio to [-1, 1] range if it's not already
            if np.max(np.abs(audio_to_process)) > 0:
                audio_to_process /= np.max(np.abs(audio_to_process))

            if is_voice_active(audio_to_process):
                if not in_speech:
                    print("\n[Voice detected, transcribing...]", end="", flush=True)
                    in_speech = True
                process_voice(audio_to_process)
            else:
                if in_speech:
                    print("\n[Silence detected]", end="", flush=True)
                    in_speech = False

def process_voice(audio):
    """
    Transcribe audio using whisperx, align the results, and save the text.
    """
    global last_write_time

    try:
        # 1. Transcribe audio using the base model
        result = model.transcribe(
            audio,
            language=LANGUAGE_CODE,
            beam_size=5,
            temperature=0.0,
            initial_prompt=initial_context_prompt
        )

        # Exit if no speech is detected in the chunk
        if not result["segments"]:
            return

        # 2. Align the transcribed segments for more accurate word timings
        aligned_result = whisperx.align(result["segments"], align_model, align_metadata, audio, device="cuda")

        # 3. Concatenate the text from aligned segments
        full_text = " ".join([segment['text'].strip() for segment in aligned_result["segments"]])
        text = full_text.strip()
        
        # Common "hallucinated" phrases to filter out
        filter_phrases = [
            "thanks for watching!", "thank you for watching", "thank you for watching!",
            "see you next time", "thank you for watching now!",
            "i'll see you guys in the next video. bye."
        ]

        # Only write meaningful text to the console and file
        if text and len(text) > 2 and text.lower() not in filter_phrases:
            # Print the new transcription, clearing the line first
            print("\r" + text + " " * 20, end="", flush=True)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            last_write_time = time.time() # IMPORTANT: Update timestamp when content is written

    except Exception as e:
        print(f"\nError during transcription: {str(e)}", file=sys.stderr)

# --- Main Execution ---
def main():
    clear_output_file() # Clear any previous content on startup
    global last_write_time
    last_write_time = time.time()
    
    print(f"Live transcription started. Listening...")
    print(f"(Output will be saved to '{OUTPUT_FILE}' and cleared after {SILENCE_TIMEOUT}s of no new content)")

    try:
        # Start the audio stream from the microphone
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.1), # Read in 100ms chunks
            dtype='float32'
        ):
            while True:
                process_audio()
                time.sleep(0.1) # Small delay to prevent a busy loop
    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()