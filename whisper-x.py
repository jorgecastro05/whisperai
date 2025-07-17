import sounddevice as sd
import numpy as np
import whisperx
import queue
import sys
import time
import torch
import tempfile
import soundfile as sf

# --- Settings ---
# You can change the model size to "large-v2" for better accuracy,
# but it will be slower and require more memory.
MODEL_SIZE = "small"
LANGUAGE = "en"
SAMPLE_RATE = 16000
SILENCE_TIMEOUT = 3  # Seconds of silence to trigger transcription
MIN_VOICE_DURATION = 0.5  # Minimum speech duration to process
SILENCE_THRESHOLD = 0.03  # RMS threshold to detect silence
OUTPUT_FILE = "transcript_whisperx.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# --- Initialization ---
print("Initializing WhisperX...")

# Load WhisperX model
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
print("WhisperX model loaded.")

# Audio processing
audio_buffer = queue.Queue()
current_audio = np.array([], dtype=np.float32)
last_voice_time = time.time()
is_speaking = False

def audio_callback(indata, frames, time, status):
    """This function is called for each audio chunk from the microphone."""
    if status:
        print(status, file=sys.stderr)
    # Add the audio chunk to the buffer
    audio_buffer.put(indata.copy())

def is_voice_active(audio_chunk):
    """Check if the audio chunk contains speech."""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms > SILENCE_THRESHOLD

def process_and_transcribe_audio(audio_data):
    """
    Transcribe the accumulated audio using WhisperX.
    This includes transcription, alignment, and speaker diarization.
    """
    print("\nProcessing accumulated audio...")
    try:
        # Save the accumulated audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            sf.write(temp_audio_file.name, audio_data, SAMPLE_RATE)
            temp_file_path = temp_audio_file.name

        # --- WhisperX Pipeline ---
        # 1. Load audio
        audio = whisperx.load_audio(temp_file_path)

        # 2. Transcribe
        result = model.transcribe(audio, batch_size=16)

        # 3. Align transcription
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

        # 4. Diarize speakers
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="HUHING_FACE_TOKENS", device=DEVICE)
        # Add min/max number of speakers if known
        # diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # --- Output ---
        print("Transcription complete. Appending to file.")
        with open(OUTPUT_FILE, "a") as f:
            for segment in result["segments"]:
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                text = segment['text']
                line = f"[{speaker}] {text.strip()}"
                print(line)
                f.write(line + "\n")
            f.write("\n" + "="*30 + "\n\n")


    except Exception as e:
        print(f"\nAn error occurred during transcription: {e}", file=sys.stderr)

def main_loop():
    """Main loop to capture and process audio."""
    global current_audio, last_voice_time, is_speaking

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.1), # 100ms chunks
            dtype='float32'
        ):
            print(f"\nListening... Speak into the microphone. Silence for {SILENCE_TIMEOUT} seconds will trigger transcription.")
            while True:
                while not audio_buffer.empty():
                    chunk = audio_buffer.get()
                    if is_voice_active(chunk):
                        if not is_speaking:
                            is_speaking = True
                            print("Speaking detected...", end="", flush=True)
                        current_audio = np.concatenate((current_audio, chunk.flatten()))
                        last_voice_time = time.time()
                    elif is_speaking:
                        current_audio = np.concatenate((current_audio, chunk.flatten()))


                if is_speaking and (time.time() - last_voice_time > SILENCE_TIMEOUT):
                    print("\nSilence detected.")
                    if len(current_audio) > SAMPLE_RATE * MIN_VOICE_DURATION:
                         process_and_transcribe_audio(current_audio)
                    else:
                        print("Not enough voice data to process.")

                    # Reset for next round
                    current_audio = np.array([], dtype=np.float32)
                    is_speaking = False
                    print(f"\nListening...")


                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Clear the output file at the start
    open(OUTPUT_FILE, "w").close()
    main_loop()