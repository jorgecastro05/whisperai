from RealtimeSTT import AudioToTextRecorder
from RealtimeSTT.transcription_engines.base import BaseTranscriptionEngine
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from http.server import ThreadingHTTPServer
import re
import signal

# Ensure base engine has a close method so RealtimeSTT shutdown doesn't raise AttributeError
if not hasattr(BaseTranscriptionEngine, "close"):
    BaseTranscriptionEngine.close = lambda self: None

HOST = "0.0.0.0"
PORT = 8765

# Get script directory (important!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_FILE = os.path.join(BASE_DIR, "prompt.txt")
HTML_FILE = os.path.join(BASE_DIR, "captions.html")
BLACKLIST_FILE = os.path.join(BASE_DIR, "blacklistwords.txt")

latest_text = ""
last_update = 0

stop_event = threading.Event()
server = None  # global reference
recorder = None
shutting_down = False


def load_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def clean_word(word):
    return re.sub(r'[^\w]', '', word).lower()

def load_blacklist():
    return load_file(BLACKLIST_FILE).splitlines()

blacklist = load_blacklist()
print(blacklist)

def process_text(text):
    global latest_text, last_update

    cleaned_blacklist = set(w.strip().lower() for w in blacklist if w.strip())

    filtered_words = []
    for word in text.split():
        clean = clean_word(word)
        if clean not in cleaned_blacklist:
            filtered_words.append(word)

    latest_text = " ".join(filtered_words)
    last_update = time.time()


class CaptionHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def do_GET(self):

        # Serve captions.html
        if self.path == "/":
            if os.path.exists(HTML_FILE):
                with open(HTML_FILE, "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        # Return captions text
        elif self.path == "/captions":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(latest_text.encode("utf-8"))

        # Return last update timestamp
        elif self.path == "/last_update":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(str(last_update).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


def realtime_update(text):
    global latest_text, last_update
    cleaned_blacklist = set(w.strip().lower() for w in blacklist if w.strip())

    filtered_words = []
    for word in text.split():
        clean = clean_word(word)
        if clean not in cleaned_blacklist:
            filtered_words.append(word)

    latest_text = " ".join(filtered_words)
    last_update = time.time()


def start_server():
    global server
    server = ThreadingHTTPServer((HOST, PORT), CaptionHandler)
    print(f"HTTP server running at http://localhost:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down HTTP server...")
        server.server_close()

def recorder_loop():
    global recorder
    recorder = AudioToTextRecorder(**recorder_config)
    try:
        while not stop_event.is_set():
            recorder.text(process_text)
    except Exception as e:
        print(f"Recorder exception: {e}")
    finally:
        print("Recorder loop exited")

def shutdown_handler(signum=None, frame=None):
    global shutting_down

    if shutting_down:
        print("Force exiting...")
        os._exit(0)

    shutting_down = True
    print("\nStopping gracefully...")

    stop_event.set()

    # Unblock the HTTP server thread/main loop first
    if server:
        threading.Thread(target=server.shutdown, daemon=True).start()

    # Shut down recorder
    if recorder:
        try:
            recorder.shutdown()
        except Exception as e:
            print(f"Error stopping recorder: {e}")


if __name__ == '__main__':
    print("Wait until it says 'speak now'", flush=True)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    unknown_sentence_detection_pause = 0.7

    recorder_config = {
        'spinner': False,
        #'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        # 'input_device_index': 1,
        'realtime_model_type': 'small.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.5,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        #'on_realtime_transcription_update': realtime_update,
        #'on_realtime_transcription_update': text_detected,
        'on_realtime_transcription_stabilized': realtime_update,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 1,
        'beam_size_realtime': 1,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt_realtime': load_file(PROMPT_FILE),
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
        'initial_prompt': load_file(PROMPT_FILE)
    }

    # Start recorder in background thread
    recorder_thread = threading.Thread(target=recorder_loop, daemon=True)
    recorder_thread.start()

    try:
        start_server()
    except KeyboardInterrupt:
        shutdown_handler()
    finally:
        shutdown_handler()
        recorder_thread.join(timeout=2)
        print("Exited cleanly")
        os._exit(0)
