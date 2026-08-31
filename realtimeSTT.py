import os

# Route this process's default audio input to the DJI mic via PipeWire's
# pulse-compatibility layer. This must be set before pyaudio/PortAudio
# initializes, so PipeWire handles rate conversion (e.g. 48kHz -> 16kHz)
# instead of PortAudio trying to open the source's native format directly,
# which fails validation and can crash the process.
os.environ.setdefault(
    "PULSE_SOURCE",
    "alsa_input.usb-DJI_Technology_Co.__Ltd._DJI_MIC_MINI_XSP12345678B-01.analog-stereo",
)

import ctypes

# Silence harmless ALSA lib stderr noise (unknown PCM cards.pcm.rear/hdmi/
# modem/etc, dmix "unable to open slave"). These come from alsa-lib probing
# generic surround/hdmi/modem slots defined in the system's default
# alsa.conf that don't exist on this hardware -- cosmetic only, doesn't
# affect functionality. Must be set before any PyAudio()/Pa_Initialize call.
_ALSA_ERROR_HANDLER = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _noop_alsa_error_handler(filename, line, function, err, fmt):
    pass


_c_alsa_error_handler = _ALSA_ERROR_HANDLER(_noop_alsa_error_handler)


def suppress_alsa_errors():
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        asound.snd_lib_error_set_handler(_c_alsa_error_handler)
    except OSError:
        pass  # libasound not found under this name; skip silently


#Uncomment for debug alsa errors
suppress_alsa_errors()

from RealtimeSTT import AudioToTextRecorder
from RealtimeSTT.transcription_engines.base import BaseTranscriptionEngine
import time
import pyaudio
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


def find_input_device_index(name_substr, fallback=None):
    """Return the PortAudio device index whose name contains name_substr
    (case-insensitive) and has at least one input channel. Note: this index
    is PortAudio's own numbering and will NOT match pw-top/pw-dump IDs."""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if (name_substr.lower() in info.get('name', '').lower()
                    and info.get('maxInputChannels', 0) > 0):
                print(f"Found input device [{i}]: {info['name']}")
                return i
    finally:
        p.terminate()
    print(f"Warning: no input device matching '{name_substr}' found, using fallback={fallback}")
    return fallback


def list_input_devices():
    """Debug helper: print all PortAudio devices with input channels."""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                print(f"[{i}] {info['name']} (in ch: {info['maxInputChannels']})")
    finally:
        p.terminate()


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
    'device': 'cpu',

    # --- Final transcript: Parakeet, decoded once per turn ---
    'transcription_engine': 'sherpa_onnx_parakeet',
    'model': './models/sherpa-onnx/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8',
    'transcription_engine_options': {
        'num_threads': 4,
        'provider': 'cpu',
    },

    # --- Realtime/partial transcript: Nemotron streaming ---
    'enable_realtime_transcription': True,
    'realtime_transcription_engine': 'sherpa_onnx_nemotron',
    'realtime_model_type': './models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11',
    'realtime_transcription_engine_options': {
        'num_threads': 2,
        'provider': 'cpu',
    },
    'realtime_processing_pause': 0.15,
    'on_realtime_transcription_stabilized': realtime_update,

    'language': 'en',
    'silero_sensitivity': 0.05,
    'webrtc_sensitivity': 3,
    'post_speech_silence_duration': unknown_sentence_detection_pause,
    'min_length_of_recording': 1.5,
    'min_gap_between_recordings': 0,
    'silero_deactivity_detection': True,
    'early_transcription_on_silence': 0,
    'no_log_file': True,
    'silero_use_onnx': True,
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
