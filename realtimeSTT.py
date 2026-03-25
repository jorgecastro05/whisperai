from RealtimeSTT import AudioToTextRecorder
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from http.server import ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 8765

# Get script directory (important!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_FILE = os.path.join(BASE_DIR, "prompt.txt")
HTML_FILE = os.path.join(BASE_DIR, "captions.html")

latest_text = ""
last_update = 0


def load_prompt(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def process_text(text):
    global latest_text, last_update

    print(text, flush=True)
    latest_text = text
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
    latest_text = text
    last_update = time.time()


def start_server():
    #server = HTTPServer((HOST, PORT), CaptionHandler)
    server = ThreadingHTTPServer((HOST, PORT), CaptionHandler)
    print(f"HTTP server running at http://localhost:{PORT}", flush=True)
    server.serve_forever()

def recorder_loop():
    recorder = AudioToTextRecorder(**recorder_config)
    while True:
        recorder.text(process_text)


if __name__ == '__main__':
    print("Wait until it says 'speak now'", flush=True)

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
        'on_realtime_transcription_update': realtime_update,
        #'on_realtime_transcription_update': text_detected,
        #'on_realtime_transcription_stabilized': realtime_update,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 3,
        'beam_size_realtime': 3,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt_realtime': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
        'initial_prompt': load_prompt(PROMPT_FILE)
    }

    # Start recorder in background thread
    threading.Thread(target=recorder_loop, daemon=True).start()

    # Run server in MAIN thread (important!)
    start_server()
