from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    """
    This function is called every time a new text segment is transcribed.
    """
    print(text)

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    
    # Initialize the recorder with the spinner disabled
    # The corrected code
    recorder = AudioToTextRecorder(sample_rate=48000)

    print("Speak now...")
    
    # The while loop will keep listening and transcribing until you stop the script
    while True:
        recorder.text(process_text)