import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
BLOCK_SIZE = 1280
THRESHOLD = 0.5

model = Model(wakeword_models=["hey_mycroft"])

print("=" * 50)
print("HEY MYCROFT — REAL MICROPHONE TEST")
print("=" * 50)
print("Model: hey_mycroft_v0.1")
print(f"Threshold: {THRESHOLD}")
print()
print("Listening...")
print('Say: "Hey Mycroft"')
print("Press Ctrl+C to stop.")
print()

last_print = 0

def callback(indata, frames, time_info, status):
    global last_print

    if status:
        print("Audio status:", status)

    audio = (indata[:, 0] * 32767).astype(np.int16)

    prediction = model.predict(audio)

    score = float(prediction["hey_mycroft"])

    now = time.time()

    if now - last_print > 0.5:
        print(f"score = {score:.3f}")
        last_print = now

    if score >= THRESHOLD:
        print()
        print("========================================")
        print("   HEY MYCROFT DETECTED!")
        print("========================================")
        print()


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=BLOCK_SIZE,
    callback=callback,
):
    while True:
        time.sleep(0.1)