import wave
import sounddevice as sd

DURATION_SECONDS = 5
WARMUP_SECONDS = 1.5
OUTPUT_FILE = "microphone_test.wav"

device_info = sd.query_devices(kind="input")
sample_rate = int(device_info["default_samplerate"])

print("Microphone:", device_info["name"])
print("Native sample rate:", sample_rate)

with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype="int16",
    blocksize=0,
    latency="high",
) as stream:
    print("Warming up microphone...")

    # Record and discard the unstable startup period.
    _, warmup_overflow = stream.read(
        int(sample_rate * WARMUP_SECONDS)
    )

    print("Speak now: 你好，这是录音测试。")

    audio, recording_overflow = stream.read(
        int(sample_rate * DURATION_SECONDS)
    )

print("Warmup overflow:", warmup_overflow)
print("Recording overflow:", recording_overflow)

with wave.open(OUTPUT_FILE, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio.tobytes())

print("Saved:", OUTPUT_FILE)