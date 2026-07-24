from pathlib import Path
import wave

import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16_000
RECORD_SECONDS = 5
AUDIO_FILE = Path("my_recording.wav")


def record_audio():
    print(f"Recording for {RECORD_SECONDS} seconds...")
    print("Please speak Mandarin Chinese now.")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    with wave.open(str(AUDIO_FILE), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    print(f"Recording saved as: {AUDIO_FILE.resolve()}")
    return audio


def play_audio(audio):
    print("Playing the recording through the speaker...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def recognize_audio():
    print("Loading the speech-recognition model...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Recognizing Chinese speech...")

    segments, information = model.transcribe(
        str(AUDIO_FILE),
        language="zh",
        beam_size=1,
    )

    text = "".join(segment.text for segment in segments).strip()

    print(f"Detected language: {information.language}")
    print(f"Recognized text: {text or '[No speech recognized]'}")

    return text


def main():
    audio = record_audio()
    play_audio(audio)
    recognize_audio()


if __name__ == "__main__":
    main()