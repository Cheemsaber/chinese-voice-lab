"""Test Whisper start-boundary behavior with VAD explicitly disabled."""

from pathlib import Path
from time import perf_counter
import unicodedata
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# Use the only microphone path that passed the previous capture experiments.
DEVICE_INDEX = 1
CHANNELS = 1

# The pre-roll ensures that the complete first phoneme reaches the WAV file.
# It is removed before constructing the controlled padding variants.
PRE_SECONDS = 1.5
RECORD_SECONDS = 6.0

TARGET_TEXT = "你好，这是录音开始测试，请完整识别开头的文字。"

# The first variant begins approximately 50 ms before detected speech. Every
# other variant uses that exact same waveform with additional digital silence.
ONSET_MARGIN_MS = 50
PADDING_VALUES_MS = (0, 250, 500, 1000)

# If automatic onset detection crops the first phoneme incorrectly, listen to
# speech_crop.wav and set this to a manually chosen time in the raw recording.
ONSET_OVERRIDE_SECONDS = None

MODEL_SIZE = "base"
BEAM_SIZE = 1

OUTPUT_DIR = Path("experiment_output") / "leading_silence_whisper"


def dbfs_rms(samples):
    """Return RMS level in dBFS."""

    samples = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples * samples))
    return 20 * np.log10(max(rms, 1e-9))


def save_wav(path, mono_audio, sample_rate):
    """Save float audio as mono 16-bit PCM."""

    pcm = np.clip(mono_audio * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def record_phrase(sample_rate):
    """Capture one phrase with enough pre-roll to preserve its beginning."""

    raw_path = OUTPUT_DIR / "raw_recording.wav"

    print(f"Target phrase: {TARGET_TEXT}")
    input("Press Enter when ready...")

    with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        blocksize=0,
        latency="high",
    ) as stream:
        print("Microphone is active. Remain silent...")
        pre_roll, overflow1 = stream.read(
            int(PRE_SECONDS * sample_rate)
        )

        print("\nSPEAK NOW\n")
        speech, overflow2 = stream.read(
            int(RECORD_SECONDS * sample_rate)
        )
        print("DONE")

    audio = np.concatenate([pre_roll, speech], axis=0)[:, 0]
    save_wav(raw_path, audio, sample_rate)

    print(f"Input overflow: {overflow1 or overflow2}")
    print(f"Raw recording: {raw_path.resolve()}")

    return audio, raw_path


def detect_onset(audio, sample_rate):
    """Estimate speech onset from short-window energy after the cue."""

    window_seconds = 0.02
    window_samples = max(1, int(window_seconds * sample_rate))

    levels = np.array(
        [
            dbfs_rms(audio[start : start + window_samples])
            for start in range(
                0,
                len(audio) - window_samples + 1,
                window_samples,
            )
        ]
    )

    # Ignore the first 500 ms to avoid stream-start transients. Use the median
    # so one brief sound in the pre-roll does not inflate the noise estimate.
    noise_start = int(0.5 / window_seconds)
    noise_end = int((PRE_SECONDS - 0.2) / window_seconds)
    noise_db = float(np.median(levels[noise_start:noise_end]))

    # This detector only locates a safe crop point. It does not alter capture
    # or act as Faster-Whisper VAD.
    onset_threshold = max(noise_db + 8, -65)
    cue_window = int(PRE_SECONDS / window_seconds)
    candidates = np.flatnonzero(levels[cue_window:] >= onset_threshold)

    if len(candidates) == 0:
        raise RuntimeError(
            "No speech onset detected. Check raw_recording.wav."
        )

    onset_window = cue_window + int(candidates[0])
    onset_seconds = onset_window * window_seconds

    print(f"Noise estimate:  {noise_db:.1f} dBFS")
    print(f"Onset threshold: {onset_threshold:.1f} dBFS")
    print(f"Detected onset:  {onset_seconds:.3f} seconds")

    return onset_seconds


def make_padding_variants(audio, sample_rate, onset_seconds):
    """Create files differing only in silence before the shared speech crop."""

    crop_start_seconds = max(
        0.0,
        onset_seconds - ONSET_MARGIN_MS / 1000,
    )
    crop_start_sample = int(crop_start_seconds * sample_rate)
    speech_crop = audio[crop_start_sample:]

    crop_path = OUTPUT_DIR / "speech_crop.wav"
    save_wav(crop_path, speech_crop, sample_rate)

    print(f"Crop starts at: {crop_start_seconds:.3f} seconds")
    print(f"Listen to this before trusting the test: {crop_path.resolve()}")

    variants = []

    for padding_ms in PADDING_VALUES_MS:
        padding_samples = int(sample_rate * padding_ms / 1000)
        padding = np.zeros(padding_samples, dtype=np.float32)
        variant_audio = np.concatenate([padding, speech_crop])

        variant_path = OUTPUT_DIR / f"leading_padding_{padding_ms}ms.wav"
        save_wav(variant_path, variant_audio, sample_rate)
        variants.append((padding_ms, variant_path))

    return variants


def normalize_text(text):
    """Normalize punctuation and spacing for character-error comparison."""

    normalized = unicodedata.normalize("NFKC", text).upper()
    return "".join(character for character in normalized if character.isalnum())


def edit_distance(left, right):
    """Calculate Levenshtein distance between two strings."""

    previous = list(range(len(right) + 1))

    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]

        for right_index, right_character in enumerate(right, start=1):
            insertion = current[-1] + 1
            deletion = previous[right_index] + 1
            substitution = (
                previous[right_index - 1]
                + (left_character != right_character)
            )
            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]


def transcribe(model, audio_path):
    """Transcribe one variant with integrated Silero VAD disabled."""

    started = perf_counter()

    segments, _ = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=BEAM_SIZE,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,  # Explicitly disable Faster-Whisper/Silero VAD.
    )

    # Faster-Whisper returns a generator; inference occurs during iteration.
    segments = list(segments)
    elapsed = perf_counter() - started
    text = "".join(segment.text for segment in segments).strip()

    return text, elapsed, segments


def print_results(results):
    """Print padding, character error, latency, and recognized text."""

    expected = normalize_text(TARGET_TEXT)

    print()
    print("=" * 100)
    print("RESULTS — vad_filter=False for every transcription")
    print(
        f"{'Padding':>9} {'CER':>8} {'Time':>8} "
        f"{'First segment':>15}  Text"
    )
    print("-" * 100)

    for padding_ms, text, elapsed, segments in results:
        recognized = normalize_text(text)
        errors = edit_distance(recognized, expected)
        character_error_rate = errors / max(1, len(expected))
        first_segment = (
            f"{segments[0].start:.2f}s"
            if segments
            else "none"
        )

        print(
            f"{padding_ms:>7}ms "
            f"{character_error_rate:>7.1%} "
            f"{elapsed:>7.2f}s "
            f"{first_segment:>15}  "
            f"{text or '[no text]'}"
        )


def main():
    """Record once and compare Whisper output across padding values."""

    device = sd.query_devices(DEVICE_INDEX)
    host_api = sd.query_hostapis(device["hostapi"])
    sample_rate = int(device["default_samplerate"])

    print(f"Device:      {DEVICE_INDEX} — {device['name']}")
    print(f"Host API:    {host_api['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print("Whisper VAD: disabled with vad_filter=False")

    sd.check_input_settings(
        device=DEVICE_INDEX,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, _ = record_phrase(sample_rate)

    if ONSET_OVERRIDE_SECONDS is None:
        onset_seconds = detect_onset(audio, sample_rate)
    else:
        onset_seconds = ONSET_OVERRIDE_SECONDS
        print(f"Using manual onset: {onset_seconds:.3f} seconds")

    variants = make_padding_variants(
        audio,
        sample_rate,
        onset_seconds,
    )

    print()
    print(f"Loading Faster-Whisper model: {MODEL_SIZE}")
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )

    # Warm the already-loaded model so the first measured padding case does
    # not include one-time inference initialization overhead.
    print("Running one unmeasured model warm-up...")
    transcribe(model, variants[-1][1])

    results = []
    for padding_ms, variant_path in variants:
        text, elapsed, segments = transcribe(model, variant_path)
        results.append((padding_ms, text, elapsed, segments))

    print_results(results)
    print()
    print("If padding improves the first characters, retain pre-roll in the")
    print("application. If every result is identical, padding is not the cause.")


if __name__ == "__main__":
    main()
