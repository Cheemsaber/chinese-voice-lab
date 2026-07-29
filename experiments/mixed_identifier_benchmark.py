"""Benchmark Chinese-English operational identifiers on fixed WAV files.

The first run can record a small corpus through the verified Realtek MME
device. Later runs reuse those exact WAV files, allowing Faster-Whisper model
sizes to be compared without introducing differences between spoken takes.

Examples:
    python experiments/mixed_identifier_benchmark.py --record
    python experiments/mixed_identifier_benchmark.py
    python experiments/mixed_identifier_benchmark.py --record --overwrite
"""

from argparse import ArgumentParser
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter
import unicodedata
import wave

from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd


# Device 1 was the only completed Windows host-API path that reliably captured
# meaningful speech in the earlier microphone experiments.
DEVICE_INDEX = 1
CHANNELS = 1

# Retain silence before the cue to protect the first phoneme. Each resulting
# WAV is deliberately reused for every model, so capture remains controlled.
PRE_ROLL_SECONDS = 1.0
SPEECH_SECONDS = 5.0

# Beam 5 did not improve the fixed-WAV benchmark and was consistently slower.
MODEL_SIZES = ("small",)
BEAM_SIZE = 1
LANGUAGE = "zh"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Two takes per phrase reveal whether a conclusion depends on one unusually
# clear pronunciation while keeping the initial corpus reasonably small.
REPEATS_PER_CASE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiment_output"
    / "mixed_identifier_benchmark"
)
AUDIO_DIR = OUTPUT_DIR / "audio"


@dataclass(frozen=True)
class PhraseCase:
    """Define one spoken phrase and its task-relevant canonical fields."""

    name: str
    prompt: str
    expected_stream: str


# The canonical stream contains only the information that would drive a
# peer-checking decision. Chinese digit pronunciations are represented as
# Arabic digits, and English letters/states remain uppercase ASCII.
PHRASE_CASES = (
    PhraseCase(
        name="device_abc123",
        prompt="设备编号 A B C 一二三。",
        expected_stream="ABC123",
    ),
    PhraseCase(
        name="verification_xq789",
        prompt="验证码 X Q 七八九。",
        expected_stream="XQ789",
    ),
    PhraseCase(
        name="room_b204",
        prompt="房间 B 二零四。",
        expected_stream="B204",
    ),
    PhraseCase(
        name="serial_rt562",
        prompt="序列号 R T 五六二。",
        expected_stream="RT562",
    ),
    PhraseCase(
        name="alternating_a1b2c3",
        prompt="编号 A 一 B 二 C 三。",
        expected_stream="A1B2C3",
    ),
    PhraseCase(
        name="device_mn508",
        prompt="设备 M N 五零八。",
        expected_stream="MN508",
    ),
    PhraseCase(
        name="confusable_bdpt147",
        prompt="设备 B D P T 一四七。",
        expected_stream="BDPT147",
    ),
    PhraseCase(
        name="device_c1_off",
        prompt="确认设备 C 一的状态为 OFF。",
        expected_stream="C1OFF",
    ),
    PhraseCase(
        name="device_a2_on",
        prompt="确认设备 A 二的状态为 ON。",
        expected_stream="A2ON",
    ),
)


CHINESE_DIGIT_TRANSLATION = str.maketrans(
    {
        "零": "0",
        "〇": "0",
        "一": "1",
        "幺": "1",
        "二": "2",
        "两": "2",
        "兩": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
)

# State synonyms are normalized because the peer-checking comparison should
# treat a Chinese state word and its English equivalent as the same value.
STATE_REPLACEMENTS = (
    ("关闭", "OFF"),
    ("關閉", "OFF"),
    ("关", "OFF"),
    ("關", "OFF"),
    ("开启", "ON"),
    ("開啟", "ON"),
    ("打开", "ON"),
    ("打開", "ON"),
    ("开", "ON"),
    ("開", "ON"),
)


def parse_arguments():
    """Parse corpus-recording options."""

    parser = ArgumentParser(
        description=(
            "Record or benchmark fixed Chinese-English identifier WAVs."
        )
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Record missing corpus WAVs before benchmarking. Existing WAVs "
            "are preserved unless --overwrite is also supplied."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing corpus WAVs; requires --record.",
    )
    arguments = parser.parse_args()

    if arguments.overwrite and not arguments.record:
        parser.error("--overwrite requires --record")

    return arguments


def audio_path(phrase_case, repeat_number):
    """Return the stable WAV path for one phrase take."""

    return AUDIO_DIR / (
        f"{phrase_case.name}_run_{repeat_number}.wav"
    )


def save_wav(path, mono_audio, sample_rate):
    """Save mono float audio as 16-bit PCM without changing sample rate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(
        mono_audio * 32767,
        -32768,
        32767,
    ).astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def record_one_take(sample_rate, phrase_case, repeat_number):
    """Record one phrase after a retained silent pre-roll."""

    path = audio_path(phrase_case, repeat_number)

    print()
    print("=" * 78)
    print(f"Case:   {phrase_case.name}")
    print(f"Take:   {repeat_number}/{REPEATS_PER_CASE}")
    print(f"Say:    {phrase_case.prompt}")
    print(f"Target: {phrase_case.expected_stream}")
    input("Press Enter when ready...")

    # Open only after the user is ready. Leaving a blocking stream open while
    # waiting for keyboard input could accumulate stale, unread audio.
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
            int(PRE_ROLL_SECONDS * sample_rate)
        )

        print("\nSPEAK NOW\n")
        speech, overflow2 = stream.read(
            int(SPEECH_SECONDS * sample_rate)
        )
        print("DONE")

    mono_audio = np.concatenate([pre_roll, speech], axis=0)[:, 0]
    save_wav(path, mono_audio, sample_rate)

    print(f"Input overflow: {bool(overflow1 or overflow2)}")
    print(f"Saved: {path}")


def record_corpus(overwrite):
    """Record missing or explicitly replaceable fixed corpus files."""

    device = sd.query_devices(DEVICE_INDEX)
    host_api = sd.query_hostapis(device["hostapi"])
    sample_rate = int(device["default_samplerate"])

    print("Recording fixed mixed-identifier corpus")
    print(f"Device:      {DEVICE_INDEX} — {device['name']}")
    print(f"Host API:    {host_api['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels:    {CHANNELS}")

    sd.check_input_settings(
        device=DEVICE_INDEX,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )

    pending = [
        (phrase_case, repeat_number)
        for phrase_case in PHRASE_CASES
        for repeat_number in range(1, REPEATS_PER_CASE + 1)
        if overwrite
        or not audio_path(phrase_case, repeat_number).is_file()
    ]

    if not pending:
        print("All corpus WAVs already exist; nothing was overwritten.")
        return

    print(f"WAVs to record: {len(pending)}")

    for phrase_case, repeat_number in pending:
        record_one_take(
            sample_rate,
            phrase_case,
            repeat_number,
        )


def expected_audio_items():
    """Yield every required phrase, take number, and fixed WAV path."""

    for phrase_case in PHRASE_CASES:
        for repeat_number in range(1, REPEATS_PER_CASE + 1):
            yield (
                phrase_case,
                repeat_number,
                audio_path(phrase_case, repeat_number),
            )


def validate_corpus():
    """Require a complete fixed corpus before comparing models."""

    missing = [
        path
        for _, _, path in expected_audio_items()
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The fixed corpus is incomplete:\n"
            f"{formatted}\n"
            "Run with --record to create the missing WAV files."
        )


def normalize_semantics(text):
    """Normalize width, case, digits, and operational state synonyms."""

    normalized = unicodedata.normalize("NFKC", text).upper()

    for source, replacement in STATE_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)

    return normalized.translate(CHINESE_DIGIT_TRANSLATION)


def canonical_ascii_stream(text):
    """Extract the task-relevant ordered ASCII letter/digit stream."""

    normalized = normalize_semantics(text)
    return "".join(re.findall(r"[A-Z0-9]+", normalized))


def normalize_for_cer(text):
    """Normalize formatting while preserving the rest of the transcript."""

    normalized = normalize_semantics(text)
    return "".join(
        character
        for character in normalized
        if character.isalnum()
    )


def edit_distance(left, right):
    """Return the Levenshtein edit distance between two strings."""

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


def bounded_accuracy(expected, recognized):
    """Return edit accuracy in [0, 1], including insertion penalties."""

    if not expected:
        return 1.0 if not recognized else 0.0

    errors = edit_distance(expected, recognized)
    return max(0.0, 1 - errors / len(expected))


def character_error_rate(reference, transcript):
    """Calculate normalized full-transcript character error rate."""

    expected = normalize_for_cer(reference)
    recognized = normalize_for_cer(transcript)
    return edit_distance(expected, recognized) / max(1, len(expected))


def wav_duration_seconds(path):
    """Return the WAV duration without loading the waveform."""

    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def transcribe_once(model, path):
    """Transcribe one fixed WAV with the baseline decoding configuration."""

    started = perf_counter()
    segments, _ = model.transcribe(
        str(path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    # Inference is lazy and occurs while the segment generator is consumed.
    segments = list(segments)
    elapsed = perf_counter() - started
    transcript = "".join(
        segment.text for segment in segments
    ).strip()

    return transcript, elapsed


def make_result(
    phrase_case,
    repeat_number,
    path,
    model_size,
    load_seconds,
    transcript,
    inference_seconds,
):
    """Build one result row with task-level and transcription metrics."""

    expected_stream = phrase_case.expected_stream
    recognized_stream = canonical_ascii_stream(transcript)
    expected_letters = "".join(
        character
        for character in expected_stream
        if character.isalpha()
    )
    recognized_letters = "".join(
        character
        for character in recognized_stream
        if character.isalpha()
    )
    expected_digits = "".join(
        character
        for character in expected_stream
        if character.isdigit()
    )
    recognized_digits = "".join(
        character
        for character in recognized_stream
        if character.isdigit()
    )
    duration = wav_duration_seconds(path)

    return {
        "case": phrase_case.name,
        "take": repeat_number,
        "audio_path": str(path),
        "duration_seconds": duration,
        "model_size": model_size,
        "beam_size": BEAM_SIZE,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "real_time_factor": inference_seconds / max(duration, 1e-9),
        "field_exact": recognized_stream == expected_stream,
        "letter_accuracy": bounded_accuracy(
            expected_letters,
            recognized_letters,
        ),
        "digit_accuracy": bounded_accuracy(
            expected_digits,
            recognized_digits,
        ),
        "normalized_cer": character_error_rate(
            phrase_case.prompt,
            transcript,
        ),
        "expected_stream": expected_stream,
        "recognized_stream": recognized_stream,
        "reference": phrase_case.prompt,
        "transcript": transcript,
    }


def benchmark_models():
    """Run each model over the exact same fixed WAV corpus."""

    first_path = next(expected_audio_items())[2]
    results = []

    for model_size in MODEL_SIZES:
        print()
        print("=" * 100)
        print(f"Loading model: {model_size}")
        load_started = perf_counter()
        model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        load_seconds = perf_counter() - load_started
        print(f"Model loaded in {load_seconds:.2f} seconds")

        # Initialize inference before measuring any corpus item.
        print("Running one unmeasured warm-up...")
        transcribe_once(model, first_path)

        for phrase_case, repeat_number, path in expected_audio_items():
            print(
                f"Transcribing {phrase_case.name}, "
                f"take {repeat_number}, model {model_size}"
            )
            transcript, inference_seconds = transcribe_once(model, path)
            result = make_result(
                phrase_case=phrase_case,
                repeat_number=repeat_number,
                path=path,
                model_size=model_size,
                load_seconds=load_seconds,
                transcript=transcript,
                inference_seconds=inference_seconds,
            )
            results.append(result)

    return results


def print_results(results):
    """Print per-file results and one aggregate row per model."""

    print()
    print("=" * 112)
    print("PER-FILE RESULTS")
    print(
        f"{'Case':<25} {'Take':>4} {'Model':<7} {'Exact':>6} "
        f"{'Letters':>8} {'Digits':>8} {'Time':>8}  "
        "Expected -> Recognized"
    )
    print("-" * 112)

    for result in results:
        print(
            f"{result['case']:<25} "
            f"{result['take']:>4} "
            f"{result['model_size']:<7} "
            f"{str(result['field_exact']):>6} "
            f"{result['letter_accuracy']:>7.1%} "
            f"{result['digit_accuracy']:>7.1%} "
            f"{result['inference_seconds']:>7.2f}s  "
            f"{result['expected_stream']} -> "
            f"{result['recognized_stream'] or '[none]'}"
        )

    print()
    print("=" * 86)
    print("MODEL SUMMARY")
    print(
        f"{'Model':<8} {'Exact cases':>14} {'Letters':>10} "
        f"{'Digits':>10} {'Median time':>13} {'Median RTF':>11}"
    )
    print("-" * 86)

    for model_size in MODEL_SIZES:
        model_results = [
            result
            for result in results
            if result["model_size"] == model_size
        ]
        exact_count = sum(
            result["field_exact"] for result in model_results
        )

        print(
            f"{model_size:<8} "
            f"{exact_count:>5}/{len(model_results):<8} "
            f"{mean(result['letter_accuracy'] for result in model_results):>9.1%} "
            f"{mean(result['digit_accuracy'] for result in model_results):>9.1%} "
            f"{median(result['inference_seconds'] for result in model_results):>12.2f}s "
            f"{median(result['real_time_factor'] for result in model_results):>11.2f}"
        )

    print()
    print("FULL TRANSCRIPTS FOR NON-EXACT CASES")

    non_exact = [
        result for result in results if not result["field_exact"]
    ]

    if not non_exact:
        print("Every canonical field stream was exact.")
        return

    for result in non_exact:
        print("-" * 86)
        print(
            f"{result['case']} | take {result['take']} "
            f"| {result['model_size']}"
        )
        print(f"Reference:  {result['reference']}")
        print(f"Transcript: {result['transcript'] or '[no text]'}")


def save_results(results):
    """Save a timestamped CSV so earlier benchmark runs are preserved."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"results_{run_id}.csv"

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results: {output_path}")


def main():
    """Optionally capture the corpus, then run the fixed-audio benchmark."""

    arguments = parse_arguments()

    if arguments.record:
        record_corpus(overwrite=arguments.overwrite)

    validate_corpus()

    print()
    print("Mixed Chinese-English identifier benchmark")
    print(f"Models: {', '.join(MODEL_SIZES)}")
    print(f"Beam size: {BEAM_SIZE}")
    print("VAD: disabled")
    print(f"Phrase cases: {len(PHRASE_CASES)}")
    print(f"Takes per case: {REPEATS_PER_CASE}")
    print(
        "Primary metric: exact canonical operational field stream "
        "(for example, A 二 -> A2)"
    )

    results = benchmark_models()
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
