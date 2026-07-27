"""Compare Faster-Whisper model sizes and beam widths on fixed WAV files.

This experiment deliberately does not record from the microphone. Every model
and beam-search setting receives the exact same audio, so changes in accuracy
or latency come from inference settings rather than from a different take.
"""

from dataclasses import dataclass
import gc
from pathlib import Path
from statistics import median
from time import perf_counter
import csv
import unicodedata
import wave

from faster_whisper import WhisperModel


# Run this file from the repository root:
#     python experiments/model_size_beam_search.py
#
# Add more AudioCase entries when you have fixed recordings with known
# transcripts. Keep these WAV files unchanged between benchmark runs.
@dataclass(frozen=True)
class AudioCase:
    name: str
    path: Path
    reference: str


AUDIO_CASES = (
    AudioCase(
        name="leading_silence_speech_crop",
        path=Path("experiment_output")
        / "leading_silence_whisper"
        / "speech_crop.wav",
        reference="你好，这是录音开始测试，请完整识别开头的文字。",
    ),
)

# "medium" and "large-v3" can be added later, but they are substantially
# slower and more memory-intensive on a CPU than these initial candidates.
MODEL_SIZES = ("tiny", "base", "small")
BEAM_SIZES = (1, 5)

# Repeat only the measured inference. The median is less sensitive to an
# occasional operating-system scheduling delay than a single timing.
MEASURED_RUNS = 3

LANGUAGE = "zh"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
OUTPUT_CSV = (
    Path("experiment_output")
    / "model_size_beam_search"
    / "results.csv"
)


def normalize_text(text):
    """Remove spacing/punctuation while preserving letters and characters."""

    normalized = unicodedata.normalize("NFKC", text).upper()
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


def character_error_rate(reference, hypothesis):
    """Calculate CER after normalizing punctuation, width, case, and spacing."""

    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    errors = edit_distance(
        normalized_reference,
        normalized_hypothesis,
    )
    return errors / max(1, len(normalized_reference))


def wav_duration_seconds(path):
    """Read a PCM WAV duration without loading its samples into memory."""

    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def transcribe_once(model, audio_path, beam_size):
    """Transcribe one WAV and return its text and measured inference time."""

    started = perf_counter()
    segments, _ = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=beam_size,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    # Faster-Whisper returns a generator. Converting it to a list ensures the
    # timer includes actual model inference rather than only generator setup.
    segments = list(segments)
    elapsed = perf_counter() - started
    text = "".join(segment.text for segment in segments).strip()
    return text, elapsed


def validate_audio_cases():
    """Fail early with all missing benchmark inputs listed together."""

    missing_paths = [
        audio_case.path
        for audio_case in AUDIO_CASES
        if not audio_case.path.is_file()
    ]

    if missing_paths:
        formatted_paths = "\n".join(
            f"  - {path.resolve()}" for path in missing_paths
        )
        raise FileNotFoundError(
            "The following fixed WAV files are missing:\n"
            f"{formatted_paths}\n"
            "Run this script from the repository root."
        )


def benchmark_case(model, model_size, load_seconds, audio_case, beam_size):
    """Measure one model/beam/audio combination several times."""

    duration = wav_duration_seconds(audio_case.path)
    run_times = []
    transcripts = []

    for _ in range(MEASURED_RUNS):
        text, elapsed = transcribe_once(
            model,
            audio_case.path,
            beam_size,
        )
        transcripts.append(text)
        run_times.append(elapsed)

    median_seconds = median(run_times)
    transcript = transcripts[-1]

    if len(set(transcripts)) > 1:
        print(
            "Warning: repeated deterministic runs produced different text "
            f"for {model_size}, beam {beam_size}, {audio_case.name}."
        )

    return {
        "audio_case": audio_case.name,
        "audio_path": str(audio_case.path.resolve()),
        "duration_seconds": duration,
        "model_size": model_size,
        "beam_size": beam_size,
        "load_seconds": load_seconds,
        "median_inference_seconds": median_seconds,
        "real_time_factor": median_seconds / max(duration, 1e-9),
        "cer": character_error_rate(
            audio_case.reference,
            transcript,
        ),
        "reference": audio_case.reference,
        "transcript": transcript,
        "all_run_seconds": ", ".join(
            f"{elapsed:.4f}" for elapsed in run_times
        ),
    }


def print_results(results):
    """Print compact metrics followed by full reference/transcript pairs."""

    print()
    print("=" * 94)
    print("RESULTS")
    print(
        f"{'Audio':<29} {'Model':<9} {'Beam':>4} "
        f"{'Load':>8} {'Infer':>8} {'RTF':>7} {'CER':>8}"
    )
    print("-" * 94)

    for result in results:
        print(
            f"{result['audio_case']:<29} "
            f"{result['model_size']:<9} "
            f"{result['beam_size']:>4} "
            f"{result['load_seconds']:>7.2f}s "
            f"{result['median_inference_seconds']:>7.2f}s "
            f"{result['real_time_factor']:>7.2f} "
            f"{result['cer']:>7.1%}"
        )

    print()
    print("TRANSCRIPTS")

    for result in results:
        print("-" * 94)
        print(
            f"{result['audio_case']} | {result['model_size']} "
            f"| beam {result['beam_size']}"
        )
        print(f"Reference:  {result['reference']}")
        print(f"Transcript: {result['transcript'] or '[no text]'}")


def save_results(results):
    """Save full benchmark results for later comparisons."""

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
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
    print(f"Saved CSV: {OUTPUT_CSV.resolve()}")


def main():
    """Load each model once and benchmark every fixed audio/beam case."""

    validate_audio_cases()

    print("Fixed-audio Faster-Whisper benchmark")
    print(f"Models: {', '.join(MODEL_SIZES)}")
    print(
        "Beam sizes: "
        + ", ".join(str(beam_size) for beam_size in BEAM_SIZES)
    )
    print(f"Measured runs per case: {MEASURED_RUNS}")
    print(f"VAD: disabled")
    print()
    print(
        "The first use of an uncached model may download it. In that case, "
        "rerun the experiment before comparing model load times."
    )

    results = []

    for model_size in MODEL_SIZES:
        print()
        print(f"Loading model: {model_size}")
        load_started = perf_counter()
        model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        load_seconds = perf_counter() - load_started
        print(f"Model loaded in {load_seconds:.2f} seconds")

        # One unmeasured run initializes inference internals before any beam
        # setting is timed. Beam 1 keeps this warm-up relatively inexpensive.
        print("Running one unmeasured warm-up...")
        transcribe_once(
            model,
            AUDIO_CASES[0].path,
            beam_size=1,
        )

        for audio_case in AUDIO_CASES:
            for beam_size in BEAM_SIZES:
                print(
                    f"Benchmarking {audio_case.name}: "
                    f"{model_size}, beam {beam_size}"
                )
                result = benchmark_case(
                    model,
                    model_size,
                    load_seconds,
                    audio_case,
                    beam_size,
                )
                results.append(result)

        # Release the previous model before loading the next, larger model.
        del model
        gc.collect()

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
