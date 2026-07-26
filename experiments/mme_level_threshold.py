"""Measure how reliably the Realtek MME input captures different voice levels."""

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd


# Device 1 was the only endpoint that produced a strong, stable recording in
# the host-API comparison.
DEVICE_INDEX = 1
CHANNELS = 1

# The silent pre-roll measures the noise floor and ensures the stream is fully
# active before the user starts speaking.
PRE_SECONDS = 1.5
SPEECH_SECONDS = 5.0
WINDOW_SECONDS = 0.05

# Repeated trials reduce the chance of treating one unusually loud or quiet
# performance as a device characteristic.
REPEATS_PER_LEVEL = 3
LEVELS = (
    (
        "quiet",
        "Speak softly, but keep the sustained “啊” clearly voiced.",
    ),
    (
        "normal",
        "Use your normal conversational volume.",
    ),
    (
        "loud",
        "Speak loudly without shouting or moving closer to the microphone.",
    ),
)

OUTPUT_DIR = Path("experiment_output") / "mme_level_threshold"


@dataclass(frozen=True)
class TrialResult:
    """Store measurements needed for the final cross-trial comparison."""

    level: str
    run_number: int
    output_path: Path
    noise_db: float
    speech_rms_db: float
    peak_db: float
    onset_delay: float | None
    initial_level_db: float | None
    steady_level_db: float | None
    gain_change_db: float | None
    longest_run: float
    dropout_before_end: float | None
    continuity_ratio: float
    speech_zero_percent: float
    overflowed: bool


def dbfs_rms(samples):
    """Return RMS amplitude in dBFS, where 0 dBFS is digital full scale."""

    samples = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples * samples))
    return 20 * np.log10(max(rms, 1e-9))


def save_wav(path, audio, sample_rate):
    """Save the first input channel as a mono 16-bit PCM WAV file."""

    mono = audio[:, 0]
    pcm = np.clip(mono * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def bridge_short_gaps(active_windows, maximum_gap_windows=2):
    """Treat threshold dips of at most 100 ms as part of one voiced run."""

    bridged = active_windows.copy()
    index = 0

    while index < len(bridged):
        if bridged[index]:
            index += 1
            continue

        gap_start = index
        while index < len(bridged) and not bridged[index]:
            index += 1

        gap_end = index
        gap_size = gap_end - gap_start
        has_activity_before = gap_start > 0 and bridged[gap_start - 1]
        has_activity_after = gap_end < len(bridged) and bridged[gap_end]

        if (
            gap_size <= maximum_gap_windows
            and has_activity_before
            and has_activity_after
        ):
            bridged[gap_start:gap_end] = True

    return bridged


def find_active_runs(active_windows):
    """Return half-open (start, end) indices for continuous active regions."""

    runs = []
    start = None

    for index, is_active in enumerate(active_windows):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            runs.append((start, index))
            start = None

    if start is not None:
        runs.append((start, len(active_windows)))

    return runs


def median_level(window_levels, start, end):
    """Return a robust median level for a window range, if it has data."""

    selected = window_levels[start:end]
    if len(selected) == 0:
        return None
    return float(np.median(selected))


def analyze(audio, sample_rate, level, run_number, output_path, overflowed):
    """Analyze activation, initial gain, continuity, and early dropout."""

    mono = audio[:, 0]
    total_seconds = PRE_SECONDS + SPEECH_SECONDS
    window_samples = max(1, int(WINDOW_SECONDS * sample_rate))

    window_levels = np.array(
        [
            dbfs_rms(mono[start : start + window_samples])
            for start in range(
                0,
                len(mono) - window_samples + 1,
                window_samples,
            )
        ]
    )

    # Ignore the first 500 ms because some host paths produce a stream-start
    # transient. Also leave 200 ms before the cue as a human-timing margin.
    noise_start_window = int(0.5 / WINDOW_SECONDS)
    noise_end_window = int((PRE_SECONDS - 0.2) / WINDOW_SECONDS)
    noise_db = median_level(
        window_levels,
        noise_start_window,
        noise_end_window,
    )

    if noise_db is None:
        raise RuntimeError("Not enough pre-roll audio to estimate noise.")

    # Ten dB over the measured noise rejects the background while remaining
    # sensitive to quiet speech. The -65 dBFS floor prevents tiny numerical
    # noise from being counted as a sustained vowel.
    activity_threshold = max(noise_db + 10, -65)
    cue_window = int(PRE_SECONDS / WINDOW_SECONDS)

    post_cue_active = window_levels[cue_window:] >= activity_threshold
    post_cue_active = bridge_short_gaps(post_cue_active)
    active_runs = find_active_runs(post_cue_active)

    peak = float(np.max(np.abs(mono)))
    peak_db = 20 * np.log10(max(peak, 1e-9))

    post_cue_samples = mono[int(PRE_SECONDS * sample_rate) :]
    speech_zero_percent = 100 * np.mean(post_cue_samples == 0)

    if active_runs:
        first_start, _ = active_runs[0]
        longest_start, longest_end = max(
            active_runs,
            key=lambda run: run[1] - run[0],
        )

        first_active_window = cue_window + first_start
        last_active_window = cue_window + active_runs[-1][1]

        first_active_time = first_active_window * WINDOW_SECONDS
        last_active_time = last_active_window * WINDOW_SECONDS
        onset_delay = max(0.0, first_active_time - PRE_SECONDS)
        longest_run = (longest_end - longest_start) * WINDOW_SECONDS
        dropout_before_end = max(0.0, total_seconds - last_active_time)

        # Compare the first 250 ms after activation with a later 750 ms region.
        # A large positive change indicates a gain ramp after speech begins.
        initial_level_db = median_level(
            window_levels,
            first_active_window,
            first_active_window + 5,
        )
        steady_level_db = median_level(
            window_levels,
            first_active_window + 15,
            first_active_window + 30,
        )

        if initial_level_db is not None and steady_level_db is not None:
            gain_change_db = steady_level_db - initial_level_db
        else:
            gain_change_db = None

        expected_activity_after_onset = max(
            WINDOW_SECONDS,
            total_seconds - first_active_time,
        )
        continuity_ratio = min(
            1.0,
            longest_run / expected_activity_after_onset,
        )

        active_start_sample = int(first_active_time * sample_rate)
        speech_rms_db = dbfs_rms(mono[active_start_sample:])
    else:
        onset_delay = None
        initial_level_db = None
        steady_level_db = None
        gain_change_db = None
        longest_run = 0.0
        dropout_before_end = None
        continuity_ratio = 0.0
        speech_rms_db = dbfs_rms(post_cue_samples)

    result = TrialResult(
        level=level,
        run_number=run_number,
        output_path=output_path,
        noise_db=noise_db,
        speech_rms_db=speech_rms_db,
        peak_db=peak_db,
        onset_delay=onset_delay,
        initial_level_db=initial_level_db,
        steady_level_db=steady_level_db,
        gain_change_db=gain_change_db,
        longest_run=longest_run,
        dropout_before_end=dropout_before_end,
        continuity_ratio=continuity_ratio,
        speech_zero_percent=speech_zero_percent,
        overflowed=overflowed,
    )

    print(f"Noise floor:              {result.noise_db:7.1f} dBFS")
    print(f"Speech RMS after onset:   {result.speech_rms_db:7.1f} dBFS")
    print(f"Peak:                     {result.peak_db:7.1f} dBFS")
    print(f"Onset delay after cue:    {format_optional(result.onset_delay, 's')}")
    print(f"Initial 250 ms level:     {format_optional(result.initial_level_db, 'dBFS')}")
    print(f"Later steady level:       {format_optional(result.steady_level_db, 'dBFS')}")
    print(f"Gain change:              {format_optional(result.gain_change_db, 'dB')}")
    print(f"Longest continuous run:  {result.longest_run:7.2f} s")
    print(f"Dropout before end:       {format_optional(result.dropout_before_end, 's')}")
    print(f"Continuity:               {result.continuity_ratio:7.1%}")
    print(f"Zero samples after cue:   {result.speech_zero_percent:7.2f}%")
    print(f"Input overflow:           {result.overflowed}")

    if result.continuity_ratio < 0.9:
        print("WARNING: The vowel was not continuously preserved.")

    if (
        result.gain_change_db is not None
        and result.gain_change_db >= 6
    ):
        print("WARNING: The signal became at least 6 dB stronger after onset.")

    return result


def format_optional(value, unit):
    """Format an optional numeric result without special-case print logic."""

    if value is None:
        return "not detected"
    return f"{value:7.2f} {unit}"


def run_trial(level, instruction, run_number, sample_rate):
    """Record one sustained vowel at the requested subjective voice level."""

    output_path = OUTPUT_DIR / (
        f"mme_{level}_run_{run_number}_{sample_rate}.wav"
    )

    print()
    print("=" * 76)
    print(f"Level:       {level}")
    print(f"Run:         {run_number}/{REPEATS_PER_LEVEL}")
    print(f"Instruction: {instruction}")
    print(f"Output:      {output_path}")

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

        print("\nSPEAK NOW — sustain “啊” until DONE appears.\n")
        speech, overflow2 = stream.read(
            int(SPEECH_SECONDS * sample_rate)
        )
        print("DONE")

    audio = np.concatenate([pre_roll, speech], axis=0)
    save_wav(output_path, audio, sample_rate)

    return analyze(
        audio=audio,
        sample_rate=sample_rate,
        level=level,
        run_number=run_number,
        output_path=output_path,
        overflowed=bool(overflow1 or overflow2),
    )


def print_summary(results):
    """Print one compact row per trial for easy level comparison."""

    print()
    print("=" * 110)
    print("SUMMARY")
    print(
        f"{'Level':<8} {'Run':>3} {'RMS':>8} {'Peak':>8} "
        f"{'Onset':>8} {'Gain':>8} {'Longest':>9} "
        f"{'Dropout':>9} {'Continuity':>11} {'Zeros':>8}"
    )
    print("-" * 110)

    for result in results:
        onset = (
            f"{result.onset_delay:.2f}"
            if result.onset_delay is not None
            else "N/A"
        )
        gain = (
            f"{result.gain_change_db:+.1f}"
            if result.gain_change_db is not None
            else "N/A"
        )
        dropout = (
            f"{result.dropout_before_end:.2f}"
            if result.dropout_before_end is not None
            else "N/A"
        )

        print(
            f"{result.level:<8} "
            f"{result.run_number:>3} "
            f"{result.speech_rms_db:>7.1f} "
            f"{result.peak_db:>7.1f} "
            f"{onset:>8} "
            f"{gain:>8} "
            f"{result.longest_run:>9.2f} "
            f"{dropout:>9} "
            f"{result.continuity_ratio:>10.1%} "
            f"{result.speech_zero_percent:>7.2f}%"
        )

    print()
    print("A reliable level should end near the recording boundary, maintain")
    print("at least 90% continuity, and avoid a large positive gain change.")


def main():
    """Run quiet, normal, and loud trials through the verified MME path."""

    device = sd.query_devices(DEVICE_INDEX)
    host_api = sd.query_hostapis(device["hostapi"])
    sample_rate = int(device["default_samplerate"])

    print(f"Device:      {DEVICE_INDEX} — {device['name']}")
    print(f"Host API:    {host_api['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels:    {CHANNELS}")
    print()
    print("Keep the same distance and body position for every trial.")
    print("Sustain only “啊” from SPEAK NOW until DONE.")

    sd.check_input_settings(
        device=DEVICE_INDEX,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for level, instruction in LEVELS:
        for run_number in range(1, REPEATS_PER_LEVEL + 1):
            try:
                result = run_trial(
                    level=level,
                    instruction=instruction,
                    run_number=run_number,
                    sample_rate=sample_rate,
                )
                results.append(result)
            except Exception as error:
                print(
                    f"Trial failed: {type(error).__name__}: {error}"
                )

    if results:
        print_summary(results)
    else:
        print("No trials completed successfully.")


if __name__ == "__main__":
    main()
