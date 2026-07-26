"""Compare microphone onset and sustained-speech capture across Windows host APIs.
Input device check (run mic_device_check.py):
1 {'name': '麦克风阵列 (Realtek(R) Audio)', 'index': 1, 'hostapi': 0, 'max_input_channels': 4, 'max_output_channels': 0, 'default_low_input_latency': 0.09, 'default_low_output_latency': 0.09, 'default_high_input_latency': 0.18, 'default_high_output_latency': 0.18, 'default_samplerate': 44100.0} 

4 {'name': '主声音捕获驱动程序', 'index': 4, 'hostapi': 1, 'max_input_channels': 2, 'max_output_channels': 0, 'default_low_input_latency': 0.12, 'default_low_output_latency': 0.0, 'default_high_input_latency': 0.24, 'default_high_output_latency': 0.0, 'default_samplerate': 44100.0} 

9 {'name': '麦克风阵列 (Realtek(R) Audio)', 'index': 9, 'hostapi': 2, 'max_input_channels': 2, 'max_output_channels': 0, 'default_low_input_latency': 0.003, 'default_low_output_latency': 0.0, 'default_high_input_latency': 0.01, 'default_high_output_latency': 0.0, 'default_samplerate': 48000.0} 

18 {'name': '麦克风阵列 1 (Realtek HD Audio Mic input with SST)', 'index': 18, 'hostapi': 3, 'max_input_channels': 2, 'max_output_channels': 0, 'default_low_input_latency': 0.01, 'default_low_output_latency': 0.01, 'default_high_input_latency': 0.04, 'default_high_output_latency': 0.04, 'default_samplerate': 48000.0}
"""

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
import sounddevice as sd


# Record silence before the cue so microphone startup behavior and the
# pre-speech noise floor are both present in the saved file.
PRE_SECONDS = 1.5

# Five seconds is long enough to reveal a noise suppressor that initially
# admits a vowel and then incorrectly treats the steady sound as noise.
SPEECH_SECONDS = 5.0
TEST_PHRASE = "请持续发“啊”音，直到录音结束。"

# Keep the host-API comparison mono. This matches the intended ASR input and
# prevents different channel layouts from becoming an additional variable.
CHANNELS = 1

# Analyze short windows so brief onset loss and early signal dropout remain
# visible instead of being hidden inside one whole-recording average.
WINDOW_SECONDS = 0.05


@dataclass(frozen=True)
class Trial:
    """Describe one host-API/device path without opening the device yet."""

    label: str
    device_index: int
    wasapi_exclusive: bool = False


# These indices come from sd.query_devices() on this computer. Device 5 is the
# physical DirectSound Realtek endpoint; device 4 is only the generic primary
# capture driver. Devices 18-20 expose separate WDM-KS microphone-array paths.
TRIALS = (
    Trial("mme", 1),
    Trial("directsound", 5),
    Trial("wasapi_shared", 9),
    Trial("wasapi_exclusive", 9, wasapi_exclusive=True),
    Trial("wdm_ks_array_1", 18),
    Trial("wdm_ks_array_2", 19),
    Trial("wdm_ks_array_3", 20),
)


def dbfs_rms(samples):
    """Return RMS amplitude in dBFS, where 0 dBFS is the digital maximum."""

    samples = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples * samples))

    # The floor prevents log10(0) for completely silent windows.
    return 20 * np.log10(max(rms, 1e-9))


def save_wav(path, audio, sample_rate):
    """Save the first captured channel as a mono 16-bit PCM WAV file."""

    mono = audio[:, 0]

    # SoundDevice supplies float samples near [-1, 1]; WAV output uses signed
    # little-endian 16-bit integers.
    pcm = np.clip(mono * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def find_active_runs(active_windows):
    """Convert a Boolean activity mask into half-open (start, end) runs."""

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


def analyze(audio, sample_rate):
    """Measure level, onset, activity duration, and premature signal loss."""

    mono = audio[:, 0]

    # Exclude the final 200 ms of pre-roll in case the user anticipates the cue.
    noise_end = int((PRE_SECONDS - 0.2) * sample_rate)
    noise_db = dbfs_rms(mono[:noise_end])
    total_db = dbfs_rms(mono)

    peak = float(np.max(np.abs(mono)))
    peak_db = 20 * np.log10(max(peak, 1e-9))
    clipping_percent = 100 * np.mean(np.abs(mono) >= 0.99)

    # A high exact-zero percentage can indicate a digital noise gate rather
    # than a naturally quiet acoustic background.
    exact_zero_percent = 100 * np.mean(mono == 0)

    # Convert the waveform into a time series of short-window RMS levels.
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

    # A sustained vowel should remain well above both the measured noise floor
    # and a conservative absolute speech threshold.
    activity_threshold = max(noise_db + 20, -60)
    active_windows = window_levels >= activity_threshold
    active_runs = find_active_runs(active_windows)

    if active_runs:
        # For a properly captured sustained vowel, the longest run and the
        # total active time should both approach SPEECH_SECONDS.
        first_active = active_runs[0][0] * WINDOW_SECONDS
        last_active = active_runs[-1][1] * WINDOW_SECONDS
        total_active = float(np.sum(active_windows) * WINDOW_SECONDS)
        longest_run = max(
            (end - start) * WINDOW_SECONDS
            for start, end in active_runs
        )
    else:
        first_active = None
        last_active = None
        total_active = 0.0
        longest_run = 0.0

    print(f"Noise floor:             {noise_db:7.1f} dBFS")
    print(f"Whole-record RMS:        {total_db:7.1f} dBFS")
    print(f"Peak:                    {peak_db:7.1f} dBFS")
    print(f"Clipping:                {clipping_percent:7.3f}%")
    print(f"Exact-zero samples:      {exact_zero_percent:7.2f}%")
    print(f"Activity threshold:      {activity_threshold:7.1f} dBFS")
    print(f"First active time:       {first_active} seconds")
    print(f"Last active time:        {last_active} seconds")
    print(f"Total active time:       {total_active:.2f} seconds")
    print(f"Longest continuous run: {longest_run:.2f} seconds")
    print(f"Expected cue time:       {PRE_SECONDS} seconds")

    # Allow a 500 ms margin for human timing. Earlier disappearance is strong
    # evidence that capture-side processing suppressed the sustained sound.
    expected_latest_activity = PRE_SECONDS + SPEECH_SECONDS - 0.5
    if last_active is not None and last_active < expected_latest_activity:
        print("WARNING: The sustained signal disappeared before recording ended.")


def get_extra_settings(trial):
    """Request WASAPI exclusive mode only for its dedicated comparison."""

    if trial.wasapi_exclusive:
        # Exclusive mode bypasses the Windows software mixer. Driver- or
        # hardware-level microphone processing may still remain active.
        return sd.WasapiSettings(exclusive=True)
    return None


def run_trial(trial):
    """Record and analyze one explicitly selected input-device path."""

    device = sd.query_devices(trial.device_index)
    host_api = sd.query_hostapis(device["hostapi"])

    # Use the endpoint's advertised native/default rate so OS resampling does
    # not confound the host-API comparison.
    sample_rate = int(device["default_samplerate"])
    extra_settings = get_extra_settings(trial)

    # Include all relevant settings in the filename so trials never overwrite
    # one another and remain identifiable during listening tests.
    output = Path(
        f"onset_{trial.label}_device_{trial.device_index}_{sample_rate}.wav"
    )

    print()
    print("=" * 78)
    print(f"Trial:       {trial.label}")
    print(f"Device:      {trial.device_index} — {device['name']}")
    print(f"Host API:    {host_api['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels:    {CHANNELS}")
    print(f"Target:      {TEST_PHRASE}")

    # Fail before prompting the user if this endpoint does not support the
    # requested rate, channel count, sample format, or exclusive mode.
    sd.check_input_settings(
        device=trial.device_index,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        extra_settings=extra_settings,
    )

    with sd.InputStream(
        device=trial.device_index,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        blocksize=0,
        latency="high",
        extra_settings=extra_settings,
    ) as stream:
        # Opening the stream before the cue prevents PortAudio startup from
        # cutting off the user's first phoneme.
        print("Microphone stream is active. Remain silent...")

        # Keep this pre-roll instead of discarding it; it is needed for
        # measuring the true noise floor and any digital gating.
        pre_roll, overflow1 = stream.read(
            int(PRE_SECONDS * sample_rate)
        )

        print("\nSPEAK NOW — sustain “啊” until recording finishes.\n")

        speech, overflow2 = stream.read(
            int(SPEECH_SECONDS * sample_rate)
        )

    # Preserve the original timing: pre-roll begins at t=0 and the speech cue
    # occurs at approximately PRE_SECONDS.
    audio = np.concatenate([pre_roll, speech], axis=0)

    save_wav(output, audio, sample_rate)
    analyze(audio, sample_rate)

    print(f"Input overflow: {overflow1 or overflow2}")
    print(f"Saved: {output.resolve()}")


def main():
    print("This experiment compares the same Realtek microphone through")
    print("MME, DirectSound, WASAPI, and the three WDM-KS array endpoints.")

    for trial in TRIALS:
        # Interactive confirmation gives the user time to prepare and makes it
        # possible to skip unsupported or temporarily busy endpoints.
        answer = input(
            f"\nPress Enter to run {trial.label}, or type S to skip: "
        )

        if answer.strip().lower() == "s":
            print(f"Skipped: {trial.label}")
            continue

        try:
            run_trial(trial)
        except Exception as error:
            print(f"Trial failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
