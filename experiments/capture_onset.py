"""Test microphone onset, gain and native sample rate."""

from pathlib import Path
import wave
import numpy as np
import sounddevice as sd


PRE_SECONDS = 1.5 # open the microphone before asking the user to speak
SPEECH_SECONDS = 5.0
TEST_PHRASE = "你好，这是录音开始测试，设备编号 A B C 一二三。"


def dbfs_rms(samples):
    samples = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples * samples))
    return 20 * np.log10(max(rms, 1e-9)) # avoid taking lg(0)


def save_wav(path, audio, sample_rate):
    mono = audio[:, 0] # select all samples from channel 1
    pcm = np.clip(mono * 32767, -32768, 32767).astype("<i2") # converting float32 samples to 16-bit PCM
    # converting each sample to little-endian signed 16-bit integer

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def analyze(audio, sample_rate):
    mono = audio[:, 0]

    noise_end = int((PRE_SECONDS - 0.2) * sample_rate) # give 200 msec margin
    noise_db = dbfs_rms(mono[:noise_end])
    total_db = dbfs_rms(mono)

    peak = float(np.max(np.abs(mono)))
    peak_db = 20 * np.log10(max(peak, 1e-9))
    clipping_percent = 100 * np.mean(np.abs(mono) >= 0.99)

    window_samples = int(0.05 * sample_rate) # divide samples into 50 msec windows
    window_levels = []

    for start in range(0, len(mono) - window_samples + 1, window_samples):
        window = mono[start : start + window_samples]
        window_levels.append(dbfs_rms(window))

    # require a substantial rise above the measured pre-noise.
    onset_threshold = max(noise_db + 12, -42) # values adjustable depending on ambient noise level
    onset_window = next(
        (
            index
            for index, level in enumerate(window_levels) # create (index, level) tuple
            if level >= onset_threshold # only select windows louder than the onset threshold
        ),
        None,
    )

    onset_seconds = (
        onset_window * 0.05 if onset_window is not None else None
    )

    print(f"Noise floor:       {noise_db:7.1f} dBFS")
    print(f"Whole-record RMS:  {total_db:7.1f} dBFS")
    print(f"Peak:              {peak_db:7.1f} dBFS")
    print(f"Clipping:          {clipping_percent:7.3f}%")
    print(f"Onset threshold:   {onset_threshold:7.1f} dBFS")
    print(f"Detected onset:    {onset_seconds} seconds")
    print(f"Expected cue time: {PRE_SECONDS} seconds")


def run_trial(sample_rate):
    output = Path(f"onset_test_{sample_rate}.wav")

    print()
    print("=" * 70)
    print(f"Testing at {sample_rate} Hz")
    print(f"Target phrase: {TEST_PHRASE}")

    sd.check_input_settings(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=0,
        latency="high",
    ) as stream:
        print("Microphone stream is active. Remain silent...")

        pre_roll, overflow1 = stream.read(
            int(PRE_SECONDS * sample_rate)
        )

        print("\nSPEAK NOW\n")

        speech, overflow2 = stream.read(
            int(SPEECH_SECONDS * sample_rate)
        )

    audio = np.concatenate([pre_roll, speech], axis=0)

    save_wav(output, audio, sample_rate)
    analyze(audio, sample_rate)

    print(f"Input overflow: {overflow1 or overflow2}")
    print(f"Saved: {output.resolve()}")


def main():
    device = sd.query_devices(kind="input")
    native_rate = int(device["default_samplerate"])

    print(f"Input device: {device['name']}")
    print(f"Native rate:  {native_rate} Hz")

    rates = list(dict.fromkeys([16_000, native_rate]))

    for sample_rate in rates:
        input(f"\nPress Enter to start the {sample_rate}-Hz trial...")
        try:
            run_trial(sample_rate)
        except Exception as error:
            print(f"Trial failed: {error}")


if __name__ == "__main__":
    main()