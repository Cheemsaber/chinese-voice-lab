# Chinese Voice Lab

## Experiment review and achievements

The experiments have separated the original speech-recognition symptoms into
three areas: microphone capture, Whisper decoding, and runtime performance.
The most important finding is that the original first-phoneme problem was
primarily associated with the input path, rather than simply being caused by
Faster-Whisper.

## Experiment summary

| Experiment | Main achievement | Conclusion |
| --- | --- | --- |
| Host API and onset capture | Identified a usable microphone route | Explicit Realtek MME device 1 is the reliable tested path |
| MME level threshold | Measured sensitivity to speaking volume | Loud speech is preserved reliably; quiet speech is unreliable |
| Leading-silence Whisper | Separated captured audio from VAD and padding | Additional digital silence does not consistently improve recognition |
| Model size and beam search | Quantified accuracy and latency trade-offs | Beam 1 is sufficient; `base` is fast and `small` is most accurate |

## 1. Host API and sustained-vowel capture

Experiment: [`experiments/capture_onset.py`](experiments/capture_onset.py)

This experiment compared the same Realtek microphone through different
Windows audio interfaces.

Observed results:

- MME device 1 captured meaningful speech.
- DirectSound device 5 produced an all-zero WAV.
- WASAPI shared and exclusive did not preserve the sustained vowel correctly
  in listening tests.
- None of the recordings overflowed.
- Opening the stream before the cue and retaining pre-roll prevented PortAudio
  startup from simply discarding the beginning.

This showed that different device indices did not provide equivalent access
to the same physical microphone.

The resulting capture configuration was:

```text
Device:      1
Host API:    MME
Sample rate: 44100 Hz
Channels:    1
```

The experiment defines three WDM-KS trials, but no corresponding output WAVs
currently exist. Therefore, MME was the best option among the completed MME,
DirectSound, and WASAPI trials, rather than necessarily every possible Windows
endpoint.

## 2. MME speech-level experiment

Experiment:
[`experiments/mme_level_threshold.py`](experiments/mme_level_threshold.py)

Saved report:
[`experiment_output/mme_level_threshold/mme_level_test_result.txt`](experiment_output/mme_level_threshold/mme_level_test_result.txt)

This experiment tested nine sustained-vowel recordings: three quiet, three
normal, and three loud.

| Level | Reliable continuous runs |
| --- | ---: |
| Quiet | 0/3 |
| Normal | 1/3 clearly reliable; one nearly reliable |
| Loud | 3/3 |

For loud speech:

- Continuity was 100% in all three trials.
- The signal remained active until the recording boundary.
- Peaks were approximately -13 to -15 dBFS.
- There was no clipping or input overflow.
- Gain changes were small.

This establishes that MME can preserve sustained speech correctly when the
signal is strong enough. It also shows that the earlier shortened vowel was
not something Faster-Whisper did: the problem was already present or absent
inside the WAV before transcription.

The quiet results suggest some combination of:

- low signal-to-noise ratio;
- Realtek noise suppression;
- automatic microphone processing;
- the experiment's energy threshold;
- difficulty sustaining an unusually quiet vowel consistently.

The observed initial volume ramp also exists in the captured waveform, so
Faster-Whisper cannot be its cause. Microphone or driver processing is the more
plausible source.

These results do not mean operators must speak loudly. Normal operational
phrases contain changing phonemes and may survive microphone processing better
than a perfectly sustained vowel. A realistic speech corpus will provide more
representative evidence.

## 3. Leading-silence and VAD experiment

Experiment:
[`experiments/leading_silence_whisper.py`](experiments/leading_silence_whisper.py)

The experiment used the following controlled procedure:

1. Record the phrase once.
2. Locate and preserve the first phoneme.
3. Create identical copies with 0, 250, 500, and 1000 ms of digital silence.
4. Disable Faster-Whisper VAD explicitly.
5. Transcribe every copy with the same model settings.

No padding duration produced consistently better output. For example, 1000 ms
was exact in one run but produced errors in another.

Therefore, adding leading silence is not a reliable decoder-level solution.

Listening tests found that the first phoneme in the MME recording was already
clearer. This indicates that explicitly selecting the working MME route and
retaining pre-roll improved capture. Whisper then received a complete phoneme.

The experiment also eliminated one incorrect interpretation:
`vad_filter=False` did not repair the microphone. It ensured only that
Faster-Whisper's Silero VAD was not an additional variable.

The result table is not currently saved to a CSV or text file, so only the WAV
variants remain. Future experiments should persist every result automatically.

## 4. Model size and beam-search benchmark

Experiment:
[`experiments/model_size_beam_search.py`](experiments/model_size_beam_search.py)

Latest results:
[`experiment_output/model_size_beam_search/results.csv`](experiment_output/model_size_beam_search/results.csv)

This experiment used the same 5.33-second WAV for every configuration, warmed
each model, and measured three inference runs.

### Stable cached results

| Model | Beam 1 inference | Result |
| --- | ---: | --- |
| `tiny` | 0.27 s | Traditional output plus a recognition error |
| `base` | 0.47 s | One-character error |
| `small` | 1.34 s | Exact transcription |

Beam 5:

- produced identical text for every model;
- added approximately 7-11% latency;
- demonstrated no benefit on this recording.

The supported beam-search choice is:

```text
beam_size=1
```

The 1253-second initial `small` load was confirmed to be a first-time download.
Its cached load later fell to 1.12 seconds.

All three models ran faster than real time:

```text
tiny RTF:  0.05
base RTF:  0.09
small RTF: 0.25
```

This proves the computer has enough inference performance for short
post-utterance recognition.

The current model conclusions are:

- `base`: best responsiveness and accuracy balance;
- `small`: best observed accuracy and likely preferable for safety-relevant
  field comparison;
- `tiny`: not sufficiently reliable yet;
- beam 5: unnecessary based on current evidence.

This benchmark contains only one speaker and one WAV. Tiny's reported 50% CER
is also exaggerated by Traditional-versus-Simplified character differences.

## Status of the original problems

### First characters poorly recognized

**Mostly narrowed down, with a practical capture configuration identified.**

Achievements:

- Explicit MME device 1 works.
- Native 44.1 kHz avoids an unnecessary capture-side conversion.
- Opening the stream early and retaining pre-roll protects the first phoneme.
- Digital silence padding is not a stable recognition fix.
- Faster-Whisper VAD was ruled out as the primary cause in this experiment.

Quiet speech and microphone processing can still weaken the signal.

### Chinese numbers mixed with English letters

**Not tested yet.**

This remains the clearest next experiment, now tailored to real equipment
identifiers, values, states, and actions.

### High latency and repeated model loading

**Diagnosed, but not yet fixed in the application.**

The benchmark proves inference itself is fast enough. The repeated delay comes
largely from application structure.

The current [`main.py`](main.py) still:

- uses the unspecified default input device;
- requests 16 kHz directly instead of the verified native MME configuration;
- loads `WhisperModel` inside `recognize_audio()` every time;
- waits for a fixed five-second recording;
- plays the recording before recognition.

The experimental findings have therefore not yet been integrated into the
main program.

## Overall achievement

The project has moved from a general observation that the beginning sounds
vague and Whisper is slow to the following evidence-based understanding:

1. Windows microphone routes behave differently.
2. MME device 1 is the verified capture route.
3. Capture quality depends strongly on input level.
4. The first phoneme can be preserved before Whisper receives it.
5. Arbitrary leading silence does not stabilize transcription.
6. Beam 5 is slower without a demonstrated accuracy benefit.
7. `base` and `small` are both comfortably faster than real time.
8. Repeated model loading is an application-design issue.
9. The next unresolved technical risk is recognizing operational identifiers
   containing Chinese numbers and English letters.

## Next experiment

The next benchmark should use realistic peer-checking phrases containing:

- equipment identifiers such as `A2` and `B307`;
- Chinese numbers mixed with English letters;
- actions such as connect, disconnect, open, and close;
- states such as `ON` and `OFF`;
- measured values and units.

It should compare `base` and `small` with `beam_size=1` on the same fixed WAV
files and evaluate both raw transcription and normalized structured fields.

## Full-dataset LoRA training

The repository now includes a detailed, configuration-driven guide for moving
from the bounded Whisper Base smoke test to full-dataset LoRA experiments:

- [`FULL_DATA_LORA_TRAINING_GUIDE.md`](FULL_DATA_LORA_TRAINING_GUIDE.md)

The guide keeps the existing smoke test intact and proposes a separate training
path with:

- one YAML configuration per experiment;
- model switching through a single `model.id` value;
- editable LoRA rank, alpha, dropout, target modules, and bias settings;
- editable Hugging Face `Seq2SeqTrainingArguments`;
- correct use of all 49 training, 8 validation, and 8 locked test records;
- generated validation metrics and best-checkpoint selection;
- offline model caching, checkpoint resume, TensorBoard, and reproducibility
  metadata;
- a Base-first experiment sequence before trying Whisper Small or Large-v3.

Start with Whisper Base and the existing cached model. Advance to a larger model
only when validation and locked-test evidence shows that Base is insufficient.
