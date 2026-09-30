<p align="center"><img src="docs/assets/gigastt-pyannote-transcriber-hero.png" alt="gigastt-pyannote-transcriber: two people talk, a speech-to-text stenographer writes down words with timestamps, a speaker tagger assigns each word to a speaker, and a who-said-what transcript comes out" width="100%"></p>
<h1 align="center">gigastt-pyannote-transcriber</h1>
<p align="center"><b>Russian recording in, who-said-what transcript out — a local, CPU-only pipeline (GigaSTT + pyannote) for multi-speaker conversations you will have to quote and check by ear.</b></p>
<p align="center">
  <a href="pyproject.toml"><img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white"></a>
  <a href="#requirements"><img alt="Platform: Windows 10/11 x64" src="https://img.shields.io/badge/Platform-Windows%2010%2F11%20x64-0078D6.svg?logo=windows&logoColor=white"></a>
  <a href="#requirements"><img alt="CPU only" src="https://img.shields.io/badge/runs%20on-CPU%20only-2E7D57.svg"></a>
  <a href="pyproject.toml"><img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-blue.svg"></a>
  <a href="https://github.com/AndrewMoryakov/gigastt-pyannote-transcriber/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/AndrewMoryakov/gigastt-pyannote-transcriber/actions/workflows/tests.yml/badge.svg"></a>
</p>
<p align="center"><b>English</b> | <a href="README.ru.md">Русский</a></p>

```powershell
.\scripts\run.ps1 -InputAudio '.\media\recording.m4a'   # -> transcript.{md,txt,srt,vtt,json}
```

**Fourvoices: GigaSTT + pyannote for four voices.** A local, CPU-only pipeline for Russian-language recordings: **GigaSTT 2.15.0 / GigaAM v3 RNNT** recognises the words and their timings, **pyannote Community-1** says who spoke when, and the `fourvoices` Python package aligns the two in time. (The package and CLI are called `fourvoices`; the repository is `gigastt-pyannote-transcriber`.) Source audio, models, tokens and results are deliberately kept out of Git.

## Why gigastt-pyannote-transcriber?

- **If** you have a Russian-language recording of several people talking (an interview, a meeting, a call) and need to know **who said what**, **then** this turns one audio file into a transcript with a speaker label on every turn.
- **If** you want it on your own machine with no GPU, **then** it is a CPU-only pipeline for Windows 10/11 x64; any CPU works, a slower one is simply slower.
- **If** you will have to quote the result and check it by ear, **then** it keeps real word timings, flags overlapping speech and uncertain attribution instead of hiding them, never guesses people's names, and pins the GigaSTT executable by URL + SHA-256 and the pyannote model by git revision.
- **If** re-running should not waste hours, **then** finished stages are resumed from disk and changed settings stop the run instead of being silently mixed.

**Honest caveats.** It is built around four-person conversations on a single microphone (`num_speakers: 4` by default; `-NumSpeakers 3` and so on to change). It needs a **Hugging Face account and a read token** — the pyannote model is gated, and `HF_TOKEN` is required for every transcription run, not only for downloading. The first install needs the network and roughly 10 GB. It is not a hosted service and not a GPU pipeline (the pinned PyTorch build is CPU-only). Automatic transcription is not a guarantee of verbatim accuracy — see the closing note below.

The transcripts themselves are Russian, so the markers inside them (`перекрытие речи`, `спикер под вопросом`) are Russian too. This document is the reference for operating the pipeline, not a description of its output language.

## Features

- **Words with timings, then speakers** — GigaSTT (RNNT) gives per-word timestamps; pyannote gives exclusive diarization; every word is assigned a speaker by time.
- **Five output formats** — JSON, Markdown, plain text, SRT and VTT from one run; subtitles are split on word boundaries with real timings.
- **Honest uncertainty** — `[перекрытие речи]` for overlapping speech and `[спикер под вопросом]` for doubtful attribution mark where to listen again.
- **Two audio copies from one source** — a near-original one for diarization, a gently filtered one for recognition; the original file is never modified.
- **Resumable, reproducible jobs** — a manifest vouches for each finished stage; changing an option that affects the result requires `--force`.
- **Hand-assigned names** — fill in a speaker map after listening and re-render without any inference.
- **Pinned where it can be** — the GigaSTT executable by URL and SHA-256, pyannote by full git revision (the GigaAM RNNT, punctuation and VAD weights fetched by `gigastt` are not hash-pinned yet); repo-hygiene checks keep audio, tokens and models out of Git.

## How it works

<p align="center">
  <img src="docs/assets/gigastt-pyannote-transcriber-how-it-works.png" alt="gigastt-pyannote-transcriber: one audio file goes through GigaSTT (what was said, when) and pyannote (who spoke when), the two are merged by time so each word gets a speaker, and the output is a transcript in JSON, Markdown, TXT, SRT and VTT" width="100%">
</p>

1. **Prepare two audio copies.** One recording becomes a near-original mono 16 kHz copy for diarization and a gently filtered copy for recognition.
2. **Recognise the words.** GigaSTT (RNNT) produces words with start and end times.
3. **Find the speakers.** pyannote Community-1 is given `num_speakers=4` and produces an exclusive diarization.
4. **Align by time.** `fourvoices` gives each word a speaker, marks overlap and uncertain attribution, and writes the transcript in five formats.
5. **Name the speakers yourself.** Listen to long turns from each cluster, fill in a speaker map, and re-render — labels are never turned into names automatically.

## Quick start

Short version; every step is explained in the [run checklist](#run-checklist) and [the detailed Windows setup](#quick-start-on-a-clean-windows-machine) below. First accept the conditions of the gated pyannote model on Hugging Face and create a read token (see [One-time access](#one-time-access-to-the-gated-hugging-face-model)).

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1 -InstallFfmpeg

# The token lives only in the current PowerShell and is never written to disk:
$secureToken = Read-Host 'HF read token' -AsSecureString
$env:HF_TOKEN = [Net.NetworkCredential]::new('', $secureToken).Password

.\scripts\download-models.ps1       # needs network, ~10 GB
.\scripts\doctor.ps1                # every line must read [OK]

# put recordings in .\media\, then:
.\scripts\run.ps1 -InputAudio '.\media\recording.m4a'
```

Results land in `..\gigastt-pyannote-output\<job>\` next to the repository, where they cannot accidentally be committed.

## Run checklist

The whole sequence, from a clean machine to a finished transcript. Every step
is explained in detail further down.

| # | Step | When |
|---|------|------|
| 1 | Accept the model conditions on Hugging Face and create a read token | once |
| 2 | `.\scripts\install.ps1 -InstallFfmpeg` | once |
| 3 | Set `$env:HF_TOKEN` | **in every new PowerShell window** |
| 4 | `.\scripts\download-models.ps1` | once, needs network, ~10 GB |
| 5 | `.\scripts\doctor.ps1` — every line must read `[OK]` | after installing |
| 6 | `.\scripts\test.ps1` — linter, tests, repository hygiene | optional |
| 7 | Put the recordings in `.\media\` | before working |
| 8 | `.\scripts\run.ps1 -InputAudio '.\media\recording.m4a'` | per recording |
| 9 | Listen to the clusters, fill in a speaker map, `.\scripts\rerender.ps1` | after a run |

**`HF_TOKEN` is needed for transcription, not only for downloading.**
Diarization refuses to start without it even when every weight is already on
disk. If you clear the token from the environment after installing, step 8
stops with `error: Set HF_TOKEN...`. Either set the token in each new
PowerShell window, or put it once into a local `.env` (the file is ignored by
Git).

## Requirements

- Windows 10/11 x64, **any** CPU — this is a CPU-only pipeline. The timings
  quoted here were measured on a Ryzen 9 7950X, but it is not a requirement:
  a slower machine is simply slower;
- PowerShell 5.1 or 7;
- about 10 GB of free space for the install, the models and the intermediate
  WAV files;
- RAM: diarization loads the whole recording into memory, roughly 230 MB per
  hour of audio plus the model itself. 16 GB is comfortable for recordings of
  an hour or two;
- a stable internet connection for the first install;
- a Hugging Face account and a read token.

Python 3.11 and the virtual environment are managed by
[uv](https://docs.astral.sh/uv/). PyTorch is pinned to the CPU build
`2.11.0+cpu`; CUDA is not required and not supported by that build.

## One-time access to the gated Hugging Face model

Before downloading anything:

1. Sign in to Hugging Face.
2. Open
   <https://huggingface.co/pyannote/speaker-diarization-community-1>,
   read and accept the access conditions.
3. Create a **Read** token at <https://huggingface.co/settings/tokens>.
4. Never publish the token or paste it into an issue, a commit or a log.

The model revision is pinned:
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.

## Quick start on a clean Windows machine

Move the repository to the machine (`git clone` if you have published it
somewhere, otherwise just copy the directory). Do not copy `.venv`, `models`,
`tools\bin` or `tools\downloads`: `.venv` contains absolute paths from the
original machine, and the install scripts rebuild the rest.

```powershell
Set-Location .\gigastt-pyannote-transcriber
Set-ExecutionPolicy -Scope Process Bypass

# uv, Python 3.11, .venv and the CPU dependencies; optionally ffmpeg as well:
.\scripts\install.ps1 -InstallFfmpeg

# The token lives only in the current PowerShell and is never written to disk:
$secureToken = Read-Host 'HF read token' -AsSecureString
$env:HF_TOKEN = [Net.NetworkCredential]::new('', $secureToken).Password
```

If ffmpeg was installed through winget and is not on `PATH` yet, open a new
PowerShell window, return to the repository directory and set the token again
with the commands above. Then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\download-models.ps1
.\scripts\doctor.ps1
.\scripts\test.ps1
```

Do **not** clear the token at this point: transcription needs it too. Clearing
it (`Remove-Item Env:HF_TOKEN`) makes sense only when you are done for the day,
remembering that the next PowerShell window will need it set again.

`download-models.ps1` downloads exactly GigaSTT **v2.15.0** for Windows x64,
verifies the archive against the SHA-256 in `tools/tools.lock.json`, then
fetches RNNT INT8, VAD, punctuation and the pinned pyannote revision. An
unverified `gigastt.exe` is never executed.

Put your recordings in the ignored `media` directory:

```powershell
Copy-Item 'D:\Audio\interview-a.m4a' .\media\
Copy-Item 'D:\Audio\interview-b.m4a' .\media\

.\scripts\run.ps1 -InputAudio @(
    '.\media\interview-a.m4a',
    '.\media\interview-b.m4a'
)
```

If there are not four speakers, say so: `-NumSpeakers 3`.

Files are processed **strictly one at a time**, so that two heavy CPU jobs do
not compete for memory. Results land by default in `gigastt-pyannote-output\`
next to the repository, where they cannot accidentally be committed. The script
asks for four speaker clusters but never guesses people's names.

If pyannote finds a different number of speakers — one participant stayed
almost silent, say — the diarization is still saved and the run prints a
warning: losing tens of minutes of work over that mismatch is not acceptable.
Add `-StrictSpeakers` to get a refusal instead of a warning; the result is kept
on disk either way.

If the recording is stereo the program stops rather than mixing the channels
silently. Listen to the left and right channels separately. If there is no
useful spatial separation, repeat the command with `-AllowDownmix`.

## What a job directory contains

A job is named after the sanitised file name plus the first 12 characters of
the source audio's SHA-256, so two different recordings can never share a
directory.

```
<name>-<hash>/
  manifest.json          what was computed, when, and with which options
  audio/asr.wav          the recognition copy (filtered)
  audio/diarization.wav  the diarization copy (unfiltered)
  intermediate/gigastt.json    words and their timings
  intermediate/pyannote.json   speaker segments
  intermediate/merged.json     the combined result
  transcript.{md,txt,srt,vtt,json}
```

Re-running the same command does not recompute finished stages: they are read
back from `intermediate/`, but only when `manifest.json` vouches for them. A
lost or corrupted manifest therefore causes an honest recomputation rather than
a transcript spliced together from stages produced with different settings.

Changing an option that affects the result — speaker count, audio filters,
merge thresholds, model variant — stops the run and asks for `--force`: mixing
old and new stages silently is not allowed. PyTorch thread counts do not affect
the result and do not trigger a rebuild. An interrupted run (Ctrl+C) continues
from the last **completed** stage; a stage interrupted midway counts as never
computed and runs again in full.

Changing the presentation needs no inference at all — use the `render` command,
which re-reads `merged.json`.

## Assigning speakers by hand

After the first pass, listen to several long turns from each cluster. Copy the
map and replace the labels:

```powershell
Copy-Item .\config\speaker-map.example.yaml `
    ..\gigastt-pyannote-output\<job-name>\speaker-map.yaml
notepad ..\gigastt-pyannote-output\<job-name>\speaker-map.yaml

.\scripts\rerender.ps1 `
    -JobDir ..\gigastt-pyannote-output\<job-name> `
    -SpeakerMap ..\gigastt-pyannote-output\<job-name>\speaker-map.yaml
```

Never assign a name from a short "yes" or "mhm": use long, clearly audible
fragments. A label that does not exist in the transcript renames nobody, and
the run says so instead of pretending the rename worked. Interruptions and
simultaneous speech need manual proofreading regardless of the model.

## How the audio is processed

- pyannote gets a near-original version: mono, 16 kHz, no aggressive noise
  reduction (`audio.diarization_filter`).
- ASR gets a separate version with a gentle 80 Hz high-pass and loudness
  normalisation (`audio.asr_filter`).
- Stereo should be auditioned channel by channel first: sometimes the left and
  right channels already separate the participants. Automatic downmixing before
  that check is undesirable.
- The original file is never modified.

The defaults live in `config/default.yaml`. Every key there is genuinely read
by the pipeline, and unknown keys are rejected with an error, so a typo cannot
quietly look like a setting. Relative paths in the configuration resolve
against the project root — the nearest enclosing directory containing
`pyproject.toml` — not against the current working directory.

## Why the defaults are what they are

The defaults target one task: a four-person Russian conversation captured on a
single microphone, which will later have to be quoted and checked by ear.
Below is the reasoning behind each, and what changes if you move it.

### Two different files from one source

`audio.diarization_filter: null` and
`audio.asr_filter: highpass=f=80,loudnorm=I=-18:LRA=11:TP=-2`.

This is the pipeline's central decision. Recognition cares about
intelligibility; diarization cares about timbre. Loudness normalisation and
denoising alter exactly the voice characteristics that speaker clustering
relies on, so diarization receives audio as close to the source as possible:
downmixing and resampling only.

The ASR copy is cleaned gently. 80 Hz sits below the fundamental of an ordinary
speaking voice, so the high-pass removes rumble, desk knocks and wind while
barely touching speech. `loudnorm` brings the recording to a predictable
loudness (−18 LUFS) so that quiet and loud recordings are recognised alike.
Processing harder is risky: the more aggressive the filter, the greater the
chance of losing quiet word endings and short interjections.

### Diarization

`num_speakers: 4` — the speaker count is given rather than inferred, because a
fixed count makes clustering reproducible from run to run. If pyannote returns
a different number, the result is still saved and a warning is printed.

`model` + `revision` — the model is pinned by name *and* by revision. An
upstream update must never change what an already-issued transcript would say.
A configuration asking for anything else is rejected, not silently honoured.

`device: cpu` — the pinned PyTorch build is CPU-only; there is no CUDA here.

### Recognition

`model_variant: rnnt` — the entire pipeline is built on per-word timings:
without a `words` array carrying a start and an end for each word, merging with
diarization is impossible and recognition fails outright. Change the variant
only to one that also emits word timestamps.

`punctuation: true`, `itn: true` — GigaSTT applies punctuation and number
normalisation to the top-level text only, leaving `words[]` raw. Merging
projects the processed spelling back onto the timestamped words, and only where
the normalised forms match exactly: a rewrite such as "двадцать три" → "23"
falls back to the raw word instead of inventing an alignment. Turning these off
costs readability, not timings.

`vad: true` — voice activity detection keeps the recogniser out of long
silences, where it would otherwise be free to invent text.

### Merging

`max_turn_gap: 1.5` — the silence inside one person's speech after which the
turn is split in two. The value is chosen with margin over ordinary
between-sentence pauses so that a turn is not torn apart at a breath. Lower it
for more granular turns; raise it if a participant speaks with long pauses for
thought.

`nearest_max_gap: 0.5` — how far a word may sit from the nearest exclusive
segment before its speaker becomes `UNKNOWN`. A larger value means fewer
`UNKNOWN` labels but more guessing; a smaller one leaves more words honestly
unattributed. Such words feed the "speaker in question" marker.

### Output

`mark_uncertain_words: true` — a turn is flagged when at least half of its
words are attributed without confirmed overlap. One cautious hand-off in a long
sentence is normal; half of them is a reason to listen again.

`subtitle_max_seconds: 6`, `subtitle_max_chars: 84` — roughly two lines of 42
characters for a few seconds on screen. A conversational turn can run for
minutes, and without splitting the subtitles are unreadable.

### Threads

`-TorchThreads` defaults to the machine's logical processor count
(`os.cpu_count()`), so the repository behaves sensibly on any CPU without being
edited. Beware that this counts **logical** processors: on an SMT part such as
a Ryzen 9 7950X that is 32, not the 16 physical cores. Memory-bandwidth-bound
decoding often gains nothing from SMT and can lose to it, so if you want the
physical count, pass it: `-TorchThreads 16`.

`-TorchInteropThreads` defaults to 1, limiting inter-op parallelism so that
threads do not compete for the same cores. Note that PyTorch accepts this
setting only before its first parallel operation, so it is applied on a
best-effort basis and may silently have no effect. Correctness is unaffected
either way.

Neither option appears in the configuration file on purpose: they do not change
the result, they live on the command line only, and they never trigger a
rebuild. In `run.ps1` a value of `0` means "let fourvoices decide".

## What gets marked in the transcript

- `[перекрытие речи]` ("overlapping speech") — according to the regular
  diarization, two people are speaking during this interval.
- `[спикер под вопросом]` ("speaker in question") — at least half of the turn's
  words are attributed to a speaker without confirmed exclusive overlap: the
  nearest segment was used, two speakers tied, or no segment covered the word
  at all.

Switched off with the `output.mark_uncertain_words` key or the
`--no-mark-uncertain` flag of the `render` command. Both markers are a prompt
to listen to the fragment again, not a sign of an error.

## Subtitles

A conversational turn can last for minutes, so in SRT and VTT it is split into
separate cues on word boundaries: the timings stay real rather than being
interpolated across the turn. By default a cue is at most 6 seconds and 84
characters (two lines of 42), and the speaker's name is repeated in every cue.

The limits come from the `output.subtitle_max_seconds` and
`output.subtitle_max_chars` keys, or from the `--subtitle-max-seconds` /
`--subtitle-max-chars` flags of the `render` command; `0` disables the
corresponding limit. If `merged.json` carries no per-word timings, the turn is
left as a single cue — a wrong timestamp is worse than a long subtitle.

## Useful commands

```powershell
# Check the environment, the token, the models and CPU PyTorch
.\scripts\doctor.ps1

# Linter, tests, and protection against committing audio or tokens
.\scripts\test.ps1

# Re-download and re-verify GigaSTT only
.\scripts\download-models.ps1 -SkipPyannote -Force

# The gated pyannote model only
.\scripts\download-models.ps1 -SkipGigaStt
```

## Security and reproducibility

- Never use `git add -f` for audio, `.env`, `models\` or `transcripts\`.
- Run `.\scripts\check-repo-hygiene.ps1` before committing.
- Setting `HF_TOKEN` for the current PowerShell process only is preferred. A
  local `.env` is supported as a fallback and is ignored by Git.
- The GigaSTT executable is pinned by URL and SHA-256; pyannote by full git revision. The GigaAM RNNT, punctuation and VAD weights are downloaded by `gigastt` itself and are not hash-pinned.
- The first download needs the network; once the models are in place, the
  processing itself is entirely local.

Automatic transcription is not a guarantee of verbatim accuracy. For quotes
that carry legal weight, always keep the original recording and its timecodes,
and check the words by ear.
