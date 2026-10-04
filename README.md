<p align="center"><img src="docs/assets/gigastt-pyannote-transcriber-hero.png" alt="gigastt-pyannote-transcriber: two people talk, a speech-to-text stenographer writes down words with timestamps, a speaker tagger assigns each word to a speaker, and a who-said-what transcript comes out" width="100%"></p>
<h1 align="center">gigastt-pyannote-transcriber</h1>
<p align="center"><b>Russian recording in, who-said-what transcript out — a local, CPU-only Windows toolkit (GigaSTT + pyannote) with resumable jobs, five output formats, hand-assigned speaker names and pinned, hygiene-checked tooling, for multi-speaker conversations you will have to quote and check by ear.</b></p>
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

**Fourvoices: GigaSTT + pyannote for four voices.** A local, CPU-only pipeline for Russian-language recordings: **GigaSTT 2.21.0 / GigaAM v3 RNNT** recognises the words and their timings, **pyannote Community-1** says who spoke when, and the `fourvoices` Python package aligns the two in time. (The package and CLI are called `fourvoices`; the repository is `gigastt-pyannote-transcriber`.) Source audio, models, tokens and results are deliberately kept out of Git.

Around that core the repository ships eight PowerShell scripts (install, download and verify models, doctor, run, re-render, evaluate, test, repository hygiene), a five-command CLI (`run`, `render`, `diarize-preload`, `doctor`, `evaluate`), a schema-validated configuration file, resumable job directories with a manifest, a speaker-map re-render step that needs no inference, and a pytest suite with a Windows CI workflow.

It is version 0.1.0 and Windows-only. It needs a Hugging Face account and read token (`HF_TOKEN` is required for every transcription run), and the GigaSTT executable and the pyannote revision are pinned (GigaSTT itself verifies the weights it loads) — see [Status and known limits](#status-and-known-limits).

**Contents:** [In plain words](#in-plain-words) · [What's inside](#whats-inside) · [How it works](#how-it-works) · [Quick start](#quick-start) · [Run checklist](#run-checklist) · [Requirements](#requirements) · [Hugging Face access](#one-time-access-to-the-gated-hugging-face-model) · [Setup on a clean machine](#quick-start-on-a-clean-windows-machine) · [Job directory](#what-a-job-directory-contains) · [Speaker names](#assigning-speakers-by-hand) · [Audio processing](#how-the-audio-is-processed) · [Why these defaults](#why-the-defaults-are-what-they-are) · [Markers](#what-gets-marked-in-the-transcript) · [Subtitles](#subtitles) · [How words get a speaker](#how-words-get-a-speaker) · [Output files](#output-files-in-detail) · [Configuration](#configuration-reference) · [Command reference](#command-reference) · [Security](#security-and-reproducibility) · [Troubleshooting](#troubleshooting) · [Status](#status-and-known-limits) · [Docs map](#documentation-map)

## In plain words

### The problem

You have a recording of several people talking in Russian — an interview, a meeting, a phone call — and you need to know not only *what* was said but *who* said it, with times, in a form you can quote. Ordinary speech-to-text gives one block of text with no speakers. Splitting by speaker is a second, separate job, and the two results have to be lined up in time. Doing this in the cloud means uploading the recording; doing it locally means installing several models and tools that need to fit together, and a re-run after a small change should not cost hours.

### Who it is for

People who work with Russian-language, multi-speaker audio on a Windows PC and need a checkable result: journalists and researchers transcribing interviews, people documenting meetings or calls. The defaults are tuned for a four-person conversation on one microphone. Advanced users get the full CLI, a per-key configuration, a documented merge algorithm and structured JSON for their own tooling.

### What you get

- **A who-said-what transcript** from one audio file: every turn carries a speaker label and a time range; formats are Markdown, plain text, SRT, VTT and JSON.
- **Honest uncertainty.** Overlapping speech and doubtful speaker attribution are marked in the text instead of being hidden, so you know where to listen again.
- **Local processing.** Recognition and diarization run on your own CPU, with no GPU. The network is needed to install and to download the models; transcription itself runs from the local model files.
- **Nothing silently mixed.** A manifest records which stages are done; a re-run reuses finished stages, and changing an option that affects the result stops the run until you confirm with `--force`.
- **Names only when you decide.** Labels such as `SPEAKER_00` become names in the Markdown, text, SRT and VTT files only through a speaker map you fill in after listening; `transcript.json` always keeps the original `SPEAKER_nn` labels, and the pipeline never guesses names.
- **Guarded repository.** Audio, tokens and models are git-ignored and checked by a hygiene script; the GigaSTT download is verified by SHA-256 before it is unpacked or run.

### What it is not

- Not a hosted service and not a GPU pipeline: the pinned PyTorch build is CPU-only.
- Not for other languages or platforms: the recognition model is Russian, the scripts are PowerShell for Windows 10/11 x64, and CI runs only on Windows.
- Not a guarantee of verbatim accuracy: always check quotes by ear against the original recording.
- Not automatic speaker naming: labels are never turned into names for you.
- Not a stereo-channel separator: multichannel input stops the run unless you explicitly allow downmixing.
- Not fully hash-pinned: the pyannote weights are pinned by revision only; the GigaAM weights are verified by `gigastt` against digests built into it, not by this repository.
- Not offline from the first minute: install and model download need the network and about 10 GB, plus a Hugging Face account for the gated pyannote model.

### Glossary

| Term | Meaning here |
|---|---|
| **ASR / GigaSTT** | Automatic speech recognition. GigaSTT is the command-line program (a pinned release) that runs the GigaAM v3 RNNT model and returns words with start and end times. |
| **Diarization / pyannote Community-1** | Working out *who spoke when*. Community-1 is the pyannote model used, fetched from Hugging Face at a pinned revision. |
| **Exclusive diarization** | A pyannote output in which at most one speaker is active at any moment; words are assigned against it. The regular output (which allows overlaps) is used to detect simultaneous speech. |
| **Speaker label** | A cluster name such as `SPEAKER_00`, or `UNKNOWN` for a word no segment could be attributed to. |
| **Turn** | A run of consecutive words by one speaker, split when the silence between words exceeds `max_turn_gap`. |
| **Job directory** | The folder for one recording, named after the file name and the first 12 characters of the audio's SHA-256. |
| **Manifest** | `manifest.json` in the job directory: the input fingerprint, the options used and which stages finished. |
| **Speaker map** | A small YAML file mapping `SPEAKER_00` and so on to names you choose after listening. It is applied to the Markdown, text, SRT and VTT files, not to the JSON. |
| **Gated model / `HF_TOKEN`** | The pyannote model requires you to accept its conditions on Hugging Face; a read token proves it and is needed on every transcription run. |
| **ITN** | Inverse text normalisation: spoken numbers written as digits ("двадцать три" → "23"). |

## What's inside

Status labels: everything listed here is implemented in this repository; nothing is labelled experimental or planned in the code or docs. The last subsection lists what is not built.

### The transcription pipeline

- **Audio preparation.** One recording becomes two 16 kHz mono WAV files: a near-original one for diarization and a gently filtered one for recognition. The source file is never modified; multichannel input is refused unless you allow downmixing. → [How the audio is processed](#how-the-audio-is-processed)
- **Recognition.** The pinned GigaSTT executable produces words with start and end times, optional punctuation, number normalisation and voice-activity detection. → [Why these defaults](#why-the-defaults-are-what-they-are)
- **Diarization.** pyannote Community-1 at a pinned revision, with the speaker count given (default 4). A count mismatch is saved and warned about, or refused with `-StrictSpeakers`. → [Why these defaults](#why-the-defaults-are-what-they-are)
- **Merge.** Each word gets the speaker with the greatest overlap; gaps fall back to the nearest segment or `UNKNOWN`; overlap and uncertainty are recorded; words are grouped into turns. → [How words get a speaker](#how-words-get-a-speaker)
- **Rendering.** Markdown, TXT, SRT, VTT and JSON from one run; subtitles are cut on word boundaries with real timings. → [Output files](#output-files-in-detail)

### Reliability and reproducibility

- **Resumable jobs.** Finished stages are read back from disk only when the manifest vouches for them; an interrupted run continues from the last completed stage. → [What a job directory contains](#what-a-job-directory-contains)
- **Refusal instead of mixing.** A changed result-affecting option, or a different input in the same job directory, stops the run unless `--force` is given. → [What a job directory contains](#what-a-job-directory-contains)
- **Pins.** GigaSTT executable by URL and SHA-256; pyannote by full git revision; the CLI rejects a configuration that asks for another model or revision. → [Security and reproducibility](#security-and-reproducibility)
- **Strict configuration.** Unknown sections and keys are rejected, so a typo cannot look like a setting. → [Configuration reference](#configuration-reference)

### Working with the result

- **Hand-assigned names.** Fill in a speaker map after listening and re-render without any inference; the names appear in Markdown, text, SRT and VTT, while the JSON keeps the labels. → [Assigning speakers by hand](#assigning-speakers-by-hand)
- **Markers.** `[перекрытие речи]` for overlapping speech and `[спикер под вопросом]` for doubtful attribution. → [What gets marked](#what-gets-marked-in-the-transcript)
- **Structured JSON.** The merged JSON carries every word with its speaker, evidence and flags, plus the turns. → [Output files](#output-files-in-detail)

### Tooling around the pipeline

- **Eight PowerShell scripts.** `install`, `download-models`, `doctor`, `run`, `rerender`, `evaluate`, `test` and `check-repo-hygiene`. → [Command reference](#command-reference)
- **A five-command CLI.** `run`, `render`, `diarize-preload`, `doctor` and `evaluate`, with options the scripts do not expose (for example `--force`). → [Command reference](#command-reference)
- **Repository hygiene and tests.** A script that fails if audio, models, tokens or secrets are tracked, a pytest suite, ruff, and a Windows CI workflow. → [Status and known limits](#status-and-known-limits)

### Not built

Other languages, GPU/CUDA, Linux or macOS scripts, parallel processing of several files, streaming/real-time use, automatic naming of speakers and stereo-channel-aware diarization are not part of this repository. See [What it is not](#what-it-is-not) and [Status and known limits](#status-and-known-limits).

## How it works

<p align="center">
  <img src="docs/assets/gigastt-pyannote-transcriber-how-it-works.png" alt="gigastt-pyannote-transcriber: one audio file goes through GigaSTT (what was said, when) and pyannote (who spoke when), the two are merged by time so each word gets a speaker, and the output is a transcript in JSON, Markdown, TXT, SRT and VTT" width="100%">
</p>

1. **Prepare two audio copies.** One recording becomes a near-original mono 16 kHz copy for diarization and a gently filtered copy for recognition.
2. **Recognise the words.** GigaSTT (RNNT) produces words with start and end times.
3. **Find the speakers.** pyannote Community-1 is given `num_speakers=4` and produces an exclusive diarization.
4. **Align by time.** `fourvoices` gives each word a speaker, marks overlap and uncertain attribution, and writes the transcript in five formats.
5. **Name the speakers yourself.** Listen to long turns from each cluster, fill in a speaker map, and re-render — labels are never turned into names automatically. The names go into the Markdown, text, SRT and VTT files; the JSON keeps the labels.

The CLI prints the five stages as it goes, and each stage leaves a file and a manifest entry:

| Stage | Printed as | Writes | Manifest key |
|---|---|---|---|
| 1 | `[1/5] Preparing separate ASR and diarization audio…` | `audio/asr.wav`, `audio/diarization.wav` | `audio` |
| 2 | `[2/5] Running GigaSTT RNNT…` (or `Reusing GigaSTT timestamps.`) | `intermediate/gigastt.json` | `asr` |
| 3 | `[3/5] Running pinned Community-1 (N speakers)…` (or `Reusing pyannote diarization.`) | `intermediate/pyannote.json` | `diarization` |
| 4 | `[4/5] Assigning words by exclusive maximum overlap…` (or `Reusing merged transcript.`) | `intermediate/merged.json` | `merge` |
| 5 | `[5/5] Rendering …` | `transcript.{md,txt,srt,vtt,json}` | `render` |

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

`download-models.ps1` downloads exactly GigaSTT **v2.21.0** for Windows x64,
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
  intermediate/gigastt.log     GigaSTT's own log of that run
  intermediate/pyannote.json   speaker segments
  intermediate/merged.json     the combined result
  transcript.{md,txt,srt,vtt,json}
```

Recognition and diarization take minutes per hour of audio on a CPU, so both
print progress every half minute or so: recognition counts decoded windows
(with VAD their number is not known in advance, so it cannot be a percentage)
and ends with its speed; diarization names each pyannote step and its share
done. These are ordinary lines, readable in a redirected log too.

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

The recognition and diarization stages also record the software that
produced them. A GigaSTT release changes the words and their timings, so when
the GigaSTT being run (`run.ps1` always runs the one pinned in
`tools/tools.lock.json`) differs from the one recorded for a job (or none was recorded — jobs made before this was
tracked), only recognition is redone and everything built on it follows;
the diarization is kept. A different `pyannote.audio` or `torch` only prints a
warning, because the model is pinned by revision and diarization is the most
expensive stage — use `--force` if you want it redone. Whenever recognition or
diarization is redone, the merged result is rebuilt rather than reused.

GigaSTT reports some problems only in its log while still succeeding, so its
warnings are repeated on screen and the log is kept in `intermediate/`. One
such problem is checked explicitly: if punctuation was requested but a long
transcript comes back without a single sentence mark, the run warns and the
manifest records `punctuation_missing: true`.

Changing the presentation needs no inference at all — use the `render` command,
which re-reads `merged.json`. It keeps the subtitle limits and uncertainty
marks the job was last rendered with unless you pass a flag (jobs rendered
before these were recorded fall back to the defaults). It refuses to run when
the manifest no longer vouches for `merged.json` — after a run that redid
recognition or diarization was interrupted before rebuilding it; finish that
run first.

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

The map is applied only while the Markdown, text, SRT and VTT files are
rendered. `transcript.json` is the merged data written unchanged, so after a
re-render its `speakers`, `turns` and `words` still carry `SPEAKER_00` and so
on; a tool that reads the JSON has to apply the same label-to-name mapping
itself. The names that were applied are recorded in the `render` stage of
`manifest.json` (`speaker_names`).

## Measuring quality on your own recordings

Automatic transcription has to be checked against something. `evaluate`
compares a finished job with a reference you trust and reports two numbers:

- **WER** — the share of reference words that were substituted, missed or
  inserted, after lowercasing, `ё` → `е` and stripping punctuation;
- **speaker accuracy** — among the words both texts share, the share given to
  the right person. Clusters are matched to the reference names one-to-one in
  the way that agrees with the reference most; `UNKNOWN` is never matched, so
  it always counts as an error.

The reference is a text file in the shape of `transcript.txt`: one turn per
line, `[hh:mm:ss–hh:mm:ss] Name: text`. The timestamp is optional, markers such
as `[перекрытие речи]` are ignored, a line without `Name:` continues the
previous speaker, and lines starting with `#` are comments. Nobody transcribes
two hours by hand, so a reference normally covers a fragment: only the words
inside its time span are scored (or pass `-Start`/`-End`).

```powershell
# Make a reference: copy the transcript, keep a fragment of a few minutes,
# then listen and fix every word and every speaker name in it.
Copy-Item ..\gigastt-pyannote-output\<job>\transcript.txt .\media\reference.txt
notepad .\media\reference.txt

.\scripts\evaluate.ps1 ..\gigastt-pyannote-output\<job> -Reference .\media\reference.txt
```

The result goes to `<job>\evaluation.json`, together with every place where
the words differ and its time in the recording, so you know what to listen to.
The job itself is not modified.

Two cautions:

- **A reference made by correcting the transcript is optimistic.** Errors that
  read naturally are easy to miss when the text is in front of you. For an
  honest figure, type a fragment from scratch while listening.
- **Write numbers the way the transcript does.** Number normalisation turns
  "двадцать три" into "23"; a reference that spells it out counts that as an
  error.

With a reference in place, any change — a new GigaSTT, different filters,
another speaker count — can be judged by numbers rather than by ear.

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

**Why Community-1 and not Precision.** pyannote's commercial models
(`precision-2`, and since September 2026 `precision-3` behind the same
`pyannote/speaker-diarization-precision` pipeline) score a noticeably lower
diarization error rate — on pyannote's own benchmarks roughly 15 % against
20 % on meeting recordings such as AMI and AliMeeting. They are, however,
served only through the pyannoteAI cloud API: the pipeline in `pyannote.audio`
uploads the audio file to pyannoteAI and needs a paid API key, and there are
no downloadable weights (self-hosting is offered only on enterprise
contracts). This project exists to keep recordings on the machine, so it stays
on Community-1, the newest open model, and does not offer the cloud model as
an option. pyannote's usage telemetry (file duration and speaker count sent
to `otel.pyannote.ai`) is switched off unconditionally for the same reason.

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

**Why GigaSTT 2.21.0.** The previously pinned 2.15.0 had two faults that hit
exactly the recordings this pipeline is for, both fixed in 2.16.0 and both
reproduced here on real Russian speech. Once the text outgrew the punctuation
model's 2048-token window (a 28-minute recording, 1 574 words) punctuation was
silently dropped: exit code 0, a lowercase transcript without a single mark.
And with `--vad`, anything longer than 30 minutes was refused outright. 2.21.0
decodes files in bounded windows with no length cap (a 36-minute recording was
checked end to end), punctuates long transcripts in overlapping windows, and
verifies the recognition weights against pinned SHA-256 digests at every load.
Every flag this project passes behaves as before, and on the same audio the
raw words differ by a handful out of a thousand. Jobs
made with 2.15.0 are re-recognised automatically on their next run (see
[What a job directory contains](#what-a-job-directory-contains)).

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

The transcripts themselves are Russian, so the markers inside them (`перекрытие речи`, `спикер под вопросом`) are Russian too; this document describes how to operate the pipeline, not its output language.

- `[перекрытие речи]` ("overlapping speech") — according to the regular
  diarization, two people are speaking during this interval.
- `[спикер под вопросом]` ("speaker in question") — at least half of the turn's
  words are attributed to a speaker without confirmed exclusive overlap: the
  nearest segment was used, two speakers tied, or no segment covered the word
  at all.

Switched off with the `output.mark_uncertain_words` key or the
`--no-mark-uncertain` flag of the `render` command, and back on with
`--mark-uncertain`. Both markers are a prompt
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

## How words get a speaker

This is the merge step (`src/fourvoices/merge.py`). It works on the words GigaSTT returned and on the two annotations pyannote saved to `intermediate/pyannote.json`: the *exclusive* segments (one speaker at a time) and the *regular* segments (overlaps allowed).

1. **Greatest overlap wins.** For each word, the time it shares with each speaker's exclusive segments is added up; the speaker with the most wins. A zero-length word is treated as a tiny interval. Equal totals mark the word `ambiguous`.
2. **Gaps use the nearest segment.** If no exclusive segment covers the word, the nearest one is used when it lies within `nearest_max_gap` seconds (default 0.5); the word is recorded as assigned by `nearest`. Farther than that, the word's speaker is `UNKNOWN`.
3. **Overlap comes from the regular annotation.** A word or turn is marked overlapping when two *different* speakers' regular segments are active at once; segments that merely touch do not count.
4. **Words become turns.** Consecutive words by the same speaker form one turn; a new turn starts when the speaker changes or the silence between words exceeds `max_turn_gap` (default 1.5 s).
5. **Doubtful turns are flagged.** A turn is `uncertain` when at least half of its words rest on something weaker than confirmed overlap (`nearest`, `UNKNOWN` or a tie). That is what produces the `спикер под вопросом` marker.
6. **Display text keeps GigaSTT's punctuation where it is safe.** GigaSTT applies punctuation and number normalisation only to the top-level text and leaves each word raw. The merge projects the processed spelling back onto the timestamped words only where the normalised forms match exactly (case, edge punctuation and `ё`/`е` ignored); otherwise the raw word is kept. The share of matched words is saved as `punctuation_alignment_ratio`.

Lookups use sorted indexes with binary search, so merging does not slow down quadratically with recording length; `tests/test_merge_equivalence.py` compares the indexed code with simple reference implementations.

## Output files in detail

All five files are written from `intermediate/merged.json`; `render` can regenerate them without inference.

| File | What it contains |
|---|---|
| `transcript.txt` | one line per turn: `[hh:mm:ss–hh:mm:ss] Speaker [markers]: text` |
| `transcript.md` | a heading, a short note that the text is automatic and needs checking against the audio, then each turn as a bold `[time range] Speaker · markers` line followed by its text (the heading and note are in Russian) |
| `transcript.srt`, `transcript.vtt` | cues with real word-boundary timings; the speaker label (and markers) repeat in every cue; lines wrap at 42 characters; VTT starts with `WEBVTT` |
| `transcript.json` | the merged JSON, see below; speaker maps do not change it, it keeps the `SPEAKER_nn` labels |

**Merged JSON.** Top level: `schema_version`, `duration_s`, `speakers`, `text`, `raw_text`, `gigastt_processed_text`, `punctuation_alignment_ratio`, `words`, `turns`. Each **word** has `index`, `word`, `display_word`, `start`, `end`, `confidence`, `speaker`, `speaker_overlap_s`, `assignment` (`overlap`, `nearest` or `unknown`), `ambiguous` and `overlap`. Each **turn** has `id`, `speaker`, `start`, `end`, `text`, `raw_text`, `overlap`, `uncertain_words`, `uncertain`, `confidence` (duration-weighted word confidence), `word_start` and `word_end` (indexes into `words`). The `speaker` values are always the pipeline's `SPEAKER_nn` labels: a speaker map is not applied to this file.

**Other files in a job directory.** `intermediate/gigastt.json` (validated GigaSTT output; each word needs `start` and `end`), `intermediate/pyannote.json` (model, revision, requested and found speakers, `speaker_count_matches_request`, regular and exclusive segments), and `manifest.json` (below). Files are written through a `.partial` temporary and renamed, so an interrupted write does not leave a half-written result; if GigaSTT returns JSON the program cannot use, the raw output is kept as `gigastt.json.rejected`.

**The manifest.** `manifest.json` holds `schema_version`, `pipeline_version`, `created_at`, `updated_at`, the `input` fingerprint (`name`, `size_bytes`, `mtime_ns`, `sha256`; never an absolute path), the `config` that affects the result, and a `stages` map (`audio`, `asr`, `diarization`, `merge`, `render`), each with `completed_at` and stage details (for example the word count, the speaker labels found, the files rendered and the speaker names used).

## Configuration reference

`config/default.yaml` is read by `run` and `doctor` (`render` does not read it). Unknown sections and keys are rejected with a message listing the supported keys. Relative paths resolve against the project root (the nearest enclosing directory containing `pyproject.toml`). A command-line option, where one exists, overrides the YAML value.

| Key | Default | What it does |
|---|---|---|
| `project.output_root` | `../gigastt-pyannote-output` | where job directories go |
| `audio.diarization_filter` | `null` | ffmpeg `-af` filter for the diarization copy (none by default) |
| `audio.asr_filter` | `highpass=f=80,loudnorm=I=-18:LRA=11:TP=-2` | ffmpeg filter for the recognition copy |
| `asr.model_variant` | `rnnt` | GigaSTT model variant; must emit word timestamps |
| `asr.model_dir` | `models/gigastt` | GigaSTT model directory |
| `asr.punctuation`, `asr.itn` | `true` | punctuation and number normalisation (`true`/`false` or `auto`/`on`/`off`) |
| `asr.vad` | `true` | voice-activity detection |
| `diarization.model`, `diarization.revision` | Community-1 and its pinned revision | must equal the pinned values or the run is refused |
| `diarization.model_dir` | `models/pyannote` | Hugging Face cache directory for the model |
| `diarization.device` | `cpu` | the pinned PyTorch build is CPU-only |
| `diarization.num_speakers` | `4` | number of speakers given to pyannote |
| `merge.max_turn_gap` | `1.5` | seconds of silence that split a turn |
| `merge.nearest_max_gap` | `0.5` | how far the nearest segment may be before a word becomes `UNKNOWN` |
| `output.formats` | `json, md, txt, srt, vtt` | formats written |
| `output.mark_uncertain_words` | `true` | add the `спикер под вопросом` marker |
| `output.subtitle_max_seconds`, `output.subtitle_max_chars` | `6`, `84` | subtitle cue limits; `0` disables a limit |

**What triggers `--force`.** The result-affecting options are the speaker count, both audio filters, the model variant, punctuation, ITN, VAD, the device, the two merge gaps, `--allow-downmix` and the pinned model and revision. If a manifest exists and any of them differ, `run` stops with `Inference/merge options changed; use --force to rebuild.`. PyTorch thread counts, directories and the `output.*` options do not trigger it: output options are applied in the render stage, which runs every time.

**What `run.ps1` overrides.** The wrapper always passes `--output-root`, `--num-speakers` and `--model-dir` (and `--config`) explicitly. Through `run.ps1` the output directory, the speaker count and the GigaSTT model directory therefore come from the script's parameters (`-OutputRoot`, `-NumSpeakers`, default 4, and `models\gigastt`), not from `project.output_root`, `diarization.num_speakers` or `asr.model_dir` in the YAML. To use the YAML values, call the CLI directly.

Environment variables read by the code: `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) for pyannote; `HF_HOME` is used by the Hugging Face library and `.env.example` points it at `models/huggingface`. By default the code also sets `PYANNOTE_METRICS_ENABLED=0`, `HF_HUB_DISABLE_TELEMETRY=1` and `DO_NOT_TRACK=1` before importing pyannote; a value already present in your environment is kept.

## Command reference

### PowerShell scripts (`scripts/`)

All scripts stop on the first error. The scripts print their own messages in Russian; the Python CLI prints English.

| Script | Parameters | What it does |
|---|---|---|
| `install.ps1` | `-InstallFfmpeg`, `-SkipSync` | requires 64-bit Windows; installs `uv` through winget if missing; installs a managed Python 3.11; creates `.venv` and installs the pinned CPU dependencies (`uv sync`); with `-InstallFfmpeg` installs ffmpeg through winget; creates `media`, `models`, `tools\bin`, `tools\downloads` |
| `download-models.ps1` | `-SkipGigaStt`, `-SkipPyannote`, `-Force` | downloads GigaSTT v2.21.0 for Windows x64 (up to 3 attempts), checks its SHA-256 against `tools/tools.lock.json`, unpacks it, runs `gigastt download` for RNNT INT8 (up to 3 attempts), runs a one-second silent probe so the punctuation and VAD models are fetched now, then preloads the pyannote model at the pinned revision (needs `HF_TOKEN`) |
| `doctor.ps1` | none | checks `uv`, `ffmpeg`, `ffprobe`, the GigaSTT executable and `HF_TOKEN`, then runs the CLI doctor (below) |
| `run.ps1` | `-InputAudio` (one or more files), `-OutputRoot`, `-Config`, `-SpeakerMap`, `-NumSpeakers` (1–32, default 4), `-AllowDownmix`, `-StrictSpeakers`, `-TorchThreads`, `-TorchInteropThreads` | processes the files one at a time, sets `GIGASTT_OFFLINE=1` and the GigaSTT punctuation/VAD model directories, and calls `fourvoices run`; has no `-Force` |
| `rerender.ps1` | `-JobDir`, `-SpeakerMap` (both required) | calls `fourvoices render` for a finished job |
| `evaluate.ps1` | `-JobDir` (positional), `-Reference` (required), `-Start`, `-End` | calls `fourvoices evaluate`: scores a finished job against a reference transcript and writes `evaluation.json` into the job directory |
| `test.ps1` | none | runs the hygiene check, then `ruff check src tests`, then `pytest` |
| `check-repo-hygiene.ps1` | none | fails if audio files, `media/`, `transcripts/`, `secrets/`, `models/`, `tools/bin/`, `tools/downloads/`, `.env` files (except `.env.example`) or text resembling a Hugging Face token (`hf_` followed by 20 or more letters or digits) are tracked or not ignored |

`scripts/common.ps1` holds the shared helpers (repository root, `.env` loading, `uv` invocation); it is not run directly.

### The CLI

The scripts call it as `uv run --python 3.11 python -m fourvoices.cli <command>`; `pyproject.toml` also declares a `fourvoices` console script. Exit codes: `0` success, `2` a reported error (`error: …` on stderr, also a failed doctor), `130` interrupted with Ctrl+C (completed stages can be resumed).

| Command | Options |
|---|---|
| `run` | `--input` (required), `--output-root`, `--config`, `--gigastt-exe`, `--model-dir`, `--num-speakers`, `--speaker-map`, `--speaker-name LABEL=NAME` (repeatable), `--allow-downmix`, `--strict-speakers`, `--force`, `--ffmpeg`, `--ffprobe`, `--model-variant`, `--punctuation auto\|on\|off`, `--itn auto\|on\|off`, `--no-vad`, `--encoder-threads`, `--torch-threads`, `--torch-interop-threads`, `--device`, `--max-turn-gap`, `--nearest-max-gap`, `--formats`, `--output-stem` |
| `render` | `--job-dir` (required), `--speaker-map`, `--speaker-name LABEL=NAME`, `--formats`, `--output-stem`, `--subtitle-max-seconds`, `--subtitle-max-chars`, `--mark-uncertain` / `--no-mark-uncertain` |
| `diarize-preload` | `--model`, `--revision`, `--model-dir` (the pinned model and revision are the only accepted values) |
| `doctor` | `--config`, `--gigastt-exe`, `--model-dir` (accepted but not used) |
| `evaluate` | `--job-dir` (required), `--reference` (required), `--start`, `--end`, `--output` (default `<job>/evaluation.json`) |

Notes on the CLI:

- **`render` does not read `config/default.yaml`.** Its subtitle limits and uncertainty marks default to what the job was last rendered with (6 s, 84 characters and marks on for jobs rendered before these were recorded); `--subtitle-max-seconds`, `--subtitle-max-chars` and `--mark-uncertain` / `--no-mark-uncertain` override them. It refuses to run when the manifest no longer vouches for `merged.json`. Without `--formats` it re-renders the formats recorded in the job's manifest.
- **The CLI doctor checks** that `ffmpeg`, `ffprobe` and the GigaSTT executable can be found, that `pyannote.audio`, `torch`, `soundfile` and `PyYAML` are installed (it prints their versions), that a token is set, and that the configuration and the pinned model are valid. It does not check that the model files exist, that PyTorch is the CPU build, or the executable's hash.
- **Running the CLI directly.** `run.ps1` also sets `GIGASTT_OFFLINE=1`, `GIGASTT_PUNCT_MODEL_DIR` and `GIGASTT_VAD_MODEL_DIR` (to `models\gigastt\punct` and `models\gigastt\vad`) for the GigaSTT process and loads `.env` into the environment; when you call `fourvoices run` directly, set the same three variables yourself and make sure `HF_TOKEN` is set in the window (the Python code reads it only from the environment, never from `.env`). The example below does this.
- **Forcing a rebuild:**

```powershell
# From the repository root. These variables last for this PowerShell window.
# $env:HF_TOKEN must already be set (see "One-time access to the gated Hugging Face model").
$gigaModels = (Resolve-Path .\models\gigastt).Path
$env:GIGASTT_OFFLINE = '1'
$env:GIGASTT_PUNCT_MODEL_DIR = Join-Path $gigaModels 'punct'
$env:GIGASTT_VAD_MODEL_DIR = Join-Path $gigaModels 'vad'

uv run --python 3.11 python -m fourvoices.cli run --input .\media\recording.m4a `
    --output-root ..\gigastt-pyannote-output --config config\default.yaml `
    --gigastt-exe .\tools\bin\gigastt\2.21.0\gigastt.exe --model-dir .\models\gigastt `
    --force
```

## Useful commands

```powershell
# Check the tools on PATH, the GigaSTT executable, the token, the configuration and the installed Python packages
.\scripts\doctor.ps1

# Linter, tests, and protection against committing audio or tokens
.\scripts\test.ps1

# Re-download and re-verify GigaSTT only
.\scripts\download-models.ps1 -SkipPyannote -Force

# The gated pyannote model only
.\scripts\download-models.ps1 -SkipGigaStt

# Score a job against a reference fragment checked by ear
.\scripts\evaluate.ps1 <job-dir> -Reference reference.txt
```

## Security and reproducibility

- Never use `git add -f` for audio, `.env`, `models\` or `transcripts\`.
- Run `.\scripts\check-repo-hygiene.ps1` before committing.
- Setting `HF_TOKEN` for the current PowerShell process only is preferred. A
  local `.env` is supported as a fallback and is ignored by Git.
- The GigaSTT executable is pinned by URL and SHA-256; pyannote by full git revision. The GigaAM RNNT, punctuation and VAD weights are downloaded by `gigastt` itself, which checks them against SHA-256 digests built into the pinned executable; the RNNT files are checked again every time they are loaded. The pyannote weights are pinned by revision only.
- Telemetry is off: pyannote's usage metrics and Hugging Face Hub telemetry are disabled by the code itself, whatever the shell sets.
- The first download needs the network; once the models are in place, the
  processing itself is entirely local.

Automatic transcription is not a guarantee of verbatim accuracy. For quotes
that carry legal weight, always keep the original recording and its timecodes,
and check the words by ear.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `error: Set HF_TOKEN after accepting the Community-1 model conditions on Hugging Face.` | no token in this PowerShell window (it is needed for every run, not only for downloading) | set `$env:HF_TOKEN` again, or put it into a local `.env` |
| `Could not load Community-1. Confirm model access and HF_TOKEN.` or `Model download was denied.` | the model conditions were not accepted, or the token is wrong | accept the conditions on the model page and create a **Read** token |
| `error: Input has 2 channels (…). Inspect/listen to the channels first …` | multichannel input is refused on purpose | listen to the channels; use `-AllowDownmix` only if mixing cannot destroy useful speaker separation |
| `Inference/merge options changed; use --force to rebuild.` | an option that affects the result differs from the manifest | restore the option, or rebuild with `--force` through the CLI (`run.ps1` has no `-Force`) |
| `Existing job belongs to a different input; use --force.` | a job directory with the same name holds a different file | use `--force` only if you mean to replace it |
| `Configuration problems: unknown key '…'` | a typo or a stale key in the YAML | use one of the keys the message lists; see [Configuration reference](#configuration-reference) |
| `This release is pinned to … the configuration requests an unreviewed artifact.` | `diarization.model` or `revision` was changed | restore the pinned values |
| `ffmpeg is not available on PATH. Install FFmpeg and restart the shell.` | ffmpeg was just installed through winget | open a new PowerShell window, return to the repository and set the token again |
| `Не найден GigaSTT. Запустите .\scripts\download-models.ps1.` | the executable is not downloaded yet | run `.\scripts\download-models.ps1` |
| `SHA-256 GigaSTT не совпал…` | a damaged or substituted download | run `.\scripts\download-models.ps1 -SkipPyannote -Force` |
| `warning: Requested 4 speakers but pyannote returned N …` | a participant stayed almost silent, or there are not four speakers | the result is saved; pass `-NumSpeakers`, or `-StrictSpeakers` to make it an error (still saved) |
| `warning: these labels are not in the transcript and were ignored: …` | a speaker-map label does not exist in this job | use the labels the message lists |
| `GigaSTT JSON has no 'words' array; word timestamps are required.` and a `gigastt.json.rejected` file | GigaSTT returned something this program cannot use (for example a model variant without word timestamps) | inspect the `.rejected` file; keep a variant that emits word timestamps |
| `error: Cannot read …/manifest.json …` | a damaged manifest | fix or delete it; without a manifest the stages are recomputed rather than reused |
| the run is very slow or uses much memory | CPU-only, and diarization loads the whole recording into memory | see [Requirements](#requirements); tune `-TorchThreads`; process files one at a time (the script already does) |

## Status and known limits

The package is version 0.1.0.

- **Platform.** Windows 10/11 x64 only: the scripts are PowerShell and use winget; CI runs on `windows-latest`. No other platform is documented or tested.
- **Language and speakers.** Russian speech; defaults tuned for four speakers on one microphone. The speaker count is given, not inferred.
- **CPU only.** The pinned PyTorch build is `2.11.0+cpu`; there is no GPU path.
- **Files are processed one at a time** by `run.ps1`, on purpose.
- **Memory.** Diarization loads the whole recording; roughly 230 MB per hour of audio plus the model itself (see [Requirements](#requirements)).
- **Pins are partial.** The GigaSTT executable (URL and SHA-256) and the pyannote revision are pinned; the GigaAM RNNT, punctuation and VAD weights are fetched by `gigastt`, which checks them against SHA-256 digests built into the pinned executable, but this repository does not pin them itself, and the pyannote weights are pinned by revision only. Dependency versions are pinned in `pyproject.toml` and `uv.lock`.
- **Token on every run.** `HF_TOKEN` is required each time diarization runs, even when all weights are on disk.
- **Network.** Installation and model download need the network. After that, processing is local; the code turns pyannote telemetry off by default, but a value already set in your environment wins.
- **Accuracy.** Automatic transcription is not verbatim-accurate; overlapping speech and interruptions need manual proofreading.
- **Tests.** A pytest suite covers the CLI and resume behaviour, diarization handling, the GigaSTT wrapper, the merge (including equivalence with reference implementations) and rendering; CI runs ruff, pytest and the hygiene check. The suites use stand-ins for the models, so a full end-to-end run on a real recording needs the real models and a token.
- **No release process is documented**; the repository is used from a checkout.

## Documentation map

| Where | What it covers |
|---|---|
| This README | operation of the whole pipeline: setup, running, resuming, naming speakers, configuration, commands, troubleshooting |
| [README.ru.md](README.ru.md) | Russian translation of this document (the English one is authoritative) |
| [config/default.yaml](config/default.yaml) | every setting with comments on why it is what it is |
| [config/speaker-map.example.yaml](config/speaker-map.example.yaml) | template for the speaker map |
| [.env.example](.env.example) | local fallback for `HF_TOKEN` and `HF_HOME` |
| [tools/tools.lock.json](tools/tools.lock.json) | the pinned GigaSTT release (URL, SHA-256) and pyannote revision |
| `src/fourvoices/` | `audio.py`, `gigastt.py`, `diarize.py`, `merge.py`, `render.py`, `cli.py` |
| `scripts/` | the PowerShell wrappers |
| `tests/` | the pytest suite |
| [.github/workflows/tests.yml](.github/workflows/tests.yml) | the CI workflow |

