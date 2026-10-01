"""Thin, observable wrapper around the GigaSTT command-line client."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .progress import Reporter, format_duration


class GigaSTTError(RuntimeError):
    """GigaSTT failed or returned an incompatible result."""


# GigaSTT logs through `tracing` and colours its output even when piped.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_LEVEL = re.compile(r"\b(WARN|ERROR)\b")
# Logged on every run because the pipeline deliberately does not download
# GigaSTT's own speaker model: pyannote does the diarization here.
_EXPECTED_NOISE = ("gigastt_core::inference::diarization",)
_MAX_REPORTED_PROBLEMS = 10
# Logged once per decoded window (2.15.0 and 2.21.0). With VAD the number of
# windows is not known in advance, so progress is a count, not a percentage.
_WINDOW_DONE = "Decoded tokens"

# Below this many words a transcript may legitimately carry no punctuation.
_PUNCTUATION_CHECK_MIN_WORDS = 200
_SENTENCE_MARKS = frozenset(".,?!…;:")
_NUMBER_SEPARATOR = re.compile(r"(?<=\d)[.,:](?=\d)")


def gigastt_version(executable: str = "gigastt") -> str | None:
    """Version reported by ``gigastt --version``, or None if it cannot be run.

    Recorded next to the ASR result so a different binary is noticed on resume:
    releases change the words themselves, not only the speed.
    """

    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"\b(\d+\.\d+\.\d+\S*)", completed.stdout)
    return match.group(1) if match else None


def log_problems(log: str) -> list[str]:
    """WARN and ERROR lines from GigaSTT's log, without colour codes.

    GigaSTT reports some degraded results only here and still exits with 0:
    2.15.0 logged "Punctuation restore failed, returning bare text" and returned
    an unpunctuated transcript once the text outgrew the punctuation model's
    2048-token window, i.e. somewhere between ten and thirty minutes of speech.
    """

    problems: list[str] = []
    for line in _ANSI.sub("", log).splitlines():
        text = line.strip()
        if _LOG_LEVEL.search(text) and not any(item in text for item in _EXPECTED_NOISE):
            problems.append(text)
    return problems


def punctuation_missing(payload: Mapping[str, Any], punctuation: str) -> bool:
    """Whether punctuation was requested but the text plainly came back without it.

    A long transcript with not a single sentence mark is not a plausible result
    of punctuation restoration; it is a restoration step that gave up.
    """

    if punctuation != "on":
        return False
    words = payload.get("words") or []
    if len(words) < _PUNCTUATION_CHECK_MIN_WORDS:
        return False
    # ITN runs before punctuation and writes "1,5" or "10:30" on its own, so a
    # separator between digits is no evidence that punctuation ran.
    text = _NUMBER_SEPARATOR.sub("", str(payload.get("text", "")))
    return not any(character in _SENTENCE_MARKS for character in text)


def parse_gigastt_json(value: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize GigaSTT's JSON export."""

    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GigaSTTError("GigaSTT output is not valid JSON.") from exc
    words = payload.get("words")
    if not isinstance(words, list):
        raise GigaSTTError("GigaSTT JSON has no 'words' array; word timestamps are required.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(words):
        if not isinstance(item, Mapping):
            raise GigaSTTError(f"GigaSTT word #{index} is not an object.")
        try:
            text = str(item.get("word", item.get("text", ""))).strip()
            start = float(item["start"])
            end = float(item["end"])
            confidence = float(item.get("confidence") or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise GigaSTTError(f"Malformed timing in GigaSTT word #{index}.") from exc
        if not text:
            continue
        if start < 0 or end < start:
            raise GigaSTTError(f"Invalid interval in GigaSTT word #{index}: {start}..{end}")
        normalized.append(
            {"word": text, "start": start, "end": end, "confidence": confidence}
        )
    payload["words"] = normalized
    payload["text"] = str(payload.get("text") or " ".join(w["word"] for w in normalized))
    if "duration_s" not in payload:
        payload["duration_s"] = float(
            payload.get("duration", max((w["end"] for w in normalized), default=0.0))
        )
    return payload


def transcribe(
    audio_path: str | Path,
    output_json: str | Path,
    *,
    executable: str = "gigastt",
    model_variant: str = "rnnt",
    model_dir: str | Path | None = None,
    punctuation: str = "on",
    itn: str = "on",
    vad: bool = True,
    encoder_threads: int | None = None,
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run offline GigaSTT and atomically persist its timestamped JSON.

    GigaSTT's log is kept next to the result (``gigastt.json`` ->
    ``gigastt.log``), success or not, and its warnings are repeated on stderr:
    it is the only place some degraded results are reported.
    """

    source = Path(audio_path)
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    log_path = destination.with_suffix(".log")
    command = [
        executable,
        "transcribe",
        str(source),
        "--model-variant",
        model_variant,
        "--punctuation",
        punctuation,
        "--itn",
        itn,
        "--format",
        "json",
        "--output",
        str(temporary),
        "--word-timestamps",
    ]
    if model_dir is not None:
        command.extend(["--model-dir", str(model_dir)])
    if vad:
        command.append("--vad")
    if encoder_threads is not None:
        if encoder_threads < 1:
            raise ValueError("encoder_threads must be positive")
        command.extend(["--encoder-intra-threads", str(encoder_threads)])
    command.extend(extra_args)
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    reporter = Reporter("GigaSTT")
    windows = 0

    def on_line(line: str) -> None:
        nonlocal windows
        if _WINDOW_DONE in line:
            windows += 1
            elapsed = format_duration(reporter.elapsed())
            reporter.update(f"{windows} window(s) decoded, {elapsed} elapsed")

    try:
        returncode, log = _execute(command, process_env, on_line)
    except FileNotFoundError as exc:
        raise GigaSTTError(
            f"GigaSTT executable '{executable}' was not found on PATH."
        ) from exc
    _write_log(log_path, log)
    if returncode != 0:
        # The log opens with model-loading chatter; the reason is at the end.
        lines = _ANSI.sub("", log).strip().splitlines()
        detail = "\n".join(lines[-5:])
        raise GigaSTTError(
            f"GigaSTT exited with code {returncode} (full log: {log_path.name}): {detail}"
        )
    problems = log_problems(log)
    for line in problems[:_MAX_REPORTED_PROBLEMS]:
        print(f"warning: GigaSTT: {line}", file=sys.stderr, flush=True)
    if len(problems) > _MAX_REPORTED_PROBLEMS:
        hidden = len(problems) - _MAX_REPORTED_PROBLEMS
        print(
            f"warning: GigaSTT: {hidden} more warning(s) in {log_path.name}",
            file=sys.stderr,
            flush=True,
        )
    if not temporary.is_file():
        # stdout is GigaSTT's log, not a fallback copy of the JSON.
        raise GigaSTTError(
            f"GigaSTT exited with code 0 but wrote no JSON; see {log_path.name}."
        )
    raw = temporary.read_text(encoding="utf-8-sig")
    try:
        payload = parse_gigastt_json(raw)
    except GigaSTTError as exc:
        # Keep the unparsed output: if GigaSTT changes its schema, this file is
        # the only evidence of what it actually produced.
        rejected = destination.with_name(destination.name + ".rejected")
        rejected.write_text(raw, encoding="utf-8")
        temporary.unlink(missing_ok=True)
        raise GigaSTTError(f"{exc} Raw output kept in {rejected.name}.") from exc
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = reporter.elapsed()
    audio = float(payload.get("duration", payload.get("duration_s", 0.0)) or 0.0)
    speed = f", {audio / elapsed:.1f}x real time" if audio > 0 and elapsed > 0 else ""
    reporter.update(f"done in {format_duration(elapsed)}{speed}", force=True)
    return payload


def _execute(
    command: Sequence[str], env: Mapping[str, str], on_line: Callable[[str], None]
) -> tuple[int, str]:
    """Run GigaSTT, hand each output line to ``on_line``; return the exit code and log.

    GigaSTT's `tracing` log goes to stdout, not stderr (checked against 2.15.0
    and 2.21.0); with ``--output`` the JSON goes to the file. stderr is merged
    in so a panic or a release that moves the log is kept as well. Lines are
    read as they come, which is what lets a long recognition report progress.
    """

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env),
    )
    lines: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            on_line(line)
        returncode = process.wait()
    finally:
        # Ctrl+C or a failing callback must not leave GigaSTT running.
        if process.poll() is None:
            process.kill()
            process.wait()
    return returncode, "".join(lines)


def _write_log(path: Path, text: str | None) -> None:
    path.write_text(_ANSI.sub("", text or ""), encoding="utf-8")


def load_transcript(path: str | Path) -> dict[str, Any]:
    return parse_gigastt_json(Path(path).read_text(encoding="utf-8-sig"))
