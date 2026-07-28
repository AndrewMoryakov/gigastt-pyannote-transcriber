"""Thin, observable wrapper around the GigaSTT command-line client."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class GigaSTTError(RuntimeError):
    """GigaSTT failed or returned an incompatible result."""


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
    """Run offline GigaSTT and atomically persist its timestamped JSON."""

    source = Path(audio_path)
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
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
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_env,
        )
    except FileNotFoundError as exc:
        raise GigaSTTError(
            f"GigaSTT executable '{executable}' was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise GigaSTTError(f"GigaSTT exited with code {exc.returncode}: {detail}") from exc
    if temporary.is_file():
        raw = temporary.read_text(encoding="utf-8-sig")
    else:
        raw = completed.stdout
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
    return payload


def load_transcript(path: str | Path) -> dict[str, Any]:
    return parse_gigastt_json(Path(path).read_text(encoding="utf-8-sig"))
