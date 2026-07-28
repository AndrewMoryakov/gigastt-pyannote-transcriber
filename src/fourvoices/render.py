"""Deterministic human- and machine-readable transcript renderers."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence


def clock(seconds: float, *, milliseconds: bool = False) -> str:
    milliseconds_total = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _display_name(speaker: str, names: Mapping[str, str] | None) -> str:
    return str((names or {}).get(speaker, speaker))


def render_txt(data: Mapping[str, Any], *, names: Mapping[str, str] | None = None) -> str:
    lines = []
    for turn in data.get("turns", []):
        marker = " [перекрытие речи]" if turn.get("overlap") else ""
        lines.append(
            f"[{clock(turn['start'])}–{clock(turn['end'])}] "
            f"{_display_name(str(turn['speaker']), names)}{marker}: {turn['text']}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def render_md(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    title: str = "Транскрипция аудиозаписи",
) -> str:
    lines = [
        f"# {title}",
        "",
        "> Автоматическая транскрипция. Неразборчивые места и распределение "
        "говорящих требуют проверки по аудиозаписи.",
        "",
    ]
    for turn in data.get("turns", []):
        marker = " · **перекрытие речи**" if turn.get("overlap") else ""
        lines.extend(
            [
                f"**[{clock(turn['start'])}–{clock(turn['end'])}] "
                f"{_display_name(str(turn['speaker']), names)}{marker}**",
                "",
                str(turn["text"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _subtitle_time(seconds: float, *, srt: bool) -> str:
    value = clock(seconds, milliseconds=True)
    return value.replace(".", ",") if srt else value


def _subtitle_lines(
    turns: Sequence[Mapping[str, Any]],
    *,
    names: Mapping[str, str] | None,
    srt: bool,
    max_line_chars: int,
) -> list[str]:
    lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        if srt:
            lines.append(str(index))
        lines.append(
            f"{_subtitle_time(float(turn['start']), srt=srt)} --> "
            f"{_subtitle_time(float(turn['end']), srt=srt)}"
        )
        label = _display_name(str(turn["speaker"]), names)
        marker = " [перекрытие речи]" if turn.get("overlap") else ""
        value = f"{label}{marker}: {turn['text']}"
        wrapped = textwrap.wrap(
            value,
            width=max_line_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
        lines.append("")
    return lines


def render_srt(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    max_line_chars: int = 80,
) -> str:
    return "\n".join(
        _subtitle_lines(
            list(data.get("turns", [])),
            names=names,
            srt=True,
            max_line_chars=max_line_chars,
        )
    ).rstrip() + "\n"


def render_vtt(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    max_line_chars: int = 80,
) -> str:
    body = _subtitle_lines(
        list(data.get("turns", [])),
        names=names,
        srt=False,
        max_line_chars=max_line_chars,
    )
    return "WEBVTT\n\n" + "\n".join(body).rstrip() + "\n"


def render_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def render(
    data: Mapping[str, Any],
    format_name: str,
    *,
    names: Mapping[str, str] | None = None,
) -> str:
    key = format_name.lower().lstrip(".")
    functions = {
        "txt": render_txt,
        "md": render_md,
        "srt": render_srt,
        "vtt": render_vtt,
    }
    if key == "json":
        return render_json(data)
    try:
        return functions[key](data, names=names)
    except KeyError as exc:
        raise ValueError(f"Unsupported output format: {format_name}") from exc


def write_outputs(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "transcript",
    formats: Sequence[str] = ("md", "txt", "srt", "vtt", "json"),
    names: Mapping[str, str] | None = None,
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for format_name in formats:
        suffix = format_name.lower().lstrip(".")
        path = directory / f"{stem}.{suffix}"
        path.write_text(render(data, suffix, names=names), encoding="utf-8")
        written.append(path)
    return written
