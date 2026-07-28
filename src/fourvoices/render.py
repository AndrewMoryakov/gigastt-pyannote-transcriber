"""Deterministic human- and machine-readable transcript renderers."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .merge import join_words


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


def _markers(turn: Mapping[str, Any], *, mark_uncertain: bool) -> list[str]:
    """Human-readable warnings attached to a turn, in fixed order."""

    markers = []
    if turn.get("overlap"):
        markers.append("перекрытие речи")
    if mark_uncertain and turn.get("uncertain"):
        markers.append("спикер под вопросом")
    return markers


def render_txt(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    mark_uncertain: bool = True,
) -> str:
    lines = []
    for turn in data.get("turns", []):
        found = _markers(turn, mark_uncertain=mark_uncertain)
        marker = f" [{', '.join(found)}]" if found else ""
        lines.append(
            f"[{clock(turn['start'])}–{clock(turn['end'])}] "
            f"{_display_name(str(turn['speaker']), names)}{marker}: {turn['text']}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def render_md(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    mark_uncertain: bool = True,
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
        found = _markers(turn, mark_uncertain=mark_uncertain)
        marker = "".join(f" · **{item}**" for item in found)
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


# Broadcast-subtitle convention: two lines of ~42 characters, on screen for a
# few seconds. A whole turn can last minutes, which no player can display.
DEFAULT_CUE_SECONDS = 6.0
DEFAULT_CUE_CHARS = 84


def _turn_words(
    turn: Mapping[str, Any], words: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Words belonging to a turn, using the index range merge recorded.

    Returns nothing unless the whole range is present. A partial match means the
    JSON no longer matches its turns, and splitting on it would drop text from
    the subtitle without saying so.
    """

    if not words:
        return []
    first, last = turn.get("word_start"), turn.get("word_end")
    if first is None or last is None:
        return []
    selected = sorted(
        (word for word in words if first <= int(word.get("index", -1)) <= last),
        key=lambda word: int(word["index"]),
    )
    if len(selected) != int(last) - int(first) + 1:
        return []
    return selected


def split_turn(
    turn: Mapping[str, Any],
    words: Sequence[Mapping[str, Any]] = (),
    *,
    max_seconds: float = DEFAULT_CUE_SECONDS,
    max_chars: int = DEFAULT_CUE_CHARS,
    prefix_chars: int = 0,
) -> list[dict[str, Any]]:
    """Cut one turn into cues that fit a subtitle screen.

    Cuts land on word boundaries so every cue keeps real timings. Without word
    timings the turn is returned whole rather than split at guessed times: a
    wrong timestamp is worse than a long cue.
    """

    text = str(turn.get("text", ""))
    whole = [{"start": float(turn["start"]), "end": float(turn["end"]), "text": text}]
    if max_seconds <= 0 and max_chars <= 0:
        return whole
    selected = _turn_words(turn, words)
    if not selected:
        return whole
    budget = max(1, max_chars - prefix_chars) if max_chars > 0 else 0
    cues: list[dict[str, Any]] = []
    group: list[Mapping[str, Any]] = []

    def flush() -> None:
        if not group:
            return
        cues.append(
            {
                "start": float(group[0]["start"]),
                "end": max(float(word["end"]) for word in group),
                "text": join_words(
                    str(word.get("display_word", word.get("word", ""))) for word in group
                ),
            }
        )
        group.clear()

    for word in selected:
        if group:
            candidate = group + [word]
            spoken = float(word["end"]) - float(candidate[0]["start"])
            length = len(
                join_words(
                    str(item.get("display_word", item.get("word", "")))
                    for item in candidate
                )
            )
            if (max_seconds > 0 and spoken > max_seconds) or (
                budget > 0 and length > budget
            ):
                flush()
        group.append(word)
    flush()
    if not cues:
        return whole
    # Rendering the turn's own text keeps a single cue byte-identical to the
    # unsplit output; only genuinely split turns are rebuilt from words.
    return whole if len(cues) == 1 else cues


def _subtitle_lines(
    turns: Sequence[Mapping[str, Any]],
    *,
    names: Mapping[str, str] | None,
    srt: bool,
    max_line_chars: int,
    mark_uncertain: bool,
    words: Sequence[Mapping[str, Any]] = (),
    max_cue_seconds: float = DEFAULT_CUE_SECONDS,
    max_cue_chars: int = DEFAULT_CUE_CHARS,
) -> list[str]:
    lines: list[str] = []
    counter = 0
    for turn in turns:
        label = _display_name(str(turn["speaker"]), names)
        found = _markers(turn, mark_uncertain=mark_uncertain)
        marker = f" [{', '.join(found)}]" if found else ""
        prefix = f"{label}{marker}: "
        for cue in split_turn(
            turn,
            words,
            max_seconds=max_cue_seconds,
            max_chars=max_cue_chars,
            prefix_chars=len(prefix),
        ):
            counter += 1
            if srt:
                lines.append(str(counter))
            lines.append(
                f"{_subtitle_time(cue['start'], srt=srt)} --> "
                f"{_subtitle_time(cue['end'], srt=srt)}"
            )
            wrapped = textwrap.wrap(
                prefix + cue["text"],
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
    mark_uncertain: bool = True,
    max_line_chars: int = 42,
    max_cue_seconds: float = DEFAULT_CUE_SECONDS,
    max_cue_chars: int = DEFAULT_CUE_CHARS,
) -> str:
    return "\n".join(
        _subtitle_lines(
            list(data.get("turns", [])),
            names=names,
            srt=True,
            max_line_chars=max_line_chars,
            mark_uncertain=mark_uncertain,
            words=list(data.get("words", [])),
            max_cue_seconds=max_cue_seconds,
            max_cue_chars=max_cue_chars,
        )
    ).rstrip() + "\n"


def render_vtt(
    data: Mapping[str, Any],
    *,
    names: Mapping[str, str] | None = None,
    mark_uncertain: bool = True,
    max_line_chars: int = 42,
    max_cue_seconds: float = DEFAULT_CUE_SECONDS,
    max_cue_chars: int = DEFAULT_CUE_CHARS,
) -> str:
    body = _subtitle_lines(
        list(data.get("turns", [])),
        names=names,
        srt=False,
        max_line_chars=max_line_chars,
        mark_uncertain=mark_uncertain,
        words=list(data.get("words", [])),
        max_cue_seconds=max_cue_seconds,
        max_cue_chars=max_cue_chars,
    )
    return "WEBVTT\n\n" + "\n".join(body).rstrip() + "\n"


def render_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def render(
    data: Mapping[str, Any],
    format_name: str,
    *,
    names: Mapping[str, str] | None = None,
    mark_uncertain: bool = True,
    **subtitle_limits: Any,
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
    if key not in functions:
        raise ValueError(f"Unsupported output format: {format_name}")
    extra = subtitle_limits if key in {"srt", "vtt"} else {}
    return functions[key](data, names=names, mark_uncertain=mark_uncertain, **extra)


def write_outputs(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "transcript",
    formats: Sequence[str] = ("md", "txt", "srt", "vtt", "json"),
    names: Mapping[str, str] | None = None,
    mark_uncertain: bool = True,
    **subtitle_limits: Any,
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for format_name in formats:
        suffix = format_name.lower().lstrip(".")
        path = directory / f"{stem}.{suffix}"
        path.write_text(
            render(
                data,
                suffix,
                names=names,
                mark_uncertain=mark_uncertain,
                **subtitle_limits,
            ),
            encoding="utf-8",
        )
        written.append(path)
    return written
