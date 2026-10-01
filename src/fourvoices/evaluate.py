"""Score a finished job against a reference transcript checked by ear.

The reference is plain text in the shape of ``transcript.txt``: one turn per
line, ``[hh:mm:ss–hh:mm:ss] Name: text``. The timestamp is optional, markers
such as ``[перекрытие речи]`` after the name are ignored, and a line without a
``Name:`` continues the previous speaker. Nobody transcribes two hours by hand,
so a reference usually covers a fragment: only hypothesis words whose midpoint
falls inside the reference's time span are scored.

Two numbers come from one word alignment:

* WER — substitutions, deletions and insertions over reference words, after
  lowercasing, ``ё`` -> ``е`` and stripping punctuation;
* speaker accuracy — among aligned word pairs (matches and substitutions), the
  share whose pyannote cluster maps to the reference speaker, under the
  one-to-one cluster -> name mapping that agrees with the reference most.
  ``UNKNOWN`` never maps, so it always counts as an error. Inserted and deleted
  words are left out here; WER already counts them.

Nothing is written into the job except ``evaluation.json``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .merge import UNKNOWN_SPEAKER

# Pure-Python edit distance over n x m cells: a few thousand words a side is
# instant, a whole recording against itself is not. Above this the table is
# narrowed to a diagonal band (see align), and the band stops growing at the
# second limit.
EXACT_ALIGNMENT_CELLS = 4_000_000
MAX_ALIGNMENT_CELLS = 40_000_000
INITIAL_BAND = 64

# How many differing places the printed summary quotes.
SHOWN_DIFFERENCES = 10

# transcript.txt prints whole seconds, truncated, so a turn ending at 9.8 s is
# shown as 00:00:09; widen the window's end by one second to keep it.
TIMESTAMP_RESOLUTION = 1.0

# A speaker label is at most this many words; longer text before ": " is speech.
MAX_NAME_WORDS = 4

BIASED_REFERENCE_NOTE = (
    "A reference made by correcting the pipeline's own transcript is optimistic: "
    "errors that read naturally are easy to miss. For an unbiased figure, "
    "transcribe a short fragment from scratch."
)

_TIME = r"(\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)"
_LINE = re.compile(
    rf"^\s*(?:\[{_TIME}\s*[–—-]\s*{_TIME}\]\s*)?"
    r"(?:(?P<name>[^\[\]:]+?)\s*(?:\[[^\]]*\]\s*)*:\s+)?"
    r"(?P<text>.*)$"
)
_BRACKETS = re.compile(r"\[[^\]]*\]")
_EDGE_PUNCTUATION = re.compile(r"^[^\w]+|[^\w]+$")


class ReferenceError(ValueError):
    """The reference text cannot be scored."""


def normalize(token: str) -> str:
    """Lowercase, ``ё`` -> ``е``, punctuation stripped; inner hyphens stay."""

    return _EDGE_PUNCTUATION.sub("", token.lower().replace("ё", "е"))


def tokens(text: str) -> list[str]:
    return [word for word in (normalize(part) for part in text.split()) if word]


def parse_seconds(value: str) -> float:
    """``hh:mm:ss[.fff]``, ``mm:ss`` or plain seconds."""

    parts = value.strip().replace(",", ".").split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ReferenceError(f"Not a time: {value!r}") from exc
    if not 1 <= len(numbers) <= 3:
        raise ReferenceError(f"Not a time: {value!r}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


@dataclass(frozen=True)
class Reference:
    words: list[tuple[str, str]]  # (normalised word, speaker)
    start: float | None
    end: float | None


def parse_reference(text: str) -> Reference:
    words: list[tuple[str, str]] = []
    speaker: str | None = None
    starts: list[float] = []
    ends: list[float] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        assert match is not None  # every group is optional
        start, end, name = match.group(1), match.group(2), match.group("name")
        body = match.group("text")
        if name is not None and len(name.split()) > MAX_NAME_WORDS:
            # "Итак, подведём итоги по проекту: ..." is speech, not a label.
            body = line[match.start("name") :]
            name = None
        if start is not None:
            starts.append(parse_seconds(start))
            ends.append(parse_seconds(end))
        if name is not None:
            speaker = name.strip()
        line_words = tokens(_BRACKETS.sub(" ", body))
        if line_words and speaker is None:
            raise ReferenceError(
                f"Line {number} has words but no speaker yet; start it with 'Name: '."
            )
        words.extend((word, speaker) for word in line_words if speaker is not None)
    if not words:
        raise ReferenceError("The reference contains no words.")
    if starts:
        return Reference(words, min(starts), max(ends) + TIMESTAMP_RESOLUTION)
    return Reference(words, None, None)


def hypothesis_words(
    merged: Mapping[str, Any], start: float | None = None, end: float | None = None
) -> list[tuple[str, str, float]]:
    """(word, speaker, start) for the job's displayed words inside the window.

    Words are normalised like the reference, and kept if their midpoint falls
    inside ``[start, end)``.
    """

    result: list[tuple[str, str, float]] = []
    for word in merged.get("words", []):
        middle = (float(word["start"]) + float(word["end"])) / 2
        if start is not None and middle < start:
            continue
        if end is not None and middle >= end:
            continue
        shown = str(word.get("display_word", word.get("word", "")))
        for token in tokens(shown):
            result.append(
                (token, str(word.get("speaker", UNKNOWN_SPEAKER)), float(word["start"]))
            )
    return result


# --------------------------------------------------------------------- alignment

# Each step is (operation, reference index | None, hypothesis index | None).
Step = tuple[str, int | None, int | None]


def _alignment(
    ref: Sequence[str], hyp: Sequence[str], band: int | None = None
) -> tuple[list[Step], int]:
    """Levenshtein alignment and its cost, optionally within ``|i - j| <= band``.

    Ties prefer a diagonal step, then a deletion. Cells outside the band count
    as unreachable, so a banded result is the cheapest path inside the band.
    """

    rows, cols = len(ref), len(hyp)
    unreachable = rows + cols + 1
    # Without a band every cell is reachable: |i - j| never exceeds the longer side.
    width = max(rows, cols) if band is None else band
    # costs[i] holds row i for columns lows[i] .. lows[i] + len(costs[i]) - 1.
    lows: list[int] = []
    costs: list[list[int]] = []

    def cell(i: int, j: int) -> int:
        offset = j - lows[i]
        row = costs[i]
        return row[offset] if 0 <= offset < len(row) else unreachable

    for i in range(rows + 1):
        low, high = max(0, i - width), min(cols, i + width)
        row = [unreachable] * (high - low + 1)
        word = ref[i - 1] if i else None
        for j in range(low, high + 1):
            if i == 0:
                value = j
            elif j == 0:
                value = i
            else:
                value = min(
                    cell(i - 1, j - 1) + (word != hyp[j - 1]),
                    cell(i - 1, j) + 1,
                    (row[j - 1 - low] if j - 1 >= low else unreachable) + 1,
                )
            row[j - low] = value
        lows.append(low)
        costs.append(row)

    distance = cell(rows, cols)
    steps: list[Step] = []
    i, j = rows, cols
    while i or j:
        same = i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]
        here = cell(i, j)
        if i and j and here == cell(i - 1, j - 1) + (not same):
            steps.append(("equal" if same else "substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and here == cell(i - 1, j) + 1:
            steps.append(("delete", i - 1, None))
            i -= 1
        elif j:
            steps.append(("insert", None, j - 1))
            j -= 1
        else:  # only reachable through a bug in the table; never loop on it
            raise RuntimeError(f"alignment backtrace stuck at row {i}")
    steps.reverse()
    return steps, distance


def align(ref: Sequence[str], hyp: Sequence[str]) -> tuple[list[Step], str]:
    """Minimum-edit word alignment and whether it is proven minimal.

    Small inputs get the full table. Larger ones are aligned inside a diagonal
    band that doubles until the distance found fits in it: a path that strays
    more than ``band`` cells off the diagonal costs more than ``band`` edits, so
    once the distance is at most ``band`` no path outside can beat it and the
    result is exact. If the work limit is reached first, the alignment is the
    best one inside the band, the WER is an upper bound, and the method says so.
    """

    rows, cols = len(ref), len(hyp)
    if rows * cols <= EXACT_ALIGNMENT_CELLS:
        return _alignment(ref, hyp)[0], "exact"
    band = max(abs(rows - cols), INITIAL_BAND)
    while True:
        steps, distance = _alignment(ref, hyp, band)
        if distance <= band or band >= max(rows, cols):
            return steps, "exact"
        if (rows + 1) * (4 * band + 1) > MAX_ALIGNMENT_CELLS:
            return steps, "banded (WER is an upper bound)"
        band *= 2


# --------------------------------------------------------------- speaker mapping


def best_mapping(agreement: Mapping[tuple[str, str], int]) -> dict[str, str]:
    """One-to-one cluster -> reference name mapping with the most agreeing words.

    Exact, by dynamic programming over subsets of reference names. ``UNKNOWN``
    is never mapped.
    """

    clusters = sorted({c for c, _ in agreement if c != UNKNOWN_SPEAKER})
    names = sorted({n for _, n in agreement})
    if len(names) > 16:
        raise ReferenceError("More than 16 reference speakers; cannot map them exactly.")
    # best[mask] = (score, mapping) using clusters seen so far and names in mask.
    best: dict[int, tuple[int, tuple[tuple[str, str], ...]]] = {0: (0, ())}
    for cluster in clusters:
        following = dict(best)  # the cluster may also stay unmapped
        for mask, (score, mapping) in best.items():
            for index, name in enumerate(names):
                if mask & (1 << index):
                    continue
                gained = agreement.get((cluster, name), 0)
                candidate = (score + gained, mapping + ((cluster, name),))
                key = mask | (1 << index)
                if key not in following or candidate[0] > following[key][0]:
                    following[key] = candidate
        best = following
    score, mapping = max(best.values(), key=lambda item: (item[0], item[1]))
    return {cluster: name for cluster, name in mapping if agreement.get((cluster, name), 0)}


# ----------------------------------------------------------------------- scoring


def differences(
    steps: Sequence[Step],
    reference: Sequence[tuple[str, str]],
    hypothesis: Sequence[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    """Runs of consecutive errors, each with where to listen in the recording.

    A run of missed words has no time of its own; it is placed at the next
    recognised word, or the previous one at the very end.
    """

    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    waiting: list[dict[str, Any]] = []  # runs with no time yet
    last_time: float | None = None
    for operation, r, h in steps:
        if h is not None:
            last_time = hypothesis[h][2]
            for run in waiting:
                run["at"] = last_time
            waiting.clear()
        if operation == "equal":
            current = None
            continue
        if current is None:
            current = {
                "at": last_time if h is not None else None,
                "reference": [],
                "hypothesis": [],
                "speaker": None,
            }
            runs.append(current)
            if h is None:
                waiting.append(current)
        if r is not None:
            current["reference"].append(reference[r][0])
            current["speaker"] = current["speaker"] or reference[r][1]
        if h is not None:
            current["hypothesis"].append(hypothesis[h][0])
    for run in waiting:
        run["at"] = last_time
    for run in runs:
        run["reference"] = " ".join(run["reference"])
        run["hypothesis"] = " ".join(run["hypothesis"])
    return runs


def score(
    reference: Sequence[tuple[str, str]], hypothesis: Sequence[tuple[str, str, float]]
) -> dict[str, Any]:
    ref_words = [item[0] for item in reference]
    hyp_words = [item[0] for item in hypothesis]
    steps, method = align(ref_words, hyp_words)
    counts = Counter(operation for operation, _, _ in steps)

    agreement: Counter[tuple[str, str]] = Counter()
    for operation, r, h in steps:
        if operation in ("equal", "substitute"):
            assert r is not None and h is not None
            agreement[(hypothesis[h][1], reference[r][1])] += 1
    mapping = best_mapping(agreement)

    per_speaker: dict[str, dict[str, int]] = defaultdict(
        lambda: {"words": 0, "substitutions": 0, "deletions": 0, "scored": 0, "correct": 0}
    )
    for _, name in reference:
        per_speaker[name]["words"] += 1
    for operation, r, h in steps:
        if r is None:
            continue
        name = reference[r][1]
        if operation == "substitute":
            per_speaker[name]["substitutions"] += 1
        elif operation == "delete":
            per_speaker[name]["deletions"] += 1
        if operation in ("equal", "substitute"):
            assert h is not None
            per_speaker[name]["scored"] += 1
            per_speaker[name]["correct"] += mapping.get(hypothesis[h][1]) == name

    scored = sum(item["scored"] for item in per_speaker.values())
    correct = sum(item["correct"] for item in per_speaker.values())
    errors = counts["substitute"] + counts["delete"] + counts["insert"]
    return {
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "wer": errors / len(ref_words),
        "substitutions": counts["substitute"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
        "speaker_accuracy": correct / scored if scored else None,
        "speaker_scored_words": scored,
        "speaker_errors": scored - correct,
        "speaker_mapping": mapping,
        "unmapped_clusters": sorted(
            {cluster for cluster, _ in agreement} - set(mapping)
        ),
        "per_speaker": {
            name: {
                "words": item["words"],
                # Insertions belong to no reference word, so they are not split
                # by speaker; this is (S + D) / N for the speaker's own words.
                "word_error_rate": (item["substitutions"] + item["deletions"]) / item["words"],
                "speaker_accuracy": (
                    item["correct"] / item["scored"] if item["scored"] else None
                ),
            }
            for name, item in sorted(per_speaker.items())
        },
        "alignment": method,
        "differences": differences(steps, reference, hypothesis),
    }


def evaluate(
    merged: Mapping[str, Any],
    reference_text: str,
    *,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, Any]:
    reference = parse_reference(reference_text)
    window_start = start if start is not None else reference.start
    window_end = end if end is not None else reference.end
    hypothesis = hypothesis_words(merged, window_start, window_end)
    result = score(reference.words, hypothesis)
    result["window"] = (
        None if window_start is None and window_end is None else [window_start, window_end]
    )
    result["note"] = BIASED_REFERENCE_NOTE
    return result


def summary_lines(result: Mapping[str, Any]) -> list[str]:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    window = result.get("window")
    scope = (
        "whole job"
        if not window
        else f"{_seconds_text(window[0])}–{_seconds_text(window[1])}"
    )
    lines = [
        f"Scored {result['reference_words']} reference words ({scope}, "
        f"{result['alignment']} alignment).",
        f"WER {percent(result['wer'])}: {result['substitutions']} substituted, "
        f"{result['deletions']} missed, {result['insertions']} inserted.",
        f"Speaker accuracy {percent(result['speaker_accuracy'])} over "
        f"{result['speaker_scored_words']} aligned words "
        f"({result['speaker_errors']} attributed to the wrong person).",
    ]
    mapping = result.get("speaker_mapping") or {}
    if mapping:
        lines.append(
            "Mapping: " + ", ".join(f"{cluster} -> {name}" for cluster, name in mapping.items())
        )
    if result.get("unmapped_clusters"):
        lines.append("Not mapped to anyone: " + ", ".join(result["unmapped_clusters"]))
    for name, item in result.get("per_speaker", {}).items():
        lines.append(
            f"  {name}: {item['words']} words, word errors {percent(item['word_error_rate'])}, "
            f"speaker accuracy {percent(item['speaker_accuracy'])}"
        )
    found = result.get("differences") or []
    if found:
        lines.append(
            f"{len(found)} place(s) where the words differ; the first ones "
            "(all of them are in evaluation.json):"
        )
        for item in found[:SHOWN_DIFFERENCES]:
            lines.append(
                f"  [{_seconds_text(item['at'])}] {item['reference'] or '—'} "
                f"=> {item['hypothesis'] or '—'}"
            )
    lines.append(str(result.get("note", BIASED_REFERENCE_NOTE)))
    return lines


def _seconds_text(value: float | None) -> str:
    if value is None:
        return "…"
    total = int(value)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
