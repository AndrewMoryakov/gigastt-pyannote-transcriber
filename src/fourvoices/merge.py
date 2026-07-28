"""Time-align word timestamps with exclusive speaker diarization."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

# Timestamps are seconds with millisecond-scale meaning; treat anything closer
# than this as equal so float noise cannot invent or hide a tie.
EPSILON = 1e-9

UNKNOWN_SPEAKER = "UNKNOWN"

# A turn is flagged when at least this share of its words lack a confirmed
# speaker. One hedged word in a long sentence is normal; half of them is not.
UNCERTAIN_TURN_SHARE = 0.5


def word_is_uncertain(word: Mapping[str, Any]) -> bool:
    """Whether this word's speaker rests on something weaker than overlap.

    ``nearest`` and ``unknown`` mean pyannote had no exclusive segment covering
    the word, and ``ambiguous`` means two speakers tied. All three are cases a
    human should re-listen to before quoting. A word carrying no assignment
    evidence at all is not treated as doubtful.
    """

    if word.get("ambiguous"):
        return True
    assignment = word.get("assignment")
    return assignment is not None and str(assignment) != "overlap"


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _validate_segments(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for item in items:
        start, end = float(item["start"]), float(item["end"])
        if end > start:
            segments.append(
                {"start": start, "end": end, "speaker": str(item["speaker"])}
            )
    return sorted(segments, key=lambda x: (x["start"], x["end"], x["speaker"]))


class ExclusiveIndex:
    """Exclusive segments prepared for repeated interval queries.

    Scanning every segment per word made merging quadratic in recording length.
    Segments are sorted by start, so a binary search bounds the candidates; the
    running maximum of the ends tells the backward scan when to stop, which is
    what keeps the bound valid even if two segments happen to overlap.
    """

    __slots__ = ("segments", "_starts", "_max_end")

    def __init__(self, segments: Sequence[Mapping[str, Any]]) -> None:
        self.segments = _validate_segments(segments)
        self._starts = [segment["start"] for segment in self.segments]
        self._max_end: list[float] = []
        running = float("-inf")
        for segment in self.segments:
            running = max(running, segment["end"])
            self._max_end.append(running)

    def __len__(self) -> int:
        return len(self.segments)

    def overlapping(self, start: float, end: float) -> list[dict[str, Any]]:
        """Segments sharing a positive-length interval with ``[start, end]``."""

        found: list[dict[str, Any]] = []
        index = bisect_left(self._starts, end) - 1
        while index >= 0 and self._max_end[index] > start:
            segment = self.segments[index]
            if segment["end"] > start and segment["start"] < end:
                found.append(segment)
            index -= 1
        return found

    def nearest_candidates(self, start: float, end: float) -> list[dict[str, Any]]:
        """Segments that could be closest to a word lying in a diarization gap.

        Only called when nothing overlaps, so every segment sits wholly before or
        wholly after the word: the best on each side is the one ending latest
        before it or starting earliest after it, plus anything tied with it.
        """

        if not self.segments:
            return []
        candidates: list[dict[str, Any]] = []
        split = bisect_left(self._starts, end)
        if split > 0:
            best_end = self._max_end[split - 1]
            index = split - 1
            while index >= 0 and self._max_end[index] >= best_end - EPSILON:
                if self.segments[index]["end"] >= best_end - EPSILON:
                    candidates.append(self.segments[index])
                index -= 1
        if split < len(self.segments):
            best_start = self._starts[split]
            index = split
            while (
                index < len(self.segments)
                and self._starts[index] <= best_start + EPSILON
            ):
                candidates.append(self.segments[index])
                index += 1
        return candidates


def _as_exclusive_index(value: Sequence[Mapping[str, Any]] | ExclusiveIndex) -> ExclusiveIndex:
    return value if isinstance(value, ExclusiveIndex) else ExclusiveIndex(value)


def assign_speaker(
    start: float,
    end: float,
    exclusive_segments: Sequence[Mapping[str, Any]] | ExclusiveIndex,
    *,
    nearest_max_gap: float = 0.5,
) -> tuple[str, float]:
    """Choose the speaker with the greatest total exclusive overlap.

    A zero-duration token is treated as a tiny interval. When pyannote has a
    genuine gap, the nearest exclusive segment is used and overlap remains zero.
    """

    if end < start:
        raise ValueError("word end precedes start")
    detail = speaker_assignment(
        start, end, exclusive_segments, nearest_max_gap=nearest_max_gap
    )
    return str(detail["speaker"]), float(detail["overlap_s"])


def speaker_assignment(
    start: float,
    end: float,
    exclusive_segments: Sequence[Mapping[str, Any]] | ExclusiveIndex,
    *,
    nearest_max_gap: float = 0.5,
) -> dict[str, Any]:
    """Return label, evidence for the assignment, and tie ambiguity."""

    if end < start:
        raise ValueError("word end precedes start")
    if nearest_max_gap < 0:
        raise ValueError("nearest_max_gap must not be negative")
    index = _as_exclusive_index(exclusive_segments)
    effective_end = end if end > start else start + 0.001
    totals: dict[str, float] = defaultdict(float)
    for segment in index.overlapping(start, effective_end):
        overlap = interval_overlap(
            start, effective_end, segment["start"], segment["end"]
        )
        if overlap:
            totals[str(segment["speaker"])] += overlap
    if totals:
        ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        speaker, overlap = ranked[0]
        ambiguous = len(ranked) > 1 and abs(ranked[1][1] - overlap) <= EPSILON
        return {
            "speaker": speaker,
            "overlap_s": overlap,
            "assignment": "overlap",
            "ambiguous": ambiguous,
        }

    def gap(segment: Mapping[str, Any]) -> float:
        return max(
            float(segment["start"]) - effective_end,
            start - float(segment["end"]),
            0.0,
        )

    candidates = index.nearest_candidates(start, effective_end)
    if candidates:
        nearest = min(
            candidates, key=lambda segment: (gap(segment), str(segment["speaker"]))
        )
        nearest_gap = gap(nearest)
        if nearest_gap <= nearest_max_gap:
            tied = {
                str(segment["speaker"])
                for segment in candidates
                if abs(gap(segment) - nearest_gap) <= EPSILON
            }
            return {
                "speaker": str(nearest["speaker"]),
                "overlap_s": 0.0,
                "assignment": "nearest",
                "ambiguous": len(tied) > 1,
            }
    return {
        "speaker": UNKNOWN_SPEAKER,
        "overlap_s": 0.0,
        "assignment": "unknown",
        "ambiguous": False,
    }


class OverlapIndex:
    """Times where two or more different regular tracks speak at once.

    The answer does not depend on the query, so it is computed once with a
    boundary sweep instead of re-scanning every segment pair per word.
    """

    __slots__ = ("intervals", "_starts")

    def __init__(self, regular_segments: Sequence[Mapping[str, Any]]) -> None:
        self.intervals = _simultaneous_intervals(regular_segments)
        self._starts = [interval[0] for interval in self.intervals]

    def covers(self, start: float, end: float) -> bool:
        """Whether ``[start, end]`` shares positive length with any such time."""

        position = bisect_left(self._starts, end)
        for index in range(position - 1, -1, -1):
            interval_start, interval_end = self.intervals[index]
            if interval_end <= start:
                # Intervals are disjoint and sorted, so everything earlier ends
                # even sooner.
                break
            if min(end, interval_end) - max(start, interval_start) > 0:
                return True
        return False


def _simultaneous_intervals(
    regular_segments: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    events: list[tuple[float, int, str]] = []
    for segment in regular_segments:
        start, end = float(segment["start"]), float(segment["end"])
        if end > start:
            speaker = str(segment["speaker"])
            events.append((start, 1, speaker))
            events.append((end, -1, speaker))
    if not events:
        return []
    # Close before opening at the same instant: touching segments are adjacent,
    # not simultaneous.
    events.sort(key=lambda event: (event[0], event[1]))
    active: dict[str, int] = defaultdict(int)
    distinct = 0
    intervals: list[tuple[float, float]] = []
    position = 0
    total = len(events)
    while position < total:
        time = events[position][0]
        while position < total and events[position][0] == time:
            _, delta, speaker = events[position]
            active[speaker] += delta
            if delta > 0 and active[speaker] == 1:
                distinct += 1
            elif delta < 0 and active[speaker] == 0:
                distinct -= 1
            position += 1
        # The active set stays constant until the next boundary.
        if distinct >= 2 and position < total and events[position][0] > time:
            intervals.append((time, events[position][0]))
    merged: list[tuple[float, float]] = []
    for interval in intervals:
        if merged and interval[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        else:
            merged.append(interval)
    return merged


def has_regular_overlap(
    start: float,
    end: float,
    regular_segments: Sequence[Mapping[str, Any]] | OverlapIndex,
) -> bool:
    """Whether two different regular diarization tracks overlap in this interval."""

    index = (
        regular_segments
        if isinstance(regular_segments, OverlapIndex)
        else OverlapIndex(regular_segments)
    )
    return index.covers(start, end)


_NO_SPACE_BEFORE = frozenset(",.!?:;%)]}»")
_NO_SPACE_AFTER = frozenset("([{«")


def _normalized_token(value: str) -> str:
    return re.sub(r"(^[^\w]+|[^\w]+$)", "", value.lower().replace("ё", "е"))


def align_processed_text(
    words: Sequence[Mapping[str, Any]], processed_text: str
) -> tuple[list[str], float]:
    """Project GigaSTT punctuation/casing onto timestamped raw words.

    GigaSTT deliberately keeps ``words[].word`` raw while applying punctuation
    and ITN only to top-level ``text``. Exact normalized matches are therefore
    safe to reuse for display. Any ITN/rewrite mismatch falls back to the raw
    timestamped word instead of guessing an alignment.
    """

    raw = [str(word.get("word", word.get("text", ""))).strip() for word in words]
    processed = [token for token in str(processed_text).split() if token]
    raw_keys = [_normalized_token(token) for token in raw]
    processed_keys = [_normalized_token(token) for token in processed]
    display = list(raw)
    matched = 0
    matcher = SequenceMatcher(a=raw_keys, b=processed_keys, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            raw_index = block.a + offset
            processed_index = block.b + offset
            if raw_keys[raw_index]:
                display[raw_index] = processed[processed_index]
                matched += 1
    ratio = matched / len(raw) if raw else 1.0
    return display, ratio


def join_words(words: Iterable[str]) -> str:
    """Join ASR tokens without inserting spaces before standalone punctuation."""

    text = ""
    for raw in words:
        token = str(raw).strip()
        if not token:
            continue
        if not text or token[0] in _NO_SPACE_BEFORE or text[-1] in _NO_SPACE_AFTER:
            text += token
        else:
            text += " " + token
    return text


def _confidence(words: Sequence[Mapping[str, Any]]) -> float | None:
    if not words:
        return None
    weights = [max(0.0, float(w["end"]) - float(w["start"])) for w in words]
    if sum(weights) > 0:
        weighted = sum(
            float(word.get("confidence", 0.0)) * weight
            for word, weight in zip(words, weights, strict=True)
        )
        return weighted / sum(weights)
    return sum(float(w.get("confidence", 0.0)) for w in words) / len(words)


def build_turns(
    words: Sequence[Mapping[str, Any]], *, max_gap: float = 1.5
) -> list[dict[str, Any]]:
    if max_gap < 0:
        raise ValueError("max_gap must not be negative")
    groups: list[list[Mapping[str, Any]]] = []
    for word in words:
        if (
            not groups
            or word["speaker"] != groups[-1][-1]["speaker"]
            or float(word["start"]) - float(groups[-1][-1]["end"]) > max_gap
        ):
            groups.append([word])
        else:
            groups[-1].append(word)
    turns: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        uncertain_words = sum(1 for word in group if word_is_uncertain(word))
        turns.append(
            {
                "id": index,
                "speaker": str(group[0]["speaker"]),
                "start": float(group[0]["start"]),
                "end": max(float(w["end"]) for w in group),
                "text": join_words(str(w.get("display_word", w["word"])) for w in group),
                "raw_text": join_words(str(w["word"]) for w in group),
                "overlap": any(bool(w.get("overlap")) for w in group),
                "uncertain_words": uncertain_words,
                "uncertain": uncertain_words >= len(group) * UNCERTAIN_TURN_SHARE,
                "confidence": _confidence(group),
                "word_start": int(group[0].get("index", 0)),
                "word_end": int(group[-1].get("index", 0)),
            }
        )
    return turns


def merge_transcript(
    transcript: Mapping[str, Any],
    diarization: Mapping[str, Any],
    *,
    max_turn_gap: float = 1.5,
    nearest_max_gap: float = 0.5,
) -> dict[str, Any]:
    """Return an evidence-friendly merged representation."""

    exclusive = ExclusiveIndex(diarization.get("exclusive_segments", []))
    regular = OverlapIndex(_validate_segments(diarization.get("segments", [])))
    source_words = list(transcript.get("words", []))
    display_words, punctuation_alignment_ratio = align_processed_text(
        source_words, str(transcript.get("text", ""))
    )
    merged_words: list[dict[str, Any]] = []
    for index, source_word in enumerate(source_words):
        start, end = float(source_word["start"]), float(source_word["end"])
        assignment = speaker_assignment(
            start, end, exclusive, nearest_max_gap=nearest_max_gap
        )
        merged_words.append(
            {
                "index": index,
                "word": str(source_word.get("word", source_word.get("text", ""))).strip(),
                "display_word": display_words[index],
                "start": start,
                "end": end,
                "confidence": float(source_word.get("confidence", 0.0)),
                "speaker": assignment["speaker"],
                "speaker_overlap_s": assignment["overlap_s"],
                "assignment": assignment["assignment"],
                "ambiguous": assignment["ambiguous"],
                "overlap": has_regular_overlap(start, end, regular),
            }
        )
    turns = build_turns(merged_words, max_gap=max_turn_gap)
    for turn in turns:
        # The regular annotation is the authority for simultaneous speech;
        # checking the full turn also catches overlap between adjacent ASR words.
        turn["overlap"] = has_regular_overlap(
            float(turn["start"]), float(turn["end"]), regular
        )
    speakers = sorted({word["speaker"] for word in merged_words})
    duration = max(
        float(transcript.get("duration", transcript.get("duration_s", 0.0))),
        max((word["end"] for word in merged_words), default=0.0),
    )
    return {
        "schema_version": 1,
        "duration_s": duration,
        "speakers": speakers,
        "text": join_words(word["display_word"] for word in merged_words),
        "raw_text": join_words(word["word"] for word in merged_words),
        "gigastt_processed_text": str(transcript.get("text", "")),
        "punctuation_alignment_ratio": punctuation_alignment_ratio,
        "words": merged_words,
        "turns": turns,
    }
