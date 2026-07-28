"""Time-align word timestamps with exclusive speaker diarization."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import re
from typing import Any, Iterable, Mapping, Sequence

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


def assign_speaker(
    start: float,
    end: float,
    exclusive_segments: Sequence[Mapping[str, Any]],
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
    exclusive_segments: Sequence[Mapping[str, Any]],
    *,
    nearest_max_gap: float = 0.5,
) -> dict[str, Any]:
    """Return label, evidence for the assignment, and tie ambiguity."""

    if end < start:
        raise ValueError("word end precedes start")
    if nearest_max_gap < 0:
        raise ValueError("nearest_max_gap must not be negative")
    effective_end = end if end > start else start + 0.001
    totals: dict[str, float] = defaultdict(float)
    for segment in exclusive_segments:
        overlap = interval_overlap(
            start, effective_end, float(segment["start"]), float(segment["end"])
        )
        if overlap:
            totals[str(segment["speaker"])] += overlap
    if totals:
        ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        speaker, overlap = ranked[0]
        ambiguous = len(ranked) > 1 and abs(ranked[1][1] - overlap) <= 1e-9
        return {
            "speaker": speaker,
            "overlap_s": overlap,
            "assignment": "overlap",
            "ambiguous": ambiguous,
        }
    if not exclusive_segments:
        return {
            "speaker": UNKNOWN_SPEAKER,
            "overlap_s": 0.0,
            "assignment": "unknown",
            "ambiguous": False,
        }

    def gap(segment: Mapping[str, Any]) -> float:
        return max(
            float(segment["start"]) - effective_end,
            start - float(segment["end"]),
            0.0,
        )

    nearest = min(
        exclusive_segments,
        key=lambda segment: (
            gap(segment),
            str(segment["speaker"]),
        ),
    )
    nearest_gap = gap(nearest)
    if nearest_gap <= nearest_max_gap:
        tied = {
            str(segment["speaker"])
            for segment in exclusive_segments
            if abs(gap(segment) - nearest_gap) <= 1e-9
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


def has_regular_overlap(
    start: float, end: float, regular_segments: Sequence[Mapping[str, Any]]
) -> bool:
    """Whether two different regular diarization tracks overlap in this interval."""

    relevant = [
        segment
        for segment in regular_segments
        if interval_overlap(start, end, float(segment["start"]), float(segment["end"])) > 0
    ]
    for index, left in enumerate(relevant):
        for right in relevant[index + 1 :]:
            if str(left["speaker"]) == str(right["speaker"]):
                continue
            shared_start = max(start, float(left["start"]), float(right["start"]))
            shared_end = min(end, float(left["end"]), float(right["end"]))
            if shared_end > shared_start:
                return True
    return False


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
        return sum(float(w.get("confidence", 0.0)) * n for w, n in zip(words, weights)) / sum(
            weights
        )
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

    exclusive = _validate_segments(diarization.get("exclusive_segments", []))
    regular = _validate_segments(diarization.get("segments", []))
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
