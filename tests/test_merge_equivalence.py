"""The indexed merge must answer exactly like the original scan-everything code.

The reference implementations below are the pre-optimisation versions, kept
verbatim. Random cases are compared against them so a speed change cannot
quietly become a behaviour change.
"""

import random

import pytest

from fourvoices.merge import (
    UNKNOWN_SPEAKER,
    has_regular_overlap,
    interval_overlap,
    speaker_assignment,
)


def reference_speaker_assignment(start, end, exclusive_segments, *, nearest_max_gap=0.5):
    effective_end = end if end > start else start + 0.001
    totals = {}
    for segment in exclusive_segments:
        overlap = interval_overlap(
            start, effective_end, float(segment["start"]), float(segment["end"])
        )
        if overlap:
            totals[str(segment["speaker"])] = (
                totals.get(str(segment["speaker"]), 0.0) + overlap
            )
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

    def gap(segment):
        return max(
            float(segment["start"]) - effective_end, start - float(segment["end"]), 0.0
        )

    nearest = min(
        exclusive_segments, key=lambda segment: (gap(segment), str(segment["speaker"]))
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


def reference_has_regular_overlap(start, end, regular_segments):
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


def random_exclusive(rng, count):
    """Non-overlapping segments with gaps, as pyannote's exclusive output is."""

    segments = []
    cursor = 0.0
    for _ in range(count):
        cursor += rng.choice([0.0, 0.0, 0.3, 1.0])
        length = rng.choice([0.2, 0.7, 2.0])
        segments.append(
            {
                "start": round(cursor, 3),
                "end": round(cursor + length, 3),
                "speaker": f"SPEAKER_0{rng.randrange(4)}",
            }
        )
        cursor += length
    return sorted(segments, key=lambda s: (s["start"], s["end"], s["speaker"]))


def random_regular(rng, count):
    """Overlapping segments, as the regular annotation can be."""

    segments = []
    for _ in range(count):
        start = round(rng.uniform(0.0, 30.0), 3)
        segments.append(
            {
                "start": start,
                "end": round(start + rng.uniform(0.05, 4.0), 3),
                "speaker": f"SPEAKER_0{rng.randrange(4)}",
            }
        )
    return sorted(segments, key=lambda s: (s["start"], s["end"], s["speaker"]))


@pytest.mark.parametrize("seed", range(40))
def test_speaker_assignment_matches_the_reference(seed):
    rng = random.Random(seed)
    segments = random_exclusive(rng, rng.randrange(0, 25))
    for _ in range(60):
        start = round(rng.uniform(-1.0, 35.0), 3)
        end = start + rng.choice([0.0, 0.05, 0.4, 1.5])
        for gap_limit in (0.0, 0.5, 3.0):
            assert speaker_assignment(
                start, end, segments, nearest_max_gap=gap_limit
            ) == reference_speaker_assignment(
                start, end, segments, nearest_max_gap=gap_limit
            )


@pytest.mark.parametrize("seed", range(40))
def test_regular_overlap_matches_the_reference(seed):
    rng = random.Random(seed)
    segments = random_regular(rng, rng.randrange(0, 25))
    for _ in range(60):
        start = round(rng.uniform(-1.0, 35.0), 3)
        end = start + rng.choice([0.0, 0.05, 0.4, 1.5, 6.0])
        assert has_regular_overlap(start, end, segments) == reference_has_regular_overlap(
            start, end, segments
        )


def test_identical_segments_from_one_speaker_are_not_overlap():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 0.0, "end": 1.0, "speaker": "A"},
    ]
    assert has_regular_overlap(0.0, 1.0, segments) is False


def test_touching_segments_are_adjacent_not_simultaneous():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]
    assert has_regular_overlap(0.0, 2.0, segments) is False


def test_three_way_overlap_is_detected():
    segments = [
        {"start": 0.0, "end": 3.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
        {"start": 1.5, "end": 1.7, "speaker": "C"},
    ]
    assert has_regular_overlap(1.4, 1.6, segments) is True
    assert has_regular_overlap(2.5, 2.9, segments) is False


@pytest.mark.parametrize("seed", range(40))
def test_speaker_assignment_matches_the_reference_on_overlapping_segments(seed):
    # The exclusive annotation should not overlap itself, but nothing in the
    # code depends on that, so the index must stay correct if it ever does.
    rng = random.Random(1000 + seed)
    segments = random_regular(rng, rng.randrange(0, 20))
    for _ in range(60):
        start = round(rng.uniform(-1.0, 35.0), 3)
        end = start + rng.choice([0.0, 0.05, 0.4, 1.5])
        assert speaker_assignment(start, end, segments) == reference_speaker_assignment(
            start, end, segments
        )
