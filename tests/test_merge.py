from fourvoices.merge import (
    align_processed_text,
    assign_speaker,
    build_turns,
    has_regular_overlap,
    join_words,
    merge_transcript,
    speaker_assignment,
)


def test_processed_text_projects_only_exact_word_matches():
    words = [
        {"word": "привет"},
        {"word": "двадцать"},
        {"word": "три"},
        {"word": "мир"},
    ]
    display, ratio = align_processed_text(words, "Привет, 23 мир.")

    assert display == ["Привет,", "двадцать", "три", "мир."]
    assert ratio == 0.5


def test_maximum_total_exclusive_overlap_wins():
    segments = [
        {"start": 0.0, "end": 0.4, "speaker": "A"},
        {"start": 0.4, "end": 1.0, "speaker": "B"},
        {"start": 0.8, "end": 1.0, "speaker": "A"},
    ]
    assert assign_speaker(0.0, 1.0, segments) == ("A", 0.6)


def test_nearest_speaker_fills_a_real_diarization_gap():
    label, overlap = assign_speaker(
        5.0, 5.2, [{"start": 4.0, "end": 4.9, "speaker": "SPEAKER_00"}]
    )
    assert label == "SPEAKER_00"
    assert overlap == 0


def test_distant_speaker_does_not_fill_a_long_gap():
    label, overlap = assign_speaker(
        10.0, 10.2, [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    )
    assert label == "UNKNOWN"
    assert overlap == 0


def test_equal_overlap_is_marked_ambiguous():
    detail = speaker_assignment(
        0.0,
        1.0,
        [
            {"start": 0.0, "end": 0.5, "speaker": "A"},
            {"start": 0.5, "end": 1.0, "speaker": "B"},
        ],
    )
    assert detail["speaker"] == "A"
    assert detail["assignment"] == "overlap"
    assert detail["ambiguous"] is True


def test_regular_overlap_requires_simultaneous_different_tracks():
    adjacent = [
        {"start": 0.0, "end": 0.5, "speaker": "A"},
        {"start": 0.5, "end": 1.0, "speaker": "B"},
    ]
    simultaneous = [
        {"start": 0.0, "end": 0.8, "speaker": "A"},
        {"start": 0.6, "end": 1.0, "speaker": "B"},
    ]
    assert not has_regular_overlap(0.0, 1.0, adjacent)
    assert has_regular_overlap(0.0, 1.0, simultaneous)


def test_turns_split_on_speaker_and_long_gap_and_join_punctuation():
    words = [
        {"index": 0, "word": "Да", "start": 0, "end": 0.2, "speaker": "A"},
        {"index": 1, "word": ",", "start": 0.2, "end": 0.3, "speaker": "A"},
        {"index": 2, "word": "верно", "start": 0.3, "end": 0.6, "speaker": "A"},
        {"index": 3, "word": "Нет", "start": 0.7, "end": 1.0, "speaker": "B"},
        {"index": 4, "word": "Позже", "start": 3.0, "end": 3.2, "speaker": "B"},
    ]
    turns = build_turns(words, max_gap=1.0)
    assert [(turn["speaker"], turn["text"]) for turn in turns] == [
        ("A", "Да, верно"),
        ("B", "Нет"),
        ("B", "Позже"),
    ]
    assert join_words(["«", "тест", "»", "."]) == "«тест»."


def test_merge_assigns_words_flags_overlap_and_builds_turns():
    transcript = {
        "duration": 2.0,
        "duration_s": 2.0,
        "words": [
            {"word": "первый", "start": 0.1, "end": 0.5, "confidence": 0.9},
            {"word": "второй", "start": 1.1, "end": 1.6, "confidence": 0.8},
        ],
    }
    diarization = {
        "exclusive_segments": [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ],
        "segments": [
            {"start": 0.0, "end": 1.4, "speaker": "A"},
            {"start": 1.2, "end": 2.0, "speaker": "B"},
        ],
    }
    result = merge_transcript(transcript, diarization)
    assert [word["speaker"] for word in result["words"]] == ["A", "B"]
    assert result["words"][0]["overlap"] is False
    assert result["words"][1]["overlap"] is True
    assert len(result["turns"]) == 2
    assert result["duration_s"] == 2.0


def test_turn_is_flagged_only_when_half_its_words_lack_confirmed_speakers():
    def word(index, assignment, ambiguous=False):
        return {
            "index": index,
            "word": f"w{index}",
            "start": index * 0.5,
            "end": index * 0.5 + 0.4,
            "speaker": "A",
            "assignment": assignment,
            "ambiguous": ambiguous,
        }

    mostly_confirmed = build_turns([word(0, "overlap"), word(1, "overlap"), word(2, "nearest")])
    assert mostly_confirmed[0]["uncertain_words"] == 1
    assert mostly_confirmed[0]["uncertain"] is False

    half_unconfirmed = build_turns([word(0, "overlap"), word(1, "unknown")])
    assert half_unconfirmed[0]["uncertain"] is True

    tied = build_turns([word(0, "overlap", ambiguous=True)])
    assert tied[0]["uncertain"] is True


def test_merge_marks_a_turn_resting_on_the_nearest_segment():
    transcript = {"duration": 6.0, "words": [{"word": "эхо", "start": 5.0, "end": 5.2}]}
    diarization = {
        "exclusive_segments": [{"start": 4.0, "end": 4.9, "speaker": "A"}],
        "segments": [],
    }
    result = merge_transcript(transcript, diarization)
    assert result["words"][0]["assignment"] == "nearest"
    assert result["turns"][0]["uncertain"] is True
