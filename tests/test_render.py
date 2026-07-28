import json

from fourvoices.render import clock, render_json, render_md, render_srt, render_txt, render_vtt


DATA = {
    "duration_s": 62.5,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "words": [],
    "turns": [
        {
            "id": 1,
            "speaker": "SPEAKER_00",
            "start": 1.234,
            "end": 3.5,
            "text": "Привет.",
            "overlap": False,
            "confidence": 0.9,
        },
        {
            "id": 2,
            "speaker": "SPEAKER_01",
            "start": 61.0,
            "end": 62.5,
            "text": "Да.",
            "overlap": True,
            "confidence": 0.8,
        },
    ],
}


def test_clock_rounds_and_carries_milliseconds():
    assert clock(3599.9996, milliseconds=True) == "01:00:00.000"
    assert clock(-1) == "00:00:00"


def test_text_and_markdown_use_names_and_overlap_marker():
    names = {"SPEAKER_00": "Отец", "SPEAKER_01": "Мать"}
    txt = render_txt(DATA, names=names)
    md = render_md(DATA, names=names)
    assert "Отец: Привет." in txt
    assert "Мать [перекрытие речи]: Да." in txt
    assert "# Транскрипция аудиозаписи" in md
    assert "Мать · **перекрытие речи**" in md


def test_srt_and_vtt_timestamp_syntax():
    srt = render_srt(DATA)
    vtt = render_vtt(DATA)
    assert "1\n00:00:01,234 --> 00:00:03,500" in srt
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:01:01.000 --> 00:01:02.500" in vtt


def test_json_is_utf8_friendly_and_round_trips():
    rendered = render_json(DATA)
    assert "\\u041f" not in rendered
    assert json.loads(rendered) == DATA


UNCERTAIN_DATA = {
    "turns": [
        {
            "id": 1,
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.0,
            "text": "Возможно.",
            "overlap": False,
            "uncertain": True,
            "uncertain_words": 1,
        },
        {
            "id": 2,
            "speaker": "SPEAKER_01",
            "start": 1.0,
            "end": 2.0,
            "text": "Точно.",
            "overlap": True,
            "uncertain": True,
            "uncertain_words": 1,
        },
    ]
}


def test_uncertain_turns_are_marked_in_every_text_format():
    assert "[спикер под вопросом]" in render_txt(UNCERTAIN_DATA)
    assert "**спикер под вопросом**" in render_md(UNCERTAIN_DATA)
    assert "[спикер под вопросом]" in render_srt(UNCERTAIN_DATA)
    assert "[спикер под вопросом]" in render_vtt(UNCERTAIN_DATA)


def test_both_markers_appear_together_and_in_a_stable_order():
    line = render_txt(UNCERTAIN_DATA).splitlines()[1]
    assert "[перекрытие речи, спикер под вопросом]" in line


def test_marking_can_be_disabled_without_touching_the_overlap_marker():
    text = render_txt(UNCERTAIN_DATA, mark_uncertain=False)
    assert "спикер под вопросом" not in text
    assert "[перекрытие речи]" in text


def test_turns_without_the_field_are_never_marked():
    assert "спикер под вопросом" not in render_txt(DATA)


LONG_TURN = {
    "duration_s": 30.0,
    "turns": [
        {
            "id": 1,
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 20.0,
            "text": " ".join(f"слово{i}" for i in range(20)),
            "overlap": False,
            "word_start": 0,
            "word_end": 19,
        }
    ],
    "words": [
        {
            "index": i,
            "word": f"слово{i}",
            "display_word": f"слово{i}",
            "start": i * 1.0,
            "end": i * 1.0 + 0.9,
            "speaker": "SPEAKER_00",
        }
        for i in range(20)
    ],
}


def _cue_times(srt_text):
    return [line for line in srt_text.splitlines() if "-->" in line]


def test_a_long_turn_becomes_several_cues():
    cues = _cue_times(render_srt(LONG_TURN))
    assert len(cues) > 1
    assert cues[0].startswith("00:00:00,000")


def test_no_cue_exceeds_the_time_limit():
    from fourvoices.render import split_turn

    turn = LONG_TURN["turns"][0]
    for cue in split_turn(turn, LONG_TURN["words"], max_seconds=6.0, max_chars=0):
        assert cue["end"] - cue["start"] <= 6.0


def test_cues_stay_inside_the_turn_and_never_go_backwards():
    from fourvoices.render import split_turn

    turn = LONG_TURN["turns"][0]
    cues = split_turn(turn, LONG_TURN["words"])
    assert cues[0]["start"] >= turn["start"]
    assert cues[-1]["end"] <= turn["end"]
    for earlier, later in zip(cues, cues[1:]):
        assert earlier["end"] <= later["start"]


def test_split_keeps_every_word():
    from fourvoices.render import split_turn

    cues = split_turn(LONG_TURN["turns"][0], LONG_TURN["words"])
    assert " ".join(cue["text"] for cue in cues) == LONG_TURN["turns"][0]["text"]


def test_srt_numbering_is_continuous_across_split_turns():
    numbers = [
        line for line in render_srt(LONG_TURN).splitlines() if line.strip().isdigit()
    ]
    assert numbers == [str(i) for i in range(1, len(numbers) + 1)]


def test_splitting_can_be_disabled():
    cues = _cue_times(render_srt(LONG_TURN, max_cue_seconds=0, max_cue_chars=0))
    assert len(cues) == 1


def test_a_turn_without_word_timings_is_left_whole():
    without_words = {"turns": LONG_TURN["turns"]}
    assert len(_cue_times(render_srt(without_words))) == 1


def test_speaker_label_repeats_in_every_cue_of_a_turn():
    body = render_srt(LONG_TURN, names={"SPEAKER_00": "Отец"})
    assert body.count("Отец:") == len(_cue_times(body))
