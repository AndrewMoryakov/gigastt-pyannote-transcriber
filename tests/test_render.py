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
