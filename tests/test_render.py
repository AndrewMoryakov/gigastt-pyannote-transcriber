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
