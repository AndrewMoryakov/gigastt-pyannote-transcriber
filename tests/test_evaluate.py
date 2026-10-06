import json

import pytest

from fourvoices import evaluate as ev
from fourvoices.evaluate import ReferenceError, evaluate, parse_reference


def merged(*turns):
    """A merged.json with one word per 0.5 s: turns are (speaker, "text")."""

    words, clock = [], 0.0
    for speaker, text in turns:
        for word in text.split():
            words.append(
                {"display_word": word, "word": word.lower(), "start": clock,
                 "end": clock + 0.4, "speaker": speaker}
            )
            clock += 0.5
    return {"words": words}


# ------------------------------------------------------------------- the metric


def test_wer_counts_each_kind_of_error_exactly():
    # "два" -> "девять" substituted, "четыре" missed, "восемь" inserted: any other
    # alignment of these two texts costs more than three edits.
    reference = "Отец: раз два три четыре пять шесть семь\n"
    job = merged(("SPEAKER_00", "раз девять три пять шесть семь восемь"))

    result = evaluate(job, reference)

    assert (result["substitutions"], result["deletions"], result["insertions"]) == (1, 1, 1)
    assert result["wer"] == pytest.approx(3 / 7)
    assert result["alignment"] == "exact"


def test_normalisation_ignores_case_punctuation_and_yo():
    result = evaluate(merged(("A", "Ещё, раз!")), "Мать: еще раз\n")
    assert result["wer"] == 0


def test_cluster_labels_are_mapped_to_names_whatever_their_order():
    reference = "Отец: да нет\n[00:00:00–00:00:10] Мать: может быть\nОтец: конечно\n"
    job = merged(
        ("SPEAKER_01", "да нет"), ("SPEAKER_00", "может быть"), ("SPEAKER_01", "конечно")
    )

    result = evaluate(job, reference)

    assert result["speaker_mapping"] == {"SPEAKER_00": "Мать", "SPEAKER_01": "Отец"}
    assert result["speaker_accuracy"] == 1.0
    assert result["speaker_errors"] == 0


def test_a_wrongly_attributed_word_is_counted():
    reference = "Отец: да нет\n[00:00:00–00:00:10] Мать: может быть\n"
    job = merged(("SPEAKER_01", "да нет может"), ("SPEAKER_00", "быть"))

    result = evaluate(job, reference)

    assert result["wer"] == 0  # the words are right; only the speaker is not
    assert result["speaker_errors"] == 1
    assert result["speaker_accuracy"] == pytest.approx(3 / 4)


def test_unknown_words_are_speaker_errors():
    reference = "Отец: раз два три\n"
    job = merged(("SPEAKER_00", "раз два"), ("UNKNOWN", "три"))

    result = evaluate(job, reference)

    assert "UNKNOWN" not in result["speaker_mapping"]
    assert result["speaker_errors"] == 1


def test_a_surplus_cluster_stays_unmapped_and_counts_against_accuracy():
    reference = "Отец: раз два три четыре\n"
    job = merged(("SPEAKER_00", "раз два три"), ("SPEAKER_01", "четыре"))

    result = evaluate(job, reference)

    assert result["speaker_mapping"] == {"SPEAKER_00": "Отец"}
    assert result["unmapped_clusters"] == ["SPEAKER_01"]
    assert result["speaker_errors"] == 1


def test_only_words_inside_the_reference_span_are_scored():
    # Words every 0.5 s: "раз" 0.0, "два" 0.5, ..., "шесть" 2.5.
    job = merged(("SPEAKER_00", "раз два три четыре пять шесть"))
    reference = "[00:00:01–00:00:01] Отец: три четыре\n"

    result = evaluate(job, reference)

    # The span is 1.0..2.0 s once widened by the one-second timestamp resolution.
    assert result["window"] == [1.0, 2.0]
    assert result["hypothesis_words"] == 2
    assert result["wer"] == 0


def test_explicit_bounds_override_the_reference_span():
    job = merged(("SPEAKER_00", "раз два три четыре пять шесть"))
    result = evaluate(job, "Отец: пять шесть\n", start=2.0, end=3.0)
    assert result["wer"] == 0


@pytest.mark.parametrize(
    "bounds",
    [
        {"start": 3.0, "end": 2.0},  # inverted
        {"start": 2.0, "end": 2.0},  # empty
        {"start": 100.0},  # start past the reference's own end
        {"end": 0.5},  # end before the reference's own start
    ],
)
def test_an_inverted_or_empty_window_is_refused_not_scored_as_all_deleted(bounds):
    job = merged(("SPEAKER_00", "раз два три четыре пять шесть"))
    reference = "[00:00:01–00:00:02] Отец: два три\n"
    with pytest.raises(ReferenceError, match="window is empty"):
        evaluate(job, reference, **bounds)


def test_a_reference_without_timestamps_scores_the_whole_job():
    job = merged(("SPEAKER_00", "раз два три"))
    result = evaluate(job, "Отец: раз два три\n")
    assert result["window"] is None
    assert result["hypothesis_words"] == 3


def test_the_banded_alignment_agrees_with_the_full_table(monkeypatch):
    # Repetitive on purpose: anchoring on matching runs got exactly this wrong.
    words = ["раз", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    reference = "Отец: " + " ".join(words * 3) + "\n"
    spoken = list(words * 3)
    spoken[4] = "пятак"
    del spoken[12]
    spoken.insert(20, "лишнее")
    job = merged(("SPEAKER_00", " ".join(spoken)))

    full = evaluate(job, reference)
    # Force the band, and start it too narrow so that it has to double.
    monkeypatch.setattr(ev, "EXACT_ALIGNMENT_CELLS", 10)
    monkeypatch.setattr(ev, "INITIAL_BAND", 2)
    banded = evaluate(job, reference)

    assert full["alignment"] == banded["alignment"] == "exact"
    counts = ("substitutions", "deletions", "insertions")
    assert [banded[key] for key in counts] == [full[key] for key in counts] == [1, 1, 1]


def test_a_band_that_cannot_grow_far_enough_is_reported_as_an_upper_bound(monkeypatch):
    reference = ["слово"] * 300
    hypothesis = ["другое"] * 300
    monkeypatch.setattr(ev, "EXACT_ALIGNMENT_CELLS", 10)
    monkeypatch.setattr(ev, "INITIAL_BAND", 2)
    monkeypatch.setattr(ev, "MAX_ALIGNMENT_CELLS", 3000)

    steps, method = ev.align(reference, hypothesis)

    assert method == "banded (WER is an upper bound)"
    assert sum(step[0] != "equal" for step in steps) == 300


def test_texts_too_unequal_in_length_are_refused_before_any_table_is_built(monkeypatch):
    # The narrowest band that can reach the far corner is as wide as the length
    # difference, so 50 words against 300 needs ~15 000 cells: over the limit.
    # The limit has to hold before the first table, not only before doubling.
    limit = 3000
    monkeypatch.setattr(ev, "EXACT_ALIGNMENT_CELLS", 10)
    monkeypatch.setattr(ev, "INITIAL_BAND", 2)
    monkeypatch.setattr(ev, "MAX_ALIGNMENT_CELLS", limit)
    built = []
    real = ev._alignment

    def spy(ref, hyp, band=None):
        built.append((len(ref) + 1) * min(2 * band + 1, len(hyp) + 1))
        assert built[-1] <= limit, f"a {built[-1]}-cell table was built; the limit is {limit}"
        return real(ref, hyp, band)

    monkeypatch.setattr(ev, "_alignment", spy)

    with pytest.raises(ReferenceError, match="--start/--end"):
        ev.align(["слово"] * 50, ["другое"] * 300)
    assert built == []


def test_the_reported_case_of_10000_against_30000_words_is_refused_at_once():
    # With the real limits: this pair used to start on a ~250-million-cell table.
    reference = "Отец: " + " ".join(["слово"] * 10_000) + "\n"
    job = merged(("SPEAKER_00", " ".join(["другое"] * 30_000)))

    with pytest.raises(ReferenceError, match="10000 words .* has 30000"):
        evaluate(job, reference)


def test_a_length_difference_that_still_fits_the_limit_is_aligned(monkeypatch):
    # 20 words against 120: the band is 100 wide on each side but cannot be
    # wider than the 120-word text, so the table is 21 x 121 = 2541 cells.
    monkeypatch.setattr(ev, "EXACT_ALIGNMENT_CELLS", 10)
    monkeypatch.setattr(ev, "INITIAL_BAND", 2)
    monkeypatch.setattr(ev, "MAX_ALIGNMENT_CELLS", 3000)
    reference = [f"слово{i}" for i in range(20)]
    hypothesis = [f"слово{i // 6}" for i in range(120)]

    steps, method = ev.align(reference, hypothesis)

    assert method == "exact"
    assert sum(step[0] == "insert" for step in steps) >= 100


# ------------------------------------------------------------------ the parser


def test_the_shape_of_transcript_txt_is_accepted():
    text = (
        "[00:00:01–00:00:04] Отец [перекрытие речи, спикер под вопросом]: Да, конечно.\n"
        "это продолжение той же реплики\n"
        "\n"
        "# комментарий проверяющего\n"
        "[00:01:00–00:01:02] Мать: встреча в 10:30 [неразборчиво] завтра\n"
    )

    reference = parse_reference(text)

    assert reference.words == [
        ("да", "Отец"), ("конечно", "Отец"), ("это", "Отец"), ("продолжение", "Отец"),
        ("той", "Отец"), ("же", "Отец"), ("реплики", "Отец"),
        ("встреча", "Мать"), ("в", "Мать"), ("10:30", "Мать"), ("завтра", "Мать"),
    ]
    assert (reference.start, reference.end) == (1.0, 63.0)


def test_a_long_phrase_before_a_colon_is_speech_not_a_name():
    reference = parse_reference("Отец: начнём\nИтак подведём итоги по проекту: всё готово\n")
    assert {speaker for _, speaker in reference.words} == {"Отец"}
    assert ("итоги", "Отец") in reference.words


def test_a_colon_in_continuation_text_is_not_a_speaker_label():
    # "я сказал: нет" is what the turn goes on saying, not a speaker "я сказал".
    reference = parse_reference("Отец: начнём\nя сказал: нет\nа потом ушёл\n")

    assert reference.words == [
        ("начнем", "Отец"), ("я", "Отец"), ("сказал", "Отец"), ("нет", "Отец"),
        ("а", "Отец"), ("потом", "Отец"), ("ушел", "Отец"),
    ]


def test_a_colon_in_continuation_text_does_not_corrupt_the_scores():
    job = merged(("SPEAKER_00", "начнём я сказал нет"))

    result = evaluate(job, "Отец: начнём\nя сказал: нет\n")

    assert result["wer"] == 0
    assert result["speaker_mapping"] == {"SPEAKER_00": "Отец"}
    assert result["speaker_accuracy"] == 1.0
    assert list(result["per_speaker"]) == ["Отец"]


def test_a_new_name_on_an_untimestamped_line_is_refused():
    # Continuing the previous turn or starting Мать's: nothing in the line says
    # which, and guessing would score the words against the wrong person.
    with pytest.raises(ReferenceError, match=r"Line 2.*Мать.*timestamp"):
        parse_reference("Отец: да\nМать: нет\n")


def test_capitalised_prose_before_a_colon_is_refused_not_guessed():
    # It looks like a label, so the author has to settle it; the message says how.
    with pytest.raises(ReferenceError, match=r"Line 2.*Он сказал.*replace the colon"):
        parse_reference("Отец: да\nОн сказал: нет\n")


def test_a_timestamped_line_may_introduce_a_new_speaker():
    reference = parse_reference("Отец: да\n[00:00:05–00:00:06] Мать: нет\n")

    assert reference.words == [("да", "Отец"), ("нет", "Мать")]
    assert (reference.start, reference.end) == (5.0, 7.0)


def test_a_known_name_on_an_untimestamped_line_starts_a_new_turn():
    reference = parse_reference(
        "Отец: раз\n[00:00:05–00:00:06] Мать: два\nОтец: три\nМать: четыре\nпять\n"
    )

    assert reference.words == [
        ("раз", "Отец"), ("два", "Мать"), ("три", "Отец"), ("четыре", "Мать"),
        ("пять", "Мать"),
    ]


def test_a_timestamp_that_only_introduces_a_speaker_can_be_overridden_by_the_window():
    # How a conversation typed from scratch is scored: timestamps only so that
    # each speaker is named, and the window given explicitly.
    job = merged(("SPEAKER_00", "раз два"), ("SPEAKER_01", "три"))

    reference = "Отец: раз два\n[00:09:00–00:09:01] Мать: три\n"

    result = evaluate(job, reference, start=0.0, end=10.0)

    assert result["window"] == [0.0, 10.0]
    assert (result["wer"], result["speaker_accuracy"]) == (0, 1.0)


def test_the_first_line_may_name_the_first_speaker_without_a_timestamp():
    assert parse_reference("Отец: раз\n").words == [("раз", "Отец")]


def test_only_the_start_of_a_line_is_looked_at_for_a_speaker_name():
    # A colon after the first one is text, even if it follows a known name.
    reference = parse_reference(
        "Отец: раз\n[00:00:05–00:00:06] Мать: два\nОтец: она сказала Мать: три\n"
    )

    assert [word for word, _ in reference.words[-4:]] == ["она", "сказала", "мать", "три"]
    assert {speaker for _, speaker in reference.words[-4:]} == {"Отец"}


def test_words_before_any_speaker_are_rejected():
    with pytest.raises(ReferenceError, match="no speaker"):
        parse_reference("просто текст без имени\n")


def test_an_empty_reference_is_rejected():
    with pytest.raises(ReferenceError, match="no words"):
        parse_reference("# только комментарий\n")


# --------------------------------------------------------------------- the CLI


def test_evaluate_command_writes_its_result_and_leaves_the_job_alone(tmp_path, capsys):
    from fourvoices import cli

    job = tmp_path / "job"
    (job / "intermediate").mkdir(parents=True)
    (job / "intermediate" / "merged.json").write_text(
        json.dumps(merged(("SPEAKER_00", "раз два три"))), encoding="utf-8"
    )
    manifest = {"stages": {"merge": {}}}
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    reference = tmp_path / "reference.txt"
    reference.write_text("Отец: раз два четыре\n", encoding="utf-8")

    code = cli.main(["evaluate", "--job-dir", str(job), "--reference", str(reference)])

    assert code == 0
    result = json.loads((job / "evaluation.json").read_text(encoding="utf-8"))
    assert result["substitutions"] == 1 and result["reference"] == "reference.txt"
    assert json.loads((job / "manifest.json").read_text(encoding="utf-8")) == manifest
    out = capsys.readouterr().out
    assert "WER 33.3%" in out and "optimistic" in out


def test_evaluate_refuses_a_merge_the_manifest_does_not_vouch_for(tmp_path, capsys):
    from fourvoices import cli

    job = tmp_path / "job"
    (job / "intermediate").mkdir(parents=True)
    (job / "intermediate" / "merged.json").write_text(json.dumps(merged()), encoding="utf-8")
    (job / "manifest.json").write_text(json.dumps({"stages": {}}), encoding="utf-8")
    reference = tmp_path / "reference.txt"
    reference.write_text("Отец: раз\n", encoding="utf-8")

    assert cli.main(["evaluate", "--job-dir", str(job), "--reference", str(reference)]) == 2
    assert "does not vouch" in capsys.readouterr().err


def test_differences_are_grouped_and_placed_where_to_listen():
    # Words every 0.5 s: "раз" 0.0, "пять" 0.5, "шесть" 1.0, "лишнее" 1.5.
    reference = "Отец: раз два три четыре пять шесть\n"
    job = merged(("SPEAKER_00", "раз пять шесть лишнее"))

    result = evaluate(job, reference)

    assert result["differences"] == [
        # Missed words have no time: they sit at the next recognised word.
        {"at": 0.5, "reference": "два три четыре", "hypothesis": "", "speaker": "Отец"},
        {"at": 1.5, "reference": "", "hypothesis": "лишнее", "speaker": None},
    ]


def test_missed_words_at_the_very_end_take_the_last_time():
    result = evaluate(merged(("A", "раз два")), "Отец: раз два три\n")
    assert result["differences"] == [
        {"at": 0.5, "reference": "три", "hypothesis": "", "speaker": "Отец"}
    ]


def test_mapping_follows_the_words_not_the_alphabet():
    # Alphabetically "Мать" < "Отец", so a first-fit by name order would give
    # SPEAKER_00 to "Мать"; the words say otherwise.
    reference = "Отец: да нет\n[00:00:00–00:00:10] Мать: может быть\n"
    job = merged(("SPEAKER_00", "да нет"), ("SPEAKER_01", "может быть"))

    result = evaluate(job, reference)

    assert result["speaker_mapping"] == {"SPEAKER_00": "Отец", "SPEAKER_01": "Мать"}
    assert result["speaker_accuracy"] == 1.0


def test_unknown_is_not_mapped_even_where_it_would_fit():
    # Were UNKNOWN a cluster like any other, it would take "Мать" and score 100 %.
    reference = "Отец: раз два три\n[00:00:00–00:00:10] Мать: четыре\n"
    job = merged(("SPEAKER_00", "раз два три"), ("UNKNOWN", "четыре"))

    result = evaluate(job, reference)

    assert result["speaker_mapping"] == {"SPEAKER_00": "Отец"}
    assert result["speaker_errors"] == 1


def test_a_reference_span_with_no_recognised_words_scores_everything_as_missed():
    job = merged(("SPEAKER_00", "раз два"))
    result = evaluate(job, "[00:10:00–00:10:05] Отец: где то потом\n")

    assert result["hypothesis_words"] == 0
    assert (result["deletions"], result["wer"]) == (3, 1.0)
    assert result["speaker_accuracy"] is None
