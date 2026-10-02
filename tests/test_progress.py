from fourvoices.progress import DiarizationProgress, Reporter, format_duration


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_durations_read_like_a_clock():
    assert format_duration(0) == "0:00"
    assert format_duration(65.4) == "1:05"
    assert format_duration(3723) == "1:02:03"


def test_updates_are_throttled_but_forced_lines_always_print(capsys):
    clock = Clock()
    reporter = Reporter("GigaSTT", interval=30, clock=clock)

    assert reporter.update("first")  # nothing printed yet, so it is due
    clock.now = 10
    assert not reporter.update("too soon")
    assert reporter.update("forced", force=True)
    clock.now = 41
    assert reporter.update("due again")

    lines = capsys.readouterr().out.splitlines()
    assert [line.strip() for line in lines] == [
        "GigaSTT: first",
        "GigaSTT: forced",
        "GigaSTT: due again",
    ]


def test_a_new_diarization_step_is_shown_at_once_and_percent_is_capped(capsys):
    clock = Clock()
    hook = DiarizationProgress(Reporter("pyannote", interval=30, clock=clock))

    hook("segmentation", None, total=10, completed=0)
    clock.now = 5
    hook("segmentation", None, total=10, completed=5)  # inside the interval: quiet
    hook("embeddings", None, total=4, completed=0)  # new step: printed
    clock.now = 60
    hook("embeddings", None, total=4, completed=6)  # pyannote overshoots by a batch
    hook("embeddings", object())

    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        "pyannote: segmentation 0% (0:00)",
        "pyannote: embeddings 0% (0:05)",
        "pyannote: embeddings 100% (1:00)",
        "pyannote: embeddings done (1:00)",
    ]
