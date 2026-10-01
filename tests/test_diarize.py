import json

import pytest

from fourvoices import diarize as diarize_module
from fourvoices.diarize import SpeakerCountMismatch, load_diarization


class _Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for start, end, label in self._tracks:
            yield _Segment(start, end), None, label


class _Output:
    def __init__(self, tracks):
        self.speaker_diarization = _Annotation(tracks)
        self.exclusive_speaker_diarization = _Annotation(tracks)


class _Pipeline:
    def __init__(self, tracks):
        self._tracks = tracks

    def to(self, device):
        return self

    def __call__(self, _audio, num_speakers=None):
        return _Output(self._tracks)


def _install_stubs(monkeypatch, tmp_path, tracks):
    """Replace the model load and audio decode so no weights or WAV are needed."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        diarize_module, "_load_waveform", lambda path: (object(), 16000, 12.0)
    )

    class _FakePipelineClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Pipeline(tracks)

    import sys
    import types

    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda n: None,
        set_num_interop_threads=lambda n: None,
        device=lambda name: name,
    )
    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = _FakePipelineClass
    fake_pyannote = types.ModuleType("pyannote")
    fake_pyannote.audio = fake_pyannote_audio
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    source = tmp_path / "diarization.wav"
    source.write_bytes(b"RIFF")
    return source


TWO_SPEAKERS = [
    (0.0, 1.0, "SPEAKER_00"),
    (1.0, 2.0, "SPEAKER_01"),
]


def test_speaker_count_mismatch_still_persists_the_expensive_result(
    tmp_path, monkeypatch, capsys
):
    source = _install_stubs(monkeypatch, tmp_path, TWO_SPEAKERS)
    destination = tmp_path / "pyannote.json"

    payload = diarize_module.diarize(source, destination, num_speakers=4)

    assert destination.is_file()
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["found_speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert saved["speaker_count_matches_request"] is False
    assert saved["requested_num_speakers"] == 4
    assert payload["exclusive_segments"]
    assert "warning" in capsys.readouterr().err


def test_strict_mode_raises_but_only_after_writing_the_result(tmp_path, monkeypatch):
    source = _install_stubs(monkeypatch, tmp_path, TWO_SPEAKERS)
    destination = tmp_path / "pyannote.json"

    with pytest.raises(SpeakerCountMismatch):
        diarize_module.diarize(source, destination, num_speakers=4, strict_speakers=True)

    assert destination.is_file()
    assert load_diarization(destination)["exclusive_segments"]


def test_matching_speaker_count_is_silent(tmp_path, monkeypatch, capsys):
    source = _install_stubs(monkeypatch, tmp_path, TWO_SPEAKERS)
    destination = tmp_path / "pyannote.json"

    payload = diarize_module.diarize(source, destination, num_speakers=2)

    assert payload["speaker_count_matches_request"] is True
    assert capsys.readouterr().err == ""


def test_telemetry_stays_off_even_if_the_shell_turned_it_on():
    import os
    import subprocess
    import sys

    # A fresh interpreter: reloading the module here would replace the exception
    # classes other modules already imported from it.
    env = dict(os.environ, PYANNOTE_METRICS_ENABLED="1", HF_HUB_DISABLE_TELEMETRY="0")
    script = (
        "import os, fourvoices.diarize; "
        "print(os.environ['PYANNOTE_METRICS_ENABLED'], os.environ['HF_HUB_DISABLE_TELEMETRY'])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert completed.stdout.split() == ["0", "1"]
