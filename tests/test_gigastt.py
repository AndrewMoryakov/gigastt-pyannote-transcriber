import json
import subprocess

from fourvoices.gigastt import transcribe


def test_transcribe_uses_supported_v215_flags(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    destination = tmp_path / "result.json"
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output = command[command.index("--output") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "duration": 1.0,
                    "text": "тест",
                    "words": [
                        {
                            "word": "тест",
                            "start": 0.1,
                            "end": 0.5,
                            "confidence": 0.9,
                        }
                    ],
                },
                stream,
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    payload = transcribe(
        source,
        destination,
        executable="gigastt.exe",
        model_dir=tmp_path / "models",
    )

    assert "--skip-diarization" not in seen["command"]
    assert "--word-timestamps" in seen["command"]
    assert "--vad" in seen["command"]
    assert payload["words"][0]["word"] == "тест"
    assert destination.is_file()


def test_unparsable_output_is_kept_for_inspection(tmp_path, monkeypatch):
    import pytest

    from fourvoices.gigastt import GigaSTTError

    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    destination = tmp_path / "result.json"

    def fake_run(command, **kwargs):
        output = command[command.index("--output") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            stream.write('{"segments": []}')  # no word timestamps
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GigaSTTError, match="rejected"):
        transcribe(source, destination, executable="gigastt.exe")

    rejected = tmp_path / "result.json.rejected"
    assert rejected.read_text(encoding="utf-8") == '{"segments": []}'
    assert not destination.exists()
    assert not (tmp_path / "result.json.partial").exists()
