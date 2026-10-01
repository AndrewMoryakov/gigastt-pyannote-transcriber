import json
import subprocess

from fourvoices.gigastt import transcribe


def test_transcribe_uses_supported_gigastt_flags(tmp_path, monkeypatch):
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


# Captured from GigaSTT 2.15.0 on a 28-minute recording: exit code 0, yet the
# transcript came back without a single punctuation mark.
DEGRADED_LOG = (
    "\x1b[2m2026-09-30T22:57:10Z\x1b[0m \x1b[32m INFO\x1b[0m gigastt_core::model: Using model\n"
    "\x1b[2m2026-09-30T22:57:10Z\x1b[0m \x1b[33m WARN\x1b[0m "
    "\x1b[2mgigastt_core::inference::diarization\x1b[0m: wespeaker_resnet34.onnx not found, "
    "diarization unavailable\n"
    "\x1b[2m2026-09-30T22:57:19Z\x1b[0m \x1b[33m WARN\x1b[0m "
    "\x1b[2mgigastt_core::punctuation\x1b[0m: Punctuation restore failed, returning bare text\n"
)


def fake_gigastt(log="", text="тест", returncode=0, stream="stdout"):
    # The real binary writes its tracing log to stdout (2.15.0 and 2.21.0 both).
    stdout, stderr = (log, "") if stream == "stdout" else ("", log)

    def fake_run(command, **kwargs):
        if returncode:
            raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
        output = command[command.index("--output") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "duration": 1.0,
                    "text": text,
                    "words": [{"word": "тест", "start": 0.1, "end": 0.5, "confidence": 0.9}],
                },
                stream,
            )
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    return fake_run


def test_the_log_is_kept_without_colour_codes(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    monkeypatch.setattr(subprocess, "run", fake_gigastt(DEGRADED_LOG))

    transcribe(source, tmp_path / "gigastt.json", executable="gigastt.exe")

    log = (tmp_path / "gigastt.log").read_text(encoding="utf-8")
    assert "Punctuation restore failed" in log
    assert "\x1b" not in log


def test_warnings_reach_the_user_but_expected_noise_does_not(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    monkeypatch.setattr(subprocess, "run", fake_gigastt(DEGRADED_LOG))

    transcribe(source, tmp_path / "gigastt.json", executable="gigastt.exe")

    error = capsys.readouterr().err
    assert "Punctuation restore failed" in error
    assert "wespeaker" not in error


def test_log_problems_ignores_info_lines():
    from fourvoices.gigastt import log_problems

    problems = log_problems(DEGRADED_LOG)
    assert len(problems) == 1
    assert problems[0].endswith("Punctuation restore failed, returning bare text")


def test_a_failure_names_the_log_and_quotes_its_end(tmp_path, monkeypatch):
    import pytest

    from fourvoices.gigastt import GigaSTTError

    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    stderr = "".join(f"INFO line {index}\n" for index in range(40))
    stderr += "Error: invalid audio: Audio file too long (1800s). Maximum supported: 1800s.\n"
    monkeypatch.setattr(subprocess, "run", fake_gigastt(stderr, returncode=1))

    with pytest.raises(GigaSTTError) as error:
        transcribe(source, tmp_path / "gigastt.json", executable="gigastt.exe")

    message = str(error.value)
    assert "gigastt.log" in message
    assert "Audio file too long" in message
    assert "INFO line 0" not in message
    assert (tmp_path / "gigastt.log").read_text(encoding="utf-8") == stderr


def test_gigastt_version_is_parsed(monkeypatch):
    from fourvoices.gigastt import gigastt_version

    def fake_run(command, **kwargs):
        assert command[1:] == ["--version"]
        return subprocess.CompletedProcess(command, 0, "gigastt 2.21.0\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gigastt_version("gigastt.exe") == "2.21.0"


def test_a_missing_executable_has_no_version(tmp_path):
    from fourvoices.gigastt import gigastt_version

    assert gigastt_version(str(tmp_path / "missing.exe")) is None


def _payload(count, text):
    words = [{"word": "слово", "start": 0.0, "end": 0.1} for _ in range(count)]
    return {"words": words, "text": text}


def test_punctuation_is_missing_only_when_requested_and_the_text_is_long():
    from fourvoices.gigastt import punctuation_missing

    bare = " ".join(["слово"] * 300)
    assert punctuation_missing(_payload(300, bare), "on")
    assert not punctuation_missing(_payload(300, bare + "."), "on")
    assert not punctuation_missing(_payload(300, bare), "off")
    assert not punctuation_missing(_payload(300, bare), "auto")
    # A short answer such as "да нет наверное" legitimately has no marks.
    assert not punctuation_missing(_payload(3, "да нет наверное"), "on")


def test_a_log_printed_to_stderr_is_kept_as_well(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    monkeypatch.setattr(subprocess, "run", fake_gigastt(DEGRADED_LOG, stream="stderr"))

    transcribe(source, tmp_path / "gigastt.json", executable="gigastt.exe")

    log = (tmp_path / "gigastt.log").read_text(encoding="utf-8")
    assert "Punctuation restore failed" in log
    assert "Punctuation restore failed" in capsys.readouterr().err
