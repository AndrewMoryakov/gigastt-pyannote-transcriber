import http.client
import json
import shutil
import sys
import threading
import time
from collections import namedtuple

import pytest

from fourvoices import webui

TOKEN = "hf_" + "a" * 30


def test_safe_filename_strips_paths_and_checks_extension():
    assert webui.safe_filename("..\\..\\evil dir\\Интервью 1.M4A") == "Интервью_1.m4a"
    assert webui.safe_filename("a/b/c.wav") == "c.wav"
    with pytest.raises(ValueError):
        webui.safe_filename("notes.txt")
    with pytest.raises(ValueError):
        webui.safe_filename("noextension")


def test_env_file_keeps_other_lines_and_replaces_in_place(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nHF_HOME=models/huggingface\nHF_TOKEN=hf_old\n", encoding="utf-8")

    webui.write_env_value(env, "HF_TOKEN", TOKEN)

    assert env.read_text(encoding="utf-8").splitlines() == [
        "# comment",
        "HF_HOME=models/huggingface",
        f"HF_TOKEN={TOKEN}",
    ]
    assert webui.read_env_file(env)["HF_TOKEN"] == TOKEN
    with pytest.raises(ValueError):
        webui.write_env_value(env, "HF_TOKEN", "a\nb")


def test_env_file_is_created_when_missing(tmp_path):
    env = tmp_path / ".env"
    webui.write_env_value(env, "HF_TOKEN", TOKEN)
    assert webui.read_env_file(env) == {"HF_TOKEN": TOKEN}


def test_child_environment_silences_the_torchcodec_warning(tmp_path):
    env = webui.base_environment(make_settings(tmp_path))
    assert env["PYTHONWARNINGS"] == "ignore::UserWarning:pyannote.audio.core.io"
    assert env["PYTHONUTF8"] == "1"


def test_mask_token_never_reveals_the_middle():
    masked = webui.mask_token(TOKEN)
    assert masked == "hf_…aaaa"
    assert TOKEN not in masked


def make_settings(root):
    (root / "config").mkdir()
    (root / "config" / "default.yaml").write_text(
        "project:\n  output_root: out\n", encoding="utf-8"
    )
    (root / "tools").mkdir()
    (root / "tools" / "tools.lock.json").write_text(
        json.dumps({"gigastt": {"version": "9.9.9"}}), encoding="utf-8"
    )
    settings = webui.Settings.load(root)
    settings.gigastt_exe.parent.mkdir(parents=True)
    settings.gigastt_exe.write_bytes(b"x")
    settings.gigastt_model_dir.mkdir(parents=True)
    (settings.gigastt_model_dir / "v3_rnnt_encoder_int8.onnx").write_bytes(b"x")
    (settings.pyannote_dir / "models--p--m" / "snapshots" / "rev").mkdir(parents=True)
    return settings


def make_job_dir(settings, name="talk-abc", speakers=("SPEAKER_00", "SPEAKER_01")):
    job = settings.output_root / name
    (job / "intermediate").mkdir(parents=True)
    (job / "intermediate" / "merged.json").write_text(
        json.dumps({"speakers": list(speakers)}), encoding="utf-8"
    )
    (job / "transcript.txt").write_text("привет", encoding="utf-8")
    (job / "transcript.json").write_text("{}", encoding="utf-8")
    (job / "manifest.json").write_text(
        json.dumps(
            {
                "input": {"name": "talk.m4a"},
                "updated_at": "2026-01-02T03:04:05",
                "stages": {
                    "render": {
                        "files": ["transcript.txt", "transcript.json"],
                        "speaker_names": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return job


class Client:
    def __init__(self, port):
        self.port = port

    def request(self, method, path, body=None, headers=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Host": host or f"127.0.0.1:{self.port}"}
        if method == "POST":
            merged["X-FV"] = "1"
        merged.update(headers or {})
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            merged["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data, response

    def json(self, method, path, body=None, **kwargs):
        status, data, _ = self.request(method, path, body, **kwargs)
        return status, json.loads(data)


_DiskUsage = namedtuple("_DiskUsage", "total used free")


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(webui, "find_tool", lambda name: name)
    # Uploads keep a 2 GB free-space reserve; do not let the machine running the
    # tests (a small tmpfs, a nearly full disk) decide whether they pass.
    monkeypatch.setattr(webui.shutil, "disk_usage", lambda path: _DiskUsage(10**13, 0, 10**13))
    settings = make_settings(tmp_path)
    server, app = webui.create_server(settings, port=18765)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield settings, app, Client(app.port)
    server.shutdown()
    server.server_close()


def test_status_reports_missing_token_and_ready_environment(ui):
    _, _, client = ui
    status, data = client.json("GET", "/api/status")
    assert status == 200
    assert data["token"] == {"set": False, "masked": ""}
    assert data["models"] == {"gigastt": True, "pyannote": True}
    assert data["ready"] is True


def test_token_is_saved_to_env_and_never_returned(ui):
    settings, _, client = ui
    status, data = client.json("POST", "/api/token", {"token": "nope"})
    assert status == 400

    status, data = client.json("POST", "/api/token", {"token": TOKEN})
    assert status == 200
    assert webui.read_env_file(settings.env_file)["HF_TOKEN"] == TOKEN

    _, raw, _ = client.request("GET", "/api/status")
    assert TOKEN.encode() not in raw
    assert json.loads(raw)["token"]["set"] is True


def test_requests_with_foreign_host_or_without_header_are_rejected(ui):
    _, _, client = ui
    assert client.request("GET", "/api/status", host="evil.example")[0] == 403
    status, _, _ = client.request("POST", "/api/token", {"token": TOKEN}, headers={"X-FV": "0"})
    assert status == 403
    status, _, _ = client.request(
        "POST",
        "/api/token",
        {"token": TOKEN},
        headers={"Origin": "http://evil.example"},
    )
    assert status == 403


def test_upload_goes_to_media_with_a_sanitised_name(ui):
    settings, _, client = ui
    status, data = client.json(
        "POST", "/api/upload?name=..%5C..%5Cx%20y.mp3", b"abc", headers={"Content-Length": "3"}
    )
    assert status == 200
    assert data["file"] == "x_y.mp3"
    assert (settings.media_dir / "x_y.mp3").read_bytes() == b"abc"
    assert not list(settings.media_dir.glob("*.part"))

    status, _ = client.json("POST", "/api/upload?name=bad.exe", b"abc")
    assert status == 400


def test_upload_is_refused_when_the_disk_is_nearly_full(ui, monkeypatch):
    settings, _, client = ui
    monkeypatch.setattr(webui.shutil, "disk_usage", lambda path: _DiskUsage(10**13, 0, 1024**3))

    status, data = client.json(
        "POST", "/api/upload?name=x.mp3", b"abc", headers={"Content-Length": "3"}
    )

    assert status == 400
    assert not (settings.media_dir / "x.mp3").exists()


def test_history_lists_details_and_serves_only_known_files(ui):
    settings, _, client = ui
    make_job_dir(settings)
    (settings.output_root / "talk-abc" / "secret.txt").write_text("no", encoding="utf-8")

    status, items = client.json("GET", "/api/history")
    assert status == 200
    assert [item["id"] for item in items] == ["talk-abc"]

    status, detail = client.json("GET", "/api/history/talk-abc")
    assert detail["speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    status, body, response = client.request("GET", "/api/history/talk-abc/file/transcript.txt")
    assert (status, body.decode()) == (200, "привет")
    assert "text/plain" in response.getheader("Content-Type")

    status, _, response = client.request(
        "GET", "/api/history/talk-abc/file/transcript.txt?download=1"
    )
    assert "attachment" in response.getheader("Content-Disposition")

    assert client.request("GET", "/api/history/talk-abc/file/secret.txt")[0] == 404
    assert client.request("GET", "/api/history/talk-abc/file/..%2Fmanifest.json")[0] == 404
    assert client.request("GET", "/api/history/..%2F..%2Fetc")[0] in {400, 404}
    assert client.request("GET", "/api/history/missing")[0] == 404


def test_speaker_rename_rejects_unknown_labels(ui):
    settings, _, client = ui
    make_job_dir(settings)
    status, data = client.json(
        "POST", "/api/history/talk-abc/speakers", {"names": {"SPEAKER_09": "Ольга"}}
    )
    assert status == 400
    assert "SPEAKER_09" in data["error"]


def test_job_requires_token_and_existing_media_file(ui, monkeypatch):
    settings, app, client = ui
    status, data = client.json("POST", "/api/jobs", {"file": "a.wav"})
    assert status == 400 and "токен" in data["error"]

    client.json("POST", "/api/token", {"token": TOKEN})
    status, data = client.json("POST", "/api/jobs", {"file": "missing.wav"})
    assert status == 400

    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")
    status, data = client.json("POST", "/api/jobs", {"file": "a.wav", "num_speakers": 99})
    assert status == 400


def test_job_runs_to_completion_and_reports_stages(ui, monkeypatch):
    settings, app, client = ui
    client.json("POST", "/api/token", {"token": TOKEN})
    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")
    script = (
        "print('[1/5] Preparing…', flush=True);"
        "print('\\x1b[2mnoise\\x1b[0m', flush=True);"
        "print('[5/5] Rendering md…', flush=True);"
        f"print('Done: ' + {str(settings.output_root / 'a-123')!r}, flush=True)"
    )
    monkeypatch.setattr(
        webui.JobManager, "command", lambda self, job: [sys.executable, "-c", script]
    )

    status, job = client.json("POST", "/api/jobs", {"file": "a.wav", "num_speakers": 2})
    assert status == 202

    for _ in range(100):
        _, current = client.json("GET", f"/api/jobs/{job['id']}")
        if current["state"] not in {"queued", "running"}:
            break
        time.sleep(0.1)

    assert current["state"] == "done"
    assert current["stage"] == 5
    assert current["output_dir"] == "a-123"
    assert "noise" in current["log"] and not any("\x1b" in line for line in current["log"])


def test_failed_job_surfaces_the_cli_error_message(ui, monkeypatch):
    settings, app, client = ui
    client.json("POST", "/api/token", {"token": TOKEN})
    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")
    script = (
        "print('error: Existing job belongs to a different input', flush=True);"
        "raise SystemExit(2)"
    )
    monkeypatch.setattr(
        webui.JobManager, "command", lambda self, job: [sys.executable, "-c", script]
    )

    _, job = client.json("POST", "/api/jobs", {"file": "a.wav", "num_speakers": 2})
    for _ in range(100):
        _, current = client.json("GET", f"/api/jobs/{job['id']}")
        if current["state"] not in {"queued", "running"}:
            break
        time.sleep(0.1)

    assert current["state"] == "failed"
    assert current["error"] == "Existing job belongs to a different input"


def test_job_without_diarization_needs_neither_token_nor_pyannote(ui):
    settings, app, client = ui
    shutil.rmtree(settings.pyannote_dir)
    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")

    _, status = client.json("GET", "/api/status")
    assert status["ready"] is False and status["ready_without_diarization"] is True

    refused, data = client.json("POST", "/api/jobs", {"file": "a.wav"})
    assert refused == 400 and "токен" in data["error"]

    accepted, job = client.json("POST", "/api/jobs", {"file": "a.wav", "diarize": False})
    assert accepted == 202
    assert job["options"]["diarize"] is False


def test_diarization_off_still_needs_gigastt_and_ffmpeg(ui, monkeypatch):
    settings, app, client = ui
    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")
    monkeypatch.setattr(webui, "find_tool", lambda name: None)

    status, data = client.json("POST", "/api/jobs", {"file": "a.wav", "diarize": False})

    assert status == 400 and "не готово" in data["error"]


def test_diarize_must_be_a_boolean(ui):
    _, _, client = ui

    status, data = client.json("POST", "/api/jobs", {"file": "a.wav", "diarize": "no"})

    assert status == 400 and "diarize" in data["error"]


def test_command_states_the_diarization_choice_explicitly(ui):
    settings, app, _ = ui

    def command(**options):
        job = webui.Job(id="x", file="a.wav", options=options)
        return app.jobs.command(job)

    on = command(num_speakers=3)
    assert "--diarization" in on and "--no-diarization" not in on
    assert on[on.index("--num-speakers") + 1] == "3"

    off = command(diarize=False, num_speakers=3)
    assert "--no-diarization" in off and "--diarization" not in off
    assert "--num-speakers" not in off


@pytest.mark.parametrize("diarize", [True, False])
def test_a_job_needs_ffprobe_as_well_as_ffmpeg(ui, monkeypatch, diarize):
    settings, app, client = ui
    client.json("POST", "/api/token", {"token": TOKEN})
    settings.media_dir.mkdir(exist_ok=True)
    (settings.media_dir / "a.wav").write_bytes(b"x")
    monkeypatch.setattr(webui, "find_tool", lambda name: None if name == "ffprobe" else name)

    _, status = client.json("GET", "/api/status")
    assert status["ffmpeg"] is False
    assert status["ready"] is False and status["ready_without_diarization"] is False

    code, data = client.json("POST", "/api/jobs", {"file": "a.wav", "diarize": diarize})
    assert code == 400 and "не готово" in data["error"]
