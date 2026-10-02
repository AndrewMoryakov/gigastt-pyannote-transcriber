"""Local web UI for fourvoices.

A deliberately small standard-library HTTP server bound to 127.0.0.1. It adds no
dependencies: the pipeline still runs as a ``python -m fourvoices.cli run``
subprocess, exactly like ``scripts/run.ps1`` does, one job at a time.

Security model: the server listens on the loopback interface only, rejects
requests whose ``Host`` is not loopback (DNS rebinding), requires a custom header
plus a matching ``Origin`` on every POST (cross-site forms), never returns the
Hugging Face token, and only touches files inside ``media/`` and the output root.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "webui_static"
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STAGE_RE = re.compile(r"^\[(\d)/5\]\s*(.*)$")
SAFE_ID_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)
SPEAKER_LABEL_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)
AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
    ".webm", ".mp4", ".mkv", ".mov",
}  # fmt: skip
OUTPUT_FORMATS = ("md", "txt", "srt", "vtt", "json")
MAX_UPLOAD_BYTES = 4 * 1024**3
LOG_LIMIT = 600
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Settings:
    repo: Path
    media_dir: Path
    output_root: Path
    config: Path
    gigastt_exe: Path
    gigastt_model_dir: Path
    pyannote_dir: Path
    env_file: Path

    @classmethod
    def load(cls, repo: Path = REPO) -> Settings:
        config = repo / "config" / "default.yaml"
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        output = (data.get("project") or {}).get("output_root") or "output"
        lock = json.loads((repo / "tools" / "tools.lock.json").read_text(encoding="utf-8"))
        version = lock["gigastt"]["version"]
        return cls(
            repo=repo,
            media_dir=repo / "media",
            output_root=(repo / output).resolve(),
            config=config,
            gigastt_exe=repo / "tools" / "bin" / "gigastt" / version / "gigastt.exe",
            gigastt_model_dir=repo / "models" / "gigastt",
            pyannote_dir=repo / "models" / "pyannote",
            env_file=repo / ".env",
        )


# ------------------------------------------------------------------------ .env helpers


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def write_env_value(path: Path, key: str, value: str) -> None:
    """Set ``key`` in a dotenv file, keeping every other line untouched."""

    if "\n" in value or "\r" in value:
        raise ValueError("value must be a single line")
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().split("=", 1)[0].strip() == key and "=" in line:
            lines[index] = f"{key}={value}"
            replaced = True
    if not replaced:
        if not lines:
            lines.append("# Local secrets. Ignored by Git. Never commit this file.")
        lines.append(f"{key}={value}")
    temporary = path.with_suffix(".env.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mask_token(token: str) -> str:
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "…"


# ------------------------------------------------------------------------ small helpers


def safe_filename(name: str) -> str:
    """Return a harmless basename that keeps a known audio extension."""

    base = Path(name.replace("\\", "/")).name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый тип файла: {suffix or 'без расширения'}")
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE).strip("._") or "audio"
    return f"{stem[:80]}{suffix}"


def unique_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    return directory / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}{suffix}"


def search_path() -> str:
    """PATH plus the places ffmpeg is commonly unpacked to without an installer."""

    extra = [Path.home() / "tools" / "ffmpeg" / "bin"]
    parts = [os.environ.get("PATH", "")] + [str(p) for p in extra if p.is_dir()]
    return os.pathsep.join(part for part in parts if part)


def find_tool(name: str) -> str | None:
    return shutil.which(name, path=search_path())


def job_directory(settings: Settings, identifier: str) -> Path:
    """Resolve a history id to a directory strictly inside the output root."""

    if not SAFE_ID_RE.match(identifier) or identifier in {".", ".."}:
        raise ValueError("Некорректный идентификатор задания")
    path = (settings.output_root / identifier).resolve()
    if path.parent != settings.output_root or not path.is_dir():
        raise FileNotFoundError(identifier)
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def kill_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            creationflags=NO_WINDOW,
            check=False,
        )
    else:
        process.kill()


def base_environment(settings: Settings) -> dict[str, str]:
    """Environment shared by every child process, mirroring scripts/run.ps1."""

    env = dict(os.environ)
    env.update(read_env_file(settings.env_file))
    env["PATH"] = search_path()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # pyannote warns (with a long traceback) that torchcodec cannot find shared FFmpeg
    # DLLs. Audio is decoded by our own ffmpeg step, so the warning is only noise.
    env["PYTHONWARNINGS"] = "ignore::UserWarning:pyannote.audio.core.io"
    env["GIGASTT_PUNCT_MODEL_DIR"] = str(settings.gigastt_model_dir / "punct")
    env["GIGASTT_VAD_MODEL_DIR"] = str(settings.gigastt_model_dir / "vad")
    env["GIGASTT_OFFLINE"] = "1"
    return env


def cli_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "fourvoices.cli", *arguments]


# ------------------------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    file: str
    options: dict[str, Any]
    state: str = "queued"  # queued | running | done | failed | cancelled
    stage: int = 0
    stage_text: str = ""
    log: list[str] = field(default_factory=list)
    output_dir: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self, tail: int = 80) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "options": self.options,
            "state": self.state,
            "stage": self.stage,
            "stage_text": self.stage_text,
            "log": self.log[-tail:],
            "output_dir": self.output_dir,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
        }


class JobManager:
    """Runs queued jobs strictly one at a time, like scripts/run.ps1."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._worker, name="fourvoices-jobs", daemon=True).start()

    def submit(self, file: str, options: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], file=file, options=options)
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, identifier: str) -> Job | None:
        with self._lock:
            return self.jobs.get(identifier)

    def listing(self) -> list[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda job: job.created, reverse=True)

    def cancel(self, identifier: str) -> bool:
        job = self.get(identifier)
        if job is None or job.state not in {"queued", "running"}:
            return False
        was_running = job.state == "running"
        job.state = "cancelled"
        job.finished = time.time()
        if was_running and job.process is not None:
            kill_tree(job.process)
        return True

    def command(self, job: Job) -> list[str]:
        s = self.settings
        opts = job.options
        command = cli_command(
            "run",
            "--input", str(s.media_dir / job.file),
            "--output-root", str(s.output_root),
            "--config", str(s.config),
            "--gigastt-exe", str(s.gigastt_exe),
            "--model-dir", str(s.gigastt_model_dir),
            "--num-speakers", str(opts["num_speakers"]),
            "--ffmpeg", find_tool("ffmpeg") or "ffmpeg",
            "--ffprobe", find_tool("ffprobe") or "ffprobe",
        )  # fmt: skip
        if opts.get("allow_downmix"):
            command.append("--allow-downmix")
        if opts.get("force"):
            command.append("--force")
        return command

    def _worker(self) -> None:
        while True:
            job = self.get(self._queue.get())
            if job is None or job.state != "queued":
                continue
            try:
                self._run(job)
            except Exception as exc:  # keep the worker alive whatever happens
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = time.time()

    def _run(self, job: Job) -> None:
        job.state = "running"
        process = subprocess.Popen(
            self.command(job),
            cwd=self.settings.repo,
            env=base_environment(self.settings),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=NO_WINDOW,
        )
        job.process = process
        last_error = ""
        assert process.stdout is not None
        for raw in process.stdout:
            line = ANSI_RE.sub("", raw).rstrip()
            if not line:
                continue
            job.log.append(line)
            del job.log[:-LOG_LIMIT]
            if match := STAGE_RE.match(line):
                job.stage, job.stage_text = int(match.group(1)), match.group(2)
            elif line.startswith("Done: "):
                job.output_dir = Path(line[6:].strip()).name
            elif line.startswith("error:"):
                last_error = line[6:].strip()
        code = process.wait()
        job.process = None
        if job.state == "cancelled":
            return
        job.finished = time.time()
        if code == 0:
            job.state, job.stage = "done", 5
        else:
            job.state = "failed"
            job.error = last_error or (job.log[-1] if job.log else f"Код выхода {code}")


# ----------------------------------------------------------------------------- history


def summarize_job_dir(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError):
        return None
    render = manifest.get("stages", {}).get("render", {})
    return {
        "id": path.name,
        "input": manifest.get("input", {}).get("name", path.name),
        "updated_at": manifest.get("updated_at", ""),
        "complete": bool(render),
        "stages": sorted(manifest.get("stages", {})),
        "files": [Path(item).name for item in render.get("files", [])],
        "speaker_names": render.get("speaker_names", {}),
    }


def history_listing(settings: Settings) -> list[dict[str, Any]]:
    if not settings.output_root.is_dir():
        return []
    items = [
        summary
        for child in settings.output_root.iterdir()
        if child.is_dir() and (summary := summarize_job_dir(child))
    ]
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def history_detail(settings: Settings, identifier: str) -> dict[str, Any]:
    path = job_directory(settings, identifier)
    summary = summarize_job_dir(path)
    if summary is None:
        raise FileNotFoundError(identifier)
    merged = path / "intermediate" / "merged.json"
    summary["speakers"] = load_json(merged).get("speakers", []) if merged.is_file() else []
    return summary


def apply_speaker_names(settings: Settings, identifier: str, names: dict[str, str]) -> None:
    detail = history_detail(settings, identifier)
    known = set(detail["speakers"])
    arguments = ["render", "--job-dir", str(job_directory(settings, identifier))]
    for label, name in names.items():
        name = name.strip()
        if not SPEAKER_LABEL_RE.match(label) or label not in known:
            raise ValueError(f"Неизвестный говорящий: {label}")
        if len(name) > 80 or "\n" in name or "\r" in name:
            raise ValueError("Имя должно быть одной строкой до 80 символов")
        if name:
            arguments += ["--speaker-name", f"{label}={name}"]
    result = subprocess.run(
        cli_command(*arguments),
        cwd=settings.repo,
        env=base_environment(settings),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip().splitlines()[-1])


# ----------------------------------------------------------------------------- server


class App:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs = JobManager(settings)
        self.port = 0

    def token(self) -> str:
        return read_env_file(self.settings.env_file).get("HF_TOKEN") or os.environ.get(
            "HF_TOKEN", ""
        )

    def status(self) -> dict[str, Any]:
        s = self.settings
        token = self.token()
        models = {
            "gigastt": s.gigastt_exe.is_file()
            and (s.gigastt_model_dir / "v3_rnnt_encoder_int8.onnx").is_file(),
            "pyannote": any((s.pyannote_dir).glob("models--*/snapshots/*")),
        }
        with contextlib.suppress(OSError):
            s.media_dir.mkdir(exist_ok=True)
        free = shutil.disk_usage(s.repo).free
        return {
            "token": {
                "set": bool(TOKEN_RE.match(token)),
                "masked": mask_token(token) if token else "",
            },
            "ffmpeg": bool(find_tool("ffmpeg") and find_tool("ffprobe")),
            "models": models,
            "ready": all(models.values()) and bool(find_tool("ffmpeg")),
            "free_gb": round(free / 1024**3, 1),
            "media": sorted(
                (p.name for p in s.media_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS),
                key=str.lower,
            )
            if s.media_dir.is_dir()
            else [],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "fourvoices-ui"
    app: App

    # -- plumbing

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{self.app.port}", f"localhost:{self.app.port}"}

    def _send(self, status: int, body: bytes, content_type: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            raise ValueError("Слишком большой запрос")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("Ожидался JSON-объект")
        return value

    def _dispatch(self, method: str) -> None:
        if not self._host_allowed():
            return self._error(HTTPStatus.FORBIDDEN, "Недопустимый Host")
        if method == "POST":
            origin = self.headers.get("Origin")
            host = (self.headers.get("Host") or "").lower()
            if self.headers.get("X-FV") != "1" or (
                origin and urlparse(origin).netloc.lower() != host
            ):
                return self._error(HTTPStatus.FORBIDDEN, "Запрос отклонён")
        url = urlparse(self.path)
        try:
            handler = self._route(method, url.path)
            if handler is None:
                return self._error(HTTPStatus.NOT_FOUND, "Не найдено")
            handler(parse_qs(url.query))
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "Не найдено")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    # -- routing

    def _route(self, method: str, path: str):  # noqa: ANN202
        parts = [unquote(part) for part in path.strip("/").split("/")]
        if method == "GET":
            if path == "/":
                return self._index
            if path == "/api/status":
                return lambda q: self._json(self.app.status())
            if path == "/api/jobs":
                return lambda q: self._json([j.public(10) for j in self.app.jobs.listing()])
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                return lambda q: self._job(parts[2])
            if path == "/api/history":
                return lambda q: self._json(history_listing(self.app.settings))
            if len(parts) == 3 and parts[:2] == ["api", "history"]:
                return lambda q: self._json(history_detail(self.app.settings, parts[2]))
            if len(parts) == 5 and parts[:2] == ["api", "history"] and parts[3] == "file":
                return lambda q: self._history_file(parts[2], parts[4], q)
        if method == "POST":
            if path == "/api/token":
                return self._save_token
            if path == "/api/upload":
                return self._upload
            if path == "/api/jobs":
                return self._start_job
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                return lambda q: self._json({"ok": self.app.jobs.cancel(parts[2])})
            if len(parts) == 4 and parts[:2] == ["api", "history"] and parts[3] == "speakers":
                return lambda q: self._rename(parts[2])
        return None

    # -- handlers

    def _index(self, _query: dict[str, list[str]]) -> None:
        self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")

    def _job(self, identifier: str) -> None:
        job = self.app.jobs.get(identifier)
        if job is None:
            raise FileNotFoundError(identifier)
        self._json(job.public())

    def _save_token(self, _query: dict[str, list[str]]) -> None:
        token = str(self._body_json().get("token", "")).strip()
        if not TOKEN_RE.match(token):
            raise ValueError("Это не похоже на токен Hugging Face (должен начинаться с hf_).")
        write_env_value(self.app.settings.env_file, "HF_TOKEN", token)
        self._json({"ok": True, "masked": mask_token(token)})

    def _upload(self, query: dict[str, list[str]]) -> None:
        settings = self.app.settings
        name = safe_filename((query.get("name") or [""])[0])
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD_BYTES:
            raise ValueError("Пустой или слишком большой файл")
        if length > shutil.disk_usage(settings.repo).free - 2 * 1024**3:
            raise ValueError("Недостаточно места на диске")
        settings.media_dir.mkdir(exist_ok=True)
        target = unique_path(settings.media_dir, name)
        partial = target.with_name(target.name + ".part")
        remaining = length
        try:
            with partial.open("wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ValueError("Загрузка прервана")
                    handle.write(chunk)
                    remaining -= len(chunk)
            if (query.get("convert") or ["0"])[0] == "1":
                # Browser recordings (webm/opus) have no duration header, which the
                # pipeline's ffprobe check rejects, so normalise them to WAV here.
                target = unique_path(settings.media_dir, target.stem + ".wav")
                self._convert_to_wav(partial, target)
            else:
                os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        self._json({"file": target.name, "size": target.stat().st_size})

    @staticmethod
    def _convert_to_wav(source: Path, target: Path) -> None:
        ffmpeg = find_tool("ffmpeg")
        if not ffmpeg:
            raise ValueError("ffmpeg не найден")
        result = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y", "-i", str(source),
             "-ac", "1", "-ar", "16000", str(target)],
            capture_output=True,
            text=True,
            creationflags=NO_WINDOW,
            check=False,
        )  # fmt: skip
        if result.returncode != 0 or not target.is_file():
            target.unlink(missing_ok=True)
            raise ValueError("Не удалось обработать запись: " + result.stderr.strip()[-200:])

    def _start_job(self, _query: dict[str, list[str]]) -> None:
        app = self.app
        body = self._body_json()
        if not TOKEN_RE.match(app.token()):
            raise ValueError("Сначала сохраните токен Hugging Face.")
        status = app.status()
        if not status["ready"]:
            raise ValueError("Окружение не готово: проверьте ffmpeg и модели.")
        name = Path(str(body.get("file", ""))).name
        if name != body.get("file") or not (app.settings.media_dir / name).is_file():
            raise ValueError("Файл не найден в папке media")
        speakers = int(body.get("num_speakers", 2))
        if not 1 <= speakers <= 32:
            raise ValueError("Число говорящих должно быть от 1 до 32")
        job = app.jobs.submit(
            name,
            {
                "num_speakers": speakers,
                "allow_downmix": bool(body.get("allow_downmix")),
                "force": bool(body.get("force")),
            },
        )
        self._json(job.public(), HTTPStatus.ACCEPTED)

    def _history_file(self, identifier: str, name: str, query: dict[str, list[str]]) -> None:
        settings = self.app.settings
        path = job_directory(settings, identifier)
        suffix = Path(name).suffix.lstrip(".")
        allowed = set(history_detail(settings, identifier)["files"])
        if name not in allowed or suffix not in OUTPUT_FORMATS or Path(name).name != name:
            raise FileNotFoundError(name)
        data = (path / name).read_bytes()
        content_type = "application/json" if suffix == "json" else "text/plain"
        headers: dict[str, str] = {}
        if (query.get("download") or ["0"])[0] == "1":
            encoded = quote(name, safe="")
            headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded}"
        self._send(200, data, f"{content_type}; charset=utf-8", **headers)

    def _rename(self, identifier: str) -> None:
        names = self._body_json().get("names")
        if not isinstance(names, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in names.items()
        ):
            raise ValueError("Ожидался объект вида {метка: имя}")
        try:
            apply_speaker_names(self.app.settings, identifier, names)
        except RuntimeError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        else:
            self._json(history_detail(self.app.settings, identifier))


def create_server(settings: Settings, host: str = "127.0.0.1", port: int = 8765):
    """Bind the first free port from ``port`` upwards and return the server."""

    app = App(settings)
    handler = type("BoundHandler", (Handler,), {"app": app})
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            continue
        app.port = candidate
        return server, app
    raise OSError(f"Нет свободного порта {port}-{port + 19}: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fourvoices-ui", description="Local web UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.load()
    server, app = create_server(settings, port=args.port)
    url = f"http://127.0.0.1:{app.port}/"
    print(f"fourvoices UI: {url}  (Ctrl+C — остановить)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлено.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
