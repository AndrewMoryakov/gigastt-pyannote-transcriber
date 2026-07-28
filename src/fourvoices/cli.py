"""CLI for inference, model preloading, diagnostics, and inference-free rerendering."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import __version__
from .audio import DEFAULT_ASR_FILTER, AudioPreparationError, prepare_audio
from .diarize import (
    DEFAULT_NUM_SPEAKERS,
    MODEL_ID,
    MODEL_REVISION,
    DiarizationError,
    diarize,
    load_diarization,
    preload_model,
)
from .gigastt import GigaSTTError, load_transcript, transcribe
from .merge import merge_transcript
from .render import write_outputs


class PipelineError(RuntimeError):
    """Configuration, validation, or resumability check failed."""


# Every key the pipeline actually reads. Anything else is rejected instead of
# being silently ignored, so a typo or a stale key cannot look like a setting.
CONFIG_SCHEMA: dict[str, frozenset[str]] = {
    "project": frozenset({"output_root"}),
    "audio": frozenset({"asr_filter", "diarization_filter"}),
    "asr": frozenset({"model_variant", "model_dir", "punctuation", "itn", "vad"}),
    "diarization": frozenset(
        {"model", "revision", "model_dir", "device", "num_speakers"}
    ),
    "merge": frozenset({"max_turn_gap", "nearest_max_gap"}),
    "output": frozenset({"formats", "mark_uncertain_words"}),
}


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unknown sections and keys."""

    problems: list[str] = []
    for section, value in config.items():
        if section not in CONFIG_SCHEMA:
            problems.append(f"unknown section '{section}'")
            continue
        if value is None:
            continue
        if not isinstance(value, Mapping):
            problems.append(f"section '{section}' must be a mapping")
            continue
        for key in value:
            if key not in CONFIG_SCHEMA[section]:
                problems.append(f"unknown key '{section}.{key}'")
    if problems:
        raise PipelineError(
            "Configuration problems: "
            + "; ".join(sorted(problems))
            + ". Supported keys: "
            + ", ".join(
                f"{section}.{key}"
                for section in sorted(CONFIG_SCHEMA)
                for key in sorted(CONFIG_SCHEMA[section])
            )
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "name": path.name,  # never store a machine-specific absolute source path
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{path} must contain a JSON object.")
    return value


def _load_yaml(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        return {}, None
    config_path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError("The YAML configuration root must be a mapping.")
    return value, config_path


def _load_config(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    """Load and validate the pipeline configuration (not the speaker map)."""

    config, config_path = _load_yaml(path)
    validate_config(config)
    return config, config_path


def _project_root(config_path: Path | None) -> Path | None:
    """Return the directory that relative configuration paths are relative to.

    Paths in the YAML are written from the project's point of view (``models/``,
    ``../output``), not from whatever directory the command happens to run in,
    so they are anchored to the nearest enclosing project (the directory holding
    ``pyproject.toml``) and fall back to the configuration's own directory.
    """

    if config_path is None:
        return None
    directory = config_path.parent
    for candidate in [directory, *directory.parents][:5]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return directory


def _config_path(value: Any, base: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or base is None:
        return path.resolve()
    return (base / path).resolve()


def _at(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _switch(value: Any, default: str = "on") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).lower()
    if text not in {"auto", "on", "off"}:
        raise PipelineError(f"Expected auto/on/off (or boolean), got {value!r}.")
    return text


def _speaker_names(
    speaker_map: str | Path | None,
    inline: Sequence[str] = (),
    *,
    known: Sequence[str] | None = None,
) -> dict[str, str]:
    names: dict[str, str] = {}
    if speaker_map:
        value, _ = _load_yaml(speaker_map)
        if not all(isinstance(key, str) and isinstance(name, str) for key, name in value.items()):
            raise PipelineError("Speaker map must contain LABEL: NAME string pairs.")
        names.update(value)
    for item in inline:
        if "=" not in item:
            raise PipelineError(f"Invalid speaker name '{item}'; expected LABEL=NAME.")
        label, name = (part.strip() for part in item.split("=", 1))
        if not label or not name:
            raise PipelineError(f"Invalid speaker name '{item}'.")
        names[label] = name
    if known is not None:
        # A mistyped label renames nobody; without this the rename looks applied.
        unused = sorted(set(names) - set(known))
        if unused:
            print(
                "warning: these labels are not in the transcript and were ignored: "
                + ", ".join(unused)
                + ". Available labels: "
                + (", ".join(known) or "none"),
                file=sys.stderr,
                flush=True,
            )
    return names


def _formats(argument: str | None, config: Mapping[str, Any]) -> list[str]:
    configured = _at(config, "output", "formats", default=["md", "txt", "srt", "vtt", "json"])
    values = argument.split(",") if argument else configured
    if not isinstance(values, list):
        raise PipelineError("output.formats must be a YAML list.")
    result = [str(value).strip().lower().lstrip(".") for value in values if str(value).strip()]
    invalid = sorted(set(result) - {"md", "txt", "srt", "vtt", "json"})
    if invalid:
        raise PipelineError("Unsupported format(s): " + ", ".join(invalid))
    if not result:
        raise PipelineError("At least one output format is required.")
    return result


def _resolved_run_options(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    model = str(_at(config, "diarization", "model", default=MODEL_ID))
    revision = str(_at(config, "diarization", "revision", default=MODEL_REVISION))
    if model != MODEL_ID or revision != MODEL_REVISION:
        raise PipelineError(
            f"This release is pinned to {MODEL_ID}@{MODEL_REVISION}; "
            "the configuration requests an unreviewed artifact."
        )
    num_speakers = (
        args.num_speakers
        if args.num_speakers is not None
        else int(_at(config, "diarization", "num_speakers", default=DEFAULT_NUM_SPEAKERS))
    )
    max_turn_gap = (
        args.max_turn_gap
        if args.max_turn_gap is not None
        else float(_at(config, "merge", "max_turn_gap", default=1.5))
    )
    nearest_max_gap = (
        args.nearest_max_gap
        if args.nearest_max_gap is not None
        else float(_at(config, "merge", "nearest_max_gap", default=0.5))
    )
    if num_speakers < 1 or max_turn_gap < 0 or nearest_max_gap < 0:
        raise PipelineError("Speaker count must be positive and gap values non-negative.")
    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise PipelineError("PyTorch thread counts must be positive.")
    if getattr(args, "encoder_threads", None) is not None and args.encoder_threads < 1:
        raise PipelineError("--encoder-threads must be positive.")
    asr_filter = _at(config, "audio", "asr_filter", default=DEFAULT_ASR_FILTER)
    diarization_filter = _at(config, "audio", "diarization_filter")
    return {
        "num_speakers": num_speakers,
        "asr_filter": str(asr_filter) if asr_filter else None,
        "diarization_filter": str(diarization_filter) if diarization_filter else None,
        "model_variant": args.model_variant
        or str(_at(config, "asr", "model_variant", default="rnnt")),
        "punctuation": args.punctuation
        or _switch(_at(config, "asr", "punctuation", default=True)),
        "itn": args.itn or _switch(_at(config, "asr", "itn", default=True)),
        "vad": False
        if args.no_vad
        else bool(_at(config, "asr", "vad", default=True)),
        "device": args.device
        or str(_at(config, "diarization", "device", default="cpu")),
        "max_turn_gap": max_turn_gap,
        "nearest_max_gap": nearest_max_gap,
        "torch_threads": args.torch_threads,
        "torch_interop_threads": args.torch_interop_threads,
    }


def _model_directories(
    args: argparse.Namespace, config: Mapping[str, Any], base: Path | None
) -> tuple[Path | None, Path | None]:
    configured_pyannote = _at(config, "diarization", "model_dir")
    pyannote = _config_path(configured_pyannote, base) if configured_pyannote else None
    if args.model_dir:
        # Wrapper/API contract: --model-dir is GigaSTT's exact model directory.
        # It comes from the command line, so it is relative to the caller's cwd.
        return Path(args.model_dir).expanduser().resolve(), pyannote
    giga = _at(config, "asr", "model_dir")
    return (_config_path(giga, base) if giga else None), pyannote


def run_pipeline(args: argparse.Namespace) -> Path:
    config, config_path = _load_config(args.config)
    options = _resolved_run_options(args, config)
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"Input audio does not exist: {source}")
    fingerprint = _fingerprint(source)
    base = _project_root(config_path)
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = _config_path(
            _at(config, "project", "output_root", default="output"), base
        )
    safe_stem = re.sub(r"[^\w.-]+", "_", source.stem, flags=re.UNICODE).strip("._") or "audio"
    output_dir = output_root / f"{safe_stem}-{fingerprint['sha256'][:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = None
    if manifest_path.is_file() and not args.force:
        previous = _load_json(manifest_path)
        old_input = previous.get("input", {})
        if any(old_input.get(key) != fingerprint[key] for key in ("name", "size_bytes", "sha256")):
            raise PipelineError("Existing job belongs to a different input; use --force.")

    inference_config = {
        **options,
        "allow_downmix": bool(args.allow_downmix),
        "pyannote_model": MODEL_ID,
        "pyannote_revision": MODEL_REVISION,
    }
    # Thread counts change speed, never the transcript; keeping them out of the
    # comparison means retuning them does not force a full rebuild.
    for key in ("torch_threads", "torch_interop_threads"):
        inference_config.pop(key, None)
    if previous and previous.get("config") != inference_config:
        raise PipelineError("Inference/merge options changed; use --force to rebuild.")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": __version__,
        "created_at": (previous or {}).get("created_at", _now()),
        "updated_at": _now(),
        "input": fingerprint,
        "config": inference_config,
        "stages": {} if args.force else dict((previous or {}).get("stages", {})),
    }

    def mark(stage: str, detail: Mapping[str, Any]) -> None:
        manifest["stages"][stage] = {"completed_at": _now(), **detail}
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)

    giga_model_dir, pyannote_cache_dir = _model_directories(args, config, base)
    print("[1/5] Preparing separate ASR and diarization audio…", flush=True)
    prepared = prepare_audio(
        source,
        output_dir,
        allow_downmix=args.allow_downmix,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        overwrite=args.force or previous is None,
        asr_filter=options["asr_filter"],
        diarization_filter=options["diarization_filter"],
    )
    mark(
        "audio",
        {
            "asr": "audio/asr.wav",
            "diarization": "audio/diarization.wav",
            "source_info": prepared.source_info.to_dict(),
        },
    )

    intermediate = output_dir / "intermediate"
    asr_path = intermediate / "gigastt.json"
    if asr_path.is_file() and not args.force:
        print("[2/5] Reusing GigaSTT timestamps.", flush=True)
        transcript = load_transcript(asr_path)
    else:
        print("[2/5] Running GigaSTT RNNT…", flush=True)
        transcript = transcribe(
            prepared.asr_path,
            asr_path,
            executable=args.gigastt_exe,
            model_variant=options["model_variant"],
            model_dir=giga_model_dir,
            punctuation=options["punctuation"],
            itn=options["itn"],
            vad=options["vad"],
            encoder_threads=args.encoder_threads,
        )
    mark(
        "asr",
        {
            "result": "intermediate/gigastt.json",
            "words": len(transcript["words"]),
            "duration": float(
                transcript.get("duration", transcript.get("duration_s", 0.0))
            ),
        },
    )

    diarization_path = intermediate / "pyannote.json"
    reused_diarization = False
    if diarization_path.is_file() and not args.force:
        print("[3/5] Reusing pyannote diarization.", flush=True)
        diarization_result = load_diarization(diarization_path)
        reused_diarization = True
    else:
        print(
            f"[3/5] Running pinned Community-1 ({options['num_speakers']} speakers)…",
            flush=True,
        )
        diarization_result = diarize(
            prepared.diarization_path,
            diarization_path,
            num_speakers=options["num_speakers"],
            device=options["device"],
            cache_dir=pyannote_cache_dir,
            torch_threads=options["torch_threads"],
            torch_interop_threads=options["torch_interop_threads"],
            strict_speakers=args.strict_speakers,
        )
    labels = sorted(
        {segment["speaker"] for segment in diarization_result["exclusive_segments"]}
    )
    if reused_diarization and len(labels) != options["num_speakers"]:
        # diarize() already reported this for a fresh run; a resumed run must not
        # silently accept a mismatch the first run refused.
        message = (
            f"Requested {options['num_speakers']} speakers but the reused diarization "
            f"has {len(labels)} ({', '.join(labels) or 'none'})."
        )
        if args.strict_speakers:
            raise PipelineError(message)
        print(f"warning: {message}", file=sys.stderr, flush=True)
    mark(
        "diarization",
        {
            "result": "intermediate/pyannote.json",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "speakers": labels,
            "requested_num_speakers": options["num_speakers"],
            "speaker_count_matches_request": len(labels) == options["num_speakers"],
        },
    )

    merged_path = intermediate / "merged.json"
    if merged_path.is_file() and not args.force:
        print("[4/5] Reusing merged transcript.", flush=True)
        merged = _load_json(merged_path)
    else:
        print("[4/5] Assigning words by exclusive maximum overlap…", flush=True)
        merged = merge_transcript(
            transcript,
            diarization_result,
            max_turn_gap=options["max_turn_gap"],
            nearest_max_gap=options["nearest_max_gap"],
        )
        _atomic_json(merged_path, merged)
    mark(
        "merge",
        {
            "result": "intermediate/merged.json",
            "turns": len(merged["turns"]),
            "words": len(merged["words"]),
        },
    )

    names = _speaker_names(args.speaker_map, args.speaker_name, known=merged["speakers"])
    formats = _formats(args.formats, config)
    mark_uncertain = bool(_at(config, "output", "mark_uncertain_words", default=True))
    print("[5/5] Rendering " + ", ".join(formats) + "…", flush=True)
    paths = write_outputs(
        merged,
        output_dir,
        stem=args.output_stem,
        formats=formats,
        names=names,
        mark_uncertain=mark_uncertain,
    )
    mark(
        "render",
        {
            "files": [path.relative_to(output_dir).as_posix() for path in paths],
            "speaker_names": names,
        },
    )
    print(f"Done: {output_dir}", flush=True)
    return output_dir


def _render_job(args: argparse.Namespace) -> Path:
    job = Path(args.job_dir).expanduser().resolve()
    merged_path = job / "intermediate" / "merged.json"
    merged = _load_json(merged_path)
    manifest_path = job / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    names = _speaker_names(
        args.speaker_map, args.speaker_name, known=merged.get("speakers", [])
    )
    previous_formats = ",".join(
        Path(item).suffix.lstrip(".")
        for item in manifest.get("stages", {}).get("render", {}).get("files", [])
    )
    formats = _formats(args.formats or previous_formats or None, {})
    paths = write_outputs(
        merged,
        job,
        stem=args.output_stem,
        formats=formats,
        names=names,
        mark_uncertain=args.mark_uncertain,
    )
    if manifest:
        manifest.setdefault("stages", {})["render"] = {
            "completed_at": _now(),
            "files": [path.relative_to(job).as_posix() for path in paths],
            "speaker_names": names,
        }
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)
    print(f"Rendered without inference: {job}")
    return job


def _doctor(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    # Validate the pin and relevant scalar configuration.
    probe_args = argparse.Namespace(
        num_speakers=None,
        max_turn_gap=None,
        nearest_max_gap=None,
        model_variant=None,
        punctuation=None,
        itn=None,
        no_vad=False,
        device=None,
        torch_threads=16,
        torch_interop_threads=1,
    )
    _resolved_run_options(probe_args, config)
    checks: list[tuple[str, bool, str]] = []
    for executable in ("ffmpeg", "ffprobe"):
        location = shutil.which(executable)
        checks.append((executable, bool(location), location or "not found"))
    giga = Path(args.gigastt_exe).expanduser()
    giga_location = str(giga.resolve()) if giga.is_file() else shutil.which(args.gigastt_exe)
    checks.append(("gigastt", bool(giga_location), giga_location or "not found"))
    for distribution in ("pyannote.audio", "torch", "soundfile", "PyYAML"):
        try:
            version = importlib.metadata.version(distribution)
            checks.append((distribution, True, version))
        except importlib.metadata.PackageNotFoundError:
            checks.append((distribution, False, "not installed"))
    token_ok = bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
        not in (None, "", "hf_REPLACE_WITH_YOUR_READ_TOKEN")
    )
    checks.append(("HF_TOKEN", token_ok, "set (hidden)" if token_ok else "not set"))
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 2


def _preload(args: argparse.Namespace) -> None:
    preload_model(
        model=args.model,
        revision=args.revision,
        cache_dir=Path(args.model_dir).expanduser().resolve() if args.model_dir else None,
    )
    print(f"Ready: {MODEL_ID}@{MODEL_REVISION}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fourvoices")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the complete resumable pipeline")
    run.add_argument("--input", required=True)
    run.add_argument("--output-root")
    run.add_argument("--config")
    run.add_argument("--gigastt-exe", "--gigastt", default="gigastt")
    run.add_argument("--model-dir", help="GigaSTT model directory")
    run.add_argument("--num-speakers", type=int)
    run.add_argument("--speaker-map")
    run.add_argument("--speaker-name", action="append", default=[], metavar="LABEL=NAME")
    run.add_argument("--allow-downmix", action="store_true")
    run.add_argument(
        "--strict-speakers",
        action="store_true",
        help="Fail when diarization finds a different speaker count (result is still saved)",
    )
    run.add_argument("--force", action="store_true")
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--ffprobe", default="ffprobe")
    run.add_argument("--model-variant")
    run.add_argument("--punctuation", choices=("auto", "on", "off"))
    run.add_argument("--itn", choices=("auto", "on", "off"))
    run.add_argument("--no-vad", action="store_true")
    run.add_argument("--encoder-threads", type=int)
    run.add_argument("--torch-threads", type=int, default=16)
    run.add_argument("--torch-interop-threads", type=int, default=1)
    run.add_argument("--device")
    run.add_argument("--max-turn-gap", type=float)
    run.add_argument("--nearest-max-gap", type=float)
    run.add_argument("--formats")
    run.add_argument("--output-stem", default="transcript")

    preload = commands.add_parser("diarize-preload", help="Preload the gated pinned model")
    preload.add_argument("--model", default=MODEL_ID)
    preload.add_argument("--revision", default=MODEL_REVISION)
    preload.add_argument("--model-dir")

    doctor = commands.add_parser("doctor", help="Check the local runtime without inference")
    doctor.add_argument("--config")
    doctor.add_argument("--gigastt-exe", default="gigastt")
    doctor.add_argument("--model-dir")  # accepted for wrapper/API stability

    render = commands.add_parser("render", help="Rerender merged JSON without inference")
    render.add_argument("--job-dir", required=True)
    render.add_argument("--speaker-map")
    render.add_argument("--speaker-name", action="append", default=[], metavar="LABEL=NAME")
    render.add_argument("--formats")
    render.add_argument("--output-stem", default="transcript")
    render.add_argument(
        "--no-mark-uncertain",
        dest="mark_uncertain",
        action="store_false",
        help="Do not mark turns whose speaker attribution is unconfirmed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            run_pipeline(args)
            return 0
        if args.command == "render":
            _render_job(args)
            return 0
        if args.command == "diarize-preload":
            _preload(args)
            return 0
        if args.command == "doctor":
            return _doctor(args)
        raise PipelineError(f"Unknown command: {args.command}")
    except (PipelineError, AudioPreparationError, GigaSTTError, DiarizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed stages can be resumed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
