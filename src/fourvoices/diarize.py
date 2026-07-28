"""Four-speaker diarization with pyannote Community-1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

# Evidence processing is local by design. Set these before importing HF/pyannote.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

MODEL_ID = "pyannote/speaker-diarization-community-1"
# Pin the model artifact so a later upstream update cannot silently change evidence.
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
DEFAULT_NUM_SPEAKERS = 4


class DiarizationError(RuntimeError):
    """Diarization failed or produced an unusable annotation."""


def _segments(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None:
        return []
    result: list[dict[str, Any]] = []
    try:
        tracks: Iterable[tuple[Any, Any, Any]] = annotation.itertracks(yield_label=True)
        for segment, _track, label in tracks:
            start, end = float(segment.start), float(segment.end)
            if end > start:
                result.append({"start": start, "end": end, "speaker": str(label)})
    except (AttributeError, TypeError, ValueError) as exc:
        raise DiarizationError("pyannote returned an incompatible annotation.") from exc
    return sorted(result, key=lambda x: (x["start"], x["end"], x["speaker"]))


def _load_waveform(audio_path: Path) -> tuple[Any, int, float]:
    """Load audio without torchaudio/torchcodec, using soundfile and a tensor."""

    try:
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise DiarizationError(
            "Diarization dependencies are missing. Install the project's diarization extras."
        ) from exc
    try:
        samples, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise DiarizationError(f"soundfile could not read {audio_path.name}: {exc}") from exc
    if samples.shape[0] == 0:
        raise DiarizationError(f"Audio is empty: {audio_path.name}")
    if samples.shape[1] != 1:
        raise DiarizationError(
            f"Prepared diarization audio must be mono, got {samples.shape[1]} channels."
        )
    waveform = torch.from_numpy(samples[:, 0].copy()).unsqueeze(0)
    return waveform, int(sample_rate), samples.shape[0] / float(sample_rate)


def diarize(
    audio_path: str | Path,
    output_json: str | Path,
    *,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
    token: str | None = None,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    torch_threads: int = 16,
    torch_interop_threads: int = 1,
) -> dict[str, Any]:
    """Run the revision-pinned pipeline and save regular + exclusive annotations."""

    if num_speakers < 1:
        raise ValueError("num_speakers must be positive")
    source = Path(audio_path)
    destination = Path(output_json)
    if not source.is_file():
        raise DiarizationError(f"Audio file does not exist: {source}")
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "Set HF_TOKEN after accepting the Community-1 model conditions on Hugging Face."
        )
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio 4.x and PyTorch are required for diarization."
        ) from exc

    if torch_threads < 1 or torch_interop_threads < 1:
        raise ValueError("PyTorch thread counts must be positive")
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        # PyTorch permits setting this only before its first parallel operation.
        pass
    waveform, sample_rate, duration = _load_waveform(source)
    try:
        pipeline = Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        if pipeline is None:
            raise DiarizationError(
                "Could not load Community-1. Confirm model access and HF_TOKEN."
            )
        pipeline.to(torch.device(device))
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )
    except DiarizationError:
        raise
    except Exception as exc:
        raise DiarizationError(f"pyannote diarization failed: {exc}") from exc

    regular_annotation = getattr(output, "speaker_diarization", output)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive_annotation is None:
        raise DiarizationError(
            "Community-1 did not return exclusive_speaker_diarization; "
            "verify that pyannote.audio 4.x is installed."
        )
    payload = {
        "schema_version": 1,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "num_speakers": num_speakers,
        "duration_s": duration,
        "segments": _segments(regular_annotation),
        "exclusive_segments": _segments(exclusive_annotation),
    }
    found = {item["speaker"] for item in payload["exclusive_segments"]}
    if len(found) != num_speakers:
        raise DiarizationError(
            f"Requested {num_speakers} speakers but pyannote returned {len(found)}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def preload_model(
    *,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    model: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> None:
    """Download the gated, revision-pinned pipeline and all referenced weights."""

    if model != MODEL_ID or revision != MODEL_REVISION:
        raise DiarizationError(
            f"This release is pinned to {MODEL_ID}@{MODEL_REVISION}; "
            "refusing an unreviewed model revision."
        )
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "Set HF_TOKEN after accepting the Community-1 model conditions on Hugging Face."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError("pyannote.audio 4.x is not installed.") from exc
    try:
        pipeline = Pipeline.from_pretrained(
            model,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise DiarizationError(f"Could not preload Community-1: {exc}") from exc
    if pipeline is None:
        raise DiarizationError(
            "Model download was denied. Accept its conditions and verify HF_TOKEN."
        )


def load_diarization(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiarizationError(f"Cannot read diarization JSON: {path}") from exc
    for key in ("segments", "exclusive_segments"):
        if not isinstance(payload.get(key), list):
            raise DiarizationError(f"Diarization JSON has no '{key}' array.")
    return payload
