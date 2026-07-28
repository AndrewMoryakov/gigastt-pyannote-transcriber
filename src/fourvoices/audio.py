"""Audio inspection and conservative preparation through FFmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


class AudioPreparationError(RuntimeError):
    """Audio could not be safely inspected or converted."""


class StereoInputError(AudioPreparationError):
    """Input has multiple channels and needs human inspection before downmixing."""


@dataclass(frozen=True)
class AudioInfo:
    codec: str
    channels: int
    sample_rate: int
    duration: float
    channel_layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedAudio:
    asr_path: Path
    diarization_path: Path
    source_info: AudioInfo


def _run(command: Sequence[str], *, tool: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise AudioPreparationError(
            f"{tool} is not available on PATH. Install FFmpeg and restart the shell."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise AudioPreparationError(f"{tool} failed: {detail}") from exc


def probe_audio(path: str | Path, *, ffprobe: str = "ffprobe") -> AudioInfo:
    """Return metadata for the first audio stream.

    ``ffprobe`` is used instead of trusting the filename extension. Files with no
    audio stream, invalid channel counts, or unknown duration are rejected.
    """

    source = Path(path)
    if not source.is_file():
        raise AudioPreparationError(f"Audio file does not exist: {source}")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,channel_layout,sample_rate,duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    result = _run(command, tool="ffprobe")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
        info = AudioInfo(
            codec=str(stream.get("codec_name") or "unknown"),
            channels=int(stream["channels"]),
            sample_rate=int(stream["sample_rate"]),
            duration=float(duration_value),
            channel_layout=stream.get("channel_layout"),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioPreparationError(
            f"Could not read a valid audio stream from {source.name}."
        ) from exc
    if info.channels < 1 or info.sample_rate < 1 or info.duration <= 0:
        raise AudioPreparationError(f"Invalid audio metadata for {source.name}: {info}")
    return info


def require_mono(info: AudioInfo, *, allow_downmix: bool = False) -> None:
    """Reject multichannel audio unless downmixing was explicitly approved."""

    if info.channels > 1 and not allow_downmix:
        raise StereoInputError(
            f"Input has {info.channels} channels ({info.channel_layout or 'layout unknown'}). "
            "Inspect/listen to the channels first; re-run with --allow-downmix only if "
            "mixing them cannot destroy useful spatial speaker separation."
        )


def _convert(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str,
    audio_filter: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
    ]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(
        ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", str(temporary)]
    )
    try:
        _run(command, tool="ffmpeg")
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise AudioPreparationError(f"FFmpeg produced an empty file: {destination.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


DEFAULT_ASR_FILTER = "highpass=f=80,loudnorm=I=-18:LRA=11:TP=-2"


def prepare_audio(
    source: str | Path,
    work_dir: str | Path,
    *,
    allow_downmix: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
    asr_filter: str | None = DEFAULT_ASR_FILTER,
    diarization_filter: str | None = None,
) -> PreparedAudio:
    """Create separate 16 kHz mono WAVs for ASR and diarization.

    Diarization gets only resampling/downmixing by default. ASR additionally gets
    a gentle high-pass and loudness normalization. Existing valid products are
    reused. Both filters come from configuration so the audio actually used stays
    inspectable rather than hidden in this module.
    """

    source_path = Path(source).resolve()
    work = Path(work_dir)
    info = probe_audio(source_path, ffprobe=ffprobe)
    require_mono(info, allow_downmix=allow_downmix)
    asr_path = work / "audio" / "asr.wav"
    diar_path = work / "audio" / "diarization.wav"

    def valid_product(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            product = probe_audio(path, ffprobe=ffprobe)
        except AudioPreparationError:
            return False
        return product.channels == 1 and product.sample_rate == 16000

    if overwrite or not valid_product(diar_path):
        _convert(source_path, diar_path, ffmpeg=ffmpeg, audio_filter=diarization_filter)
    if overwrite or not valid_product(asr_path):
        _convert(source_path, asr_path, ffmpeg=ffmpeg, audio_filter=asr_filter)
    return PreparedAudio(asr_path, diar_path, info)


def check_ffmpeg(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
    """Fail early if either executable cannot be located."""

    missing = [name for name in (ffmpeg, ffprobe) if shutil.which(name) is None]
    if missing:
        raise AudioPreparationError("Missing executable(s) on PATH: " + ", ".join(missing))
