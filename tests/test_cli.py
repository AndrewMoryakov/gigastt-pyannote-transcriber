import json
from pathlib import Path

import pytest

from fourvoices import cli

REPO = Path(__file__).resolve().parent.parent


def write_config(path, body):
    path.write_text(body, encoding="utf-8")
    return path


def test_unknown_keys_are_rejected_with_a_useful_message(tmp_path):
    config = write_config(
        tmp_path / "c.yaml",
        "asr:\n  model_varient: rnnt\nlogging:\n  level: debug\n",
    )
    with pytest.raises(cli.PipelineError) as error:
        cli._load_config(config)

    message = str(error.value)
    assert "unknown key 'asr.model_varient'" in message
    assert "unknown section 'logging'" in message
    assert "asr.model_variant" in message  # the supported spelling is offered


def test_shipped_default_config_passes_validation():
    config, path = cli._load_config(REPO / "config" / "default.yaml")
    assert config["diarization"]["num_speakers"] == 4
    assert path is not None


def test_empty_section_is_allowed(tmp_path):
    config, _ = cli._load_config(write_config(tmp_path / "c.yaml", "audio:\n"))
    assert config == {"audio": None}


def test_section_must_be_a_mapping(tmp_path):
    with pytest.raises(cli.PipelineError, match="must be a mapping"):
        cli._load_config(write_config(tmp_path / "c.yaml", "audio: [1, 2]\n"))


def test_relative_paths_follow_the_project_root_not_the_cwd(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    config = write_config(
        project / "config" / "default.yaml", "asr:\n  model_dir: models/gigastt\n"
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    _, config_path = cli._load_config(config)
    base = cli._project_root(config_path)

    assert base == project.resolve()
    expected = (project / "models" / "gigastt").resolve()
    assert cli._config_path("models/gigastt", base) == expected
    assert cli._config_path("../out", base) == (tmp_path / "out").resolve()


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = (tmp_path / "models").resolve()
    assert cli._config_path(str(absolute), tmp_path / "other") == absolute


def test_project_root_falls_back_to_the_config_directory(tmp_path):
    loose = tmp_path / "standalone"
    loose.mkdir()
    config = write_config(loose / "c.yaml", "asr:\n  model_dir: models\n")
    assert cli._project_root(config) == loose.resolve()


def test_speaker_map_typos_are_reported(tmp_path, capsys):
    names = cli._speaker_names(
        None, ["SPEAKER_1=Отец"], known=["SPEAKER_00", "SPEAKER_01"]
    )
    assert names == {"SPEAKER_1": "Отец"}
    error = capsys.readouterr().err
    assert "SPEAKER_1" in error
    assert "SPEAKER_00, SPEAKER_01" in error


def test_matching_speaker_labels_are_silent(capsys):
    cli._speaker_names(None, ["SPEAKER_00=Отец"], known=["SPEAKER_00"])
    assert capsys.readouterr().err == ""


def test_speaker_map_yaml_is_not_validated_against_the_config_schema(tmp_path):
    speaker_map = write_config(tmp_path / "map.yaml", "SPEAKER_00: Отец\n")
    assert cli._speaker_names(speaker_map) == {"SPEAKER_00": "Отец"}


@pytest.mark.parametrize(
    "argument, expected",
    [
        (None, ["md", "txt", "srt", "vtt", "json"]),
        ("md,SRT", ["md", "srt"]),
        (".vtt", ["vtt"]),
    ],
)
def test_formats_parsing(argument, expected):
    assert cli._formats(argument, {}) == expected


def test_unsupported_format_is_rejected():
    with pytest.raises(cli.PipelineError, match="docx"):
        cli._formats("md,docx", {})


# ---------------------------------------------------------------- run_pipeline

TRANSCRIPT = {
    "duration": 2.0,
    "text": "первый второй",
    "words": [
        {"word": "первый", "start": 0.1, "end": 0.5, "confidence": 0.9},
        {"word": "второй", "start": 1.1, "end": 1.6, "confidence": 0.8},
    ],
}
DIARIZATION = {
    "exclusive_segments": [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ],
    "segments": [],
}


class Recorder:
    """Counts how often each expensive stage actually ran."""

    def __init__(self):
        self.audio = 0
        self.asr = 0
        self.diarization = 0
        self.audio_overwrite = []


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from fourvoices.audio import AudioInfo, PreparedAudio

    calls = Recorder()
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    config = project / "config" / "default.yaml"
    config.write_text("project:\n  output_root: ../out\n", encoding="utf-8")
    source = tmp_path / "запись.m4a"
    source.write_bytes(b"audio-bytes")

    def fake_prepare_audio(src, work_dir, **kwargs):
        calls.audio += 1
        calls.audio_overwrite.append(kwargs["overwrite"])
        directory = Path(work_dir) / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("asr.wav", "diarization.wav"):
            (directory / name).write_bytes(b"RIFF")
        return PreparedAudio(
            directory / "asr.wav",
            directory / "diarization.wav",
            AudioInfo(codec="aac", channels=1, sample_rate=16000, duration=2.0),
        )

    def fake_transcribe(audio_path, output_json, **kwargs):
        calls.asr += 1
        cli._atomic_json(Path(output_json), TRANSCRIPT)
        return dict(TRANSCRIPT)

    def fake_diarize(audio_path, output_json, **kwargs):
        calls.diarization += 1
        cli._atomic_json(Path(output_json), DIARIZATION)
        return dict(DIARIZATION)

    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "diarize", fake_diarize)
    monkeypatch.chdir(tmp_path)
    return calls, source, config, project


def run(source, config, *extra):
    args = cli.build_parser().parse_args(
        ["run", "--input", str(source), "--config", str(config), *extra]
    )
    return cli.run_pipeline(args)


def test_run_produces_a_job_next_to_the_project_root(pipeline):
    calls, source, config, project = pipeline
    job = run(source, config)

    assert job.parent == (project.parent / "out").resolve()
    assert (job / "transcript.md").is_file()
    assert (job / "manifest.json").is_file()
    assert calls.asr == 1


def test_second_run_reuses_every_expensive_stage(pipeline):
    calls, source, config, _ = pipeline
    run(source, config)
    run(source, config)

    assert (calls.asr, calls.diarization) == (1, 1)
    assert calls.audio_overwrite == [True, False]


def test_force_reruns_everything(pipeline):
    calls, source, config, _ = pipeline
    run(source, config)
    run(source, config, "--force")

    assert (calls.asr, calls.diarization) == (2, 2)
    assert calls.audio_overwrite == [True, True]


def test_changed_inference_option_is_refused_without_force(pipeline):
    _, source, config, _ = pipeline
    run(source, config)

    with pytest.raises(cli.PipelineError, match="use --force"):
        run(source, config, "--num-speakers", "3")


def test_changed_audio_filter_is_refused_without_force(pipeline):
    _, source, config, _ = pipeline
    run(source, config)
    config.write_text(
        "project:\n  output_root: ../out\naudio:\n  asr_filter: highpass=f=200\n",
        encoding="utf-8",
    )

    with pytest.raises(cli.PipelineError, match="use --force"):
        run(source, config)


def test_thread_counts_do_not_invalidate_a_job(pipeline):
    calls, source, config, _ = pipeline
    run(source, config)
    run(source, config, "--torch-threads", "4")

    assert calls.asr == 1


def test_a_lost_manifest_does_not_let_stale_stages_through(pipeline):
    calls, source, config, _ = pipeline
    job = run(source, config)
    (job / "manifest.json").unlink()

    run(source, config, "--num-speakers", "3")

    # Without the manifest there is nothing vouching for the cached JSON, so the
    # stages must be recomputed rather than silently mixed with new options.
    assert (calls.asr, calls.diarization) == (2, 2)
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["num_speakers"] == 3


def test_a_different_input_in_the_same_job_directory_is_refused(pipeline, monkeypatch):
    _, source, config, _ = pipeline
    job = run(source, config)
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    manifest["input"]["sha256"] = "0" * 64
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(cli.PipelineError, match="different input"):
        run(source, config)


def test_speaker_map_renames_turns_on_rerender(pipeline):
    _, source, config, _ = pipeline
    job = run(source, config)
    speaker_map = job / "speaker-map.yaml"
    speaker_map.write_text("SPEAKER_00: Отец\nSPEAKER_01: Мать\n", encoding="utf-8")

    args = cli.build_parser().parse_args(
        ["render", "--job-dir", str(job), "--speaker-map", str(speaker_map)]
    )
    cli._render_job(args)

    text = (job / "transcript.txt").read_text(encoding="utf-8")
    assert "Отец" in text and "SPEAKER_00" not in text


def test_rerender_keeps_the_formats_recorded_in_the_manifest(pipeline):
    _, source, config, _ = pipeline
    job = run(source, config, "--formats", "txt,srt")

    args = cli.build_parser().parse_args(["render", "--job-dir", str(job)])
    cli._render_job(args)

    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["stages"]["render"]["files"]) == [
        "transcript.srt",
        "transcript.txt",
    ]


def test_a_stage_missing_from_the_manifest_is_recomputed(pipeline):
    calls, source, config, _ = pipeline
    job = run(source, config)
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    # An interrupted or older run can leave the product without its manifest entry.
    del manifest["stages"]["asr"]
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    run(source, config)

    assert calls.asr == 2
    assert calls.diarization == 1


def test_rerender_accepts_an_explicit_format_list(pipeline):
    _, source, config, _ = pipeline
    job = run(source, config)

    args = cli.build_parser().parse_args(
        ["render", "--job-dir", str(job), "--formats", "md", "--output-stem", "итог"]
    )
    cli._render_job(args)

    assert (job / "итог.md").is_file()
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["render"]["files"] == ["итог.md"]


def test_a_non_numeric_subtitle_limit_is_a_pipeline_error(tmp_path):
    config = write_config(
        tmp_path / "c.yaml", "output:\n  subtitle_max_seconds: скоро\n"
    )
    loaded, _ = cli._load_config(config)
    with pytest.raises(cli.PipelineError, match="subtitle_max_seconds"):
        cli._subtitle_limits(loaded)


def test_thread_default_follows_the_machine_not_a_hardcoded_number():
    import os

    args = cli.build_parser().parse_args(["run", "--input", "x"])
    assert args.torch_threads == (os.cpu_count() or 1)
    assert args.torch_threads >= 1
    assert args.torch_interop_threads == 1


def test_an_explicit_thread_count_still_wins():
    args = cli.build_parser().parse_args(["run", "--input", "x", "--torch-threads", "4"])
    assert args.torch_threads == 4
