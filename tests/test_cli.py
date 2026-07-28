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
    config = write_config(project / "config" / "default.yaml", "asr:\n  model_dir: models/gigastt\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    _, config_path = cli._load_config(config)
    base = cli._project_root(config_path)

    assert base == project.resolve()
    assert cli._config_path("models/gigastt", base) == (project / "models" / "gigastt").resolve()
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
