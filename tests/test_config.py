from pathlib import Path

import pytest

from slicegaussian.pipeline import Config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_yaml_matches_paper_json():
    assert Config.load(CONFIGS / "config.yml") == Config.load(CONFIGS / "paper.json")


def test_yaml_exponent_without_decimal_point(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("p2: {lr_log_scale: 1e-3}\n")
    assert Config.load(f).p2.lr_log_scale == 1e-3


def test_unknown_setting(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("p1: {tau: 0.1}\n")
    with pytest.raises(KeyError):
        Config.load(f)
