import pytest

from signal_resilience import config


def test_config_toml_parses():
    cfg = config.load()
    assert cfg.start <= cfg.analysis_end <= cfg.end
    assert 0 in cfg.lags and 1 in cfg.rebalance_periods
    assert (0, 1) in cfg.settings


def test_unknown_key_raises(tmp_path):
    text = config.CONFIG_FILE.read_text().replace("[sweep]", "[sweep]\ntypo_key = 1", 1)
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(KeyError, match="typo_key"):
        config.load(path)


def test_reference_overrides_resolve_per_signal():
    cfg = config.load()
    assert cfg.reference["beta"] == ("ZeroBeta",)
    assert cfg.reference["reversal"] == ("ZeroInvestment", "ZeroBeta")


def test_unknown_reference_constraint_raises(tmp_path):
    text = config.CONFIG_FILE.read_text().replace('beta = ["ZeroBeta"]', 'beta = ["LongOnly"]')
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="LongOnly"):
        config.load(path)


def test_personal_config_parses():
    cfg = config.load(config.PROJECT_ROOT / "config.personal.toml")
    assert cfg.source == "personal" and cfg.alphas is None
    assert cfg.output == config.PROJECT_ROOT / "output" / "personal"
    assert cfg.end == cfg.analysis_end


def test_library_source_requires_alphas(tmp_path):
    text = config.CONFIG_FILE.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("alphas"))
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(KeyError, match="alphas"):
        config.load(path)
