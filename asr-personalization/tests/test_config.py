from src.utils.config import load_config


def test_base_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.model.name == "openai/whisper-small"
    assert cfg.data.sample_rate == 16000
