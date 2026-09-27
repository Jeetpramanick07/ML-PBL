"""Config loading helpers built on OmegaConf."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(path: Path | str = BASE_CONFIG_PATH) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig."""
    return OmegaConf.load(path)


def load_experiment_config(path: Path | str) -> DictConfig:
    """Load configs/base.yaml, then merge an experiment-specific config on
    top of it (e.g. configs/lora_generic.yaml), so experiment configs only
    need to declare the keys they add or override."""
    base = load_config()
    override = OmegaConf.load(path)
    return OmegaConf.merge(base, override)
