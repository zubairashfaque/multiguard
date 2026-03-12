"""Configuration loading and management using OmegaConf."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> DictConfig:
    """Load a YAML config file with OmegaConf, supporting variable interpolation and merging.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dictionary of overrides to merge.

    Returns:
        Merged DictConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Handle defaults (merge parent configs)
    if "defaults" in cfg:
        base_configs = []
        config_dir = config_path.parent
        for default in cfg.defaults:
            if default == "_self_":
                continue
            base_path = (config_dir / f"{default}.yaml").resolve()
            if base_path.exists():
                base_configs.append(OmegaConf.load(base_path))
        if base_configs:
            base = OmegaConf.merge(*base_configs)
            cfg = OmegaConf.merge(base, cfg)
        # Remove defaults key after processing
        if "defaults" in cfg:
            OmegaConf.update(cfg, "defaults", None)

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to plain dict."""
    return OmegaConf.to_container(cfg, resolve=True)
