"""Structured logging setup with loguru and optional W&B integration."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Configure structured logging with loguru and optionally initialize W&B.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        wandb_enabled: Whether to initialize W&B logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/username.
        wandb_run_name: W&B run name.
    """
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, format=log_format, level=level, rotation="10 MB")

    if wandb_enabled:
        try:
            import wandb

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                reinit=True,
            )
            logger.info(f"W&B initialized: project={wandb_project}, entity={wandb_entity}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B initialization")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")


def get_logger(name: str = "multiguard") -> "logger":
    """Get a contextualized logger instance."""
    return logger.bind(module=name)
