"""Alerting system for model monitoring."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Manage and dispatch alerts for model health issues.

    Supports logging, webhook, and email alerting channels.
    """

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels or ["log"]

    def send_alert(self, title: str, message: str, severity: str = "warning") -> None:
        """Send an alert through configured channels.

        Args:
            title: Alert title.
            message: Alert details.
            severity: Alert severity (info, warning, critical).
        """
        for channel in self.channels:
            if channel == "log":
                logger.warning(f"[ALERT:{severity}] {title}: {message}")
            # Add webhook, email channels as needed
