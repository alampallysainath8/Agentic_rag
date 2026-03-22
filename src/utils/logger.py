"""Structured logging helpers for Agentic RAG.

Wraps stdlib logging + optional structlog to produce JSON structured logs.
Centralises configuration so all modules call setup_logger(__name__).

Usage
-----
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("message", key=value)   # with structlog
    logger.info("message")              # stdlib fallback
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

try:
    import structlog as _structlog
except ImportError:          # pragma: no cover
    _structlog = None        # type: ignore


class CustomLogger:
    """Configure logging exactly once per process.

    Writes JSON structured lines to console and to a timestamped file
    under `<cwd>/logs/` (or a custom directory).
    """

    configured: bool = False
    log_file_path: str | None = None

    def __init__(self, log_dir: str = "logs", enable_file_logging: bool = True):
        self.enable_file_logging = enable_file_logging

        if not CustomLogger.configured:
            if self.enable_file_logging:
                logs_dir = os.path.join(os.getcwd(), log_dir)
                os.makedirs(logs_dir, exist_ok=True)

                log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
                CustomLogger.log_file_path = os.path.join(logs_dir, log_file)

            self._configure_logging()
            CustomLogger.configured = True

        self.log_file_path = CustomLogger.log_file_path

    # ── internal ──────────────────────────────────────────────────────────────

    def _configure_logging(self) -> None:
        """Attach console + optional file handler; configure structlog."""
        handlers: list[logging.Handler] = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console)

        if self.enable_file_logging and CustomLogger.log_file_path:
            fh = logging.FileHandler(CustomLogger.log_file_path, mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            handlers.append(fh)

        root = logging.getLogger()
        root.handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=handlers,
            force=True,
        )

        if _structlog is not None:
            _structlog.configure(
                processors=[
                    _structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    _structlog.processors.add_log_level,
                    _structlog.processors.EventRenamer(to="event"),
                    _structlog.processors.JSONRenderer(),
                ],
                logger_factory=_structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

    # ── public ────────────────────────────────────────────────────────────────

    @staticmethod
    def get_logger(name: str):
        """Return a structlog logger (or stdlib fallback) for *name*."""
        logger_name = os.path.basename(name)
        if _structlog is None:
            return logging.getLogger(logger_name)
        return _structlog.get_logger(logger_name)


def setup_logger(
    name: str,
    log_dir: str = "logs",
    enable_file_logging: bool = True,
):
    """Return a configured logger for *name*.

    Calls ``CustomLogger`` internally (idempotent — safe to call multiple times).

    Parameters
    ----------
    name:
        Logger name, usually module ``__name__``.
    log_dir:
        Sub-directory under ``cwd`` where the log file is written.
    enable_file_logging:
        Set to ``False`` to disable file output (useful in tests).
    """
    cl = CustomLogger(log_dir=log_dir, enable_file_logging=enable_file_logging)
    return cl.get_logger(name)
