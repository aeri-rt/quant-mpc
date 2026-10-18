"""Loguru with two sinks (human console, JSON-lines file). Importing this module adds no sinks
and creates no files; call `setup()` once from the entrypoint. Structured events are
`log.bind(**fields).info("event_name")`.
"""

import sys
from pathlib import Path

from loguru import logger

_CONSOLE = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {extra[component]} | {message}"
)


def setup(log_dir: Path | str | None, level: str = "INFO") -> None:
    """Console sink at `level`; the DEBUG JSONL file sink only when `log_dir` is given."""
    logger.remove()
    logger.configure(extra={"component": "-"})  # unbound loggers print "-"; bind() overrides
    logger.add(sys.stderr, level=level, format=_CONSOLE)
    if log_dir is None:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "quant-mpc_{time:YYYY-MM-DD}.jsonl",
        level="DEBUG",
        serialize=True,
        rotation="10 MB",
        retention="30 days",
        enqueue=False,
    )


def get_logger(component: str):
    return logger.bind(component=component)
