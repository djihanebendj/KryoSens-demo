"""Logging setup. Named ``logs`` (not ``logging``) to avoid shadowing the stdlib.

Rules:
- Library code only does ``log = logging.getLogger(__name__)``; it never
  configures handlers. Only entry points call ``configure_logging``.
- Logs go to stderr so command output on stdout stays clean and parseable.
- Log decisions and their reasons (verdicts, refusals, evidence class), not
  raw data rows. Never log secrets.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from coldsense.settings import Settings, get_settings

_HANDLER_NAME = "coldsense"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Idempotent: calling twice replaces ColdSense's handler, never stacks it."""
    s = settings or get_settings()
    root = logging.getLogger("coldsense")
    root.setLevel(s.log_level)
    root.propagate = False
    for h in list(root.handlers):
        if h.get_name() == _HANDLER_NAME:
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(
        JsonFormatter() if s.log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    return root
