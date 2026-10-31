"""Write the API's OpenAPI document, the source of the frontend's generated TypeScript types.

    python -m coldsense.app.api.openapi_export web/openapi.json
    (cd web && npm run gen:types)

``tests/app/test_api_contract.py`` fails when the committed web/openapi.json differs from this output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import ResolvedApiConfig


def openapi_document() -> dict:
    from . import create_app

    cfg = ResolvedApiConfig(secret=b"x" * 32, cookie_secure=False, serve_docs=True, env="dev")
    return create_app(cfg).openapi()


def render() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = render()
    if args:
        Path(args[0]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
