from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coldsense.app.api import ColdSenseHTTPServer
from coldsense.app.service import ColdSenseService, copilot_enabled_from_env
from coldsense.rag_pipeline import load_dotenv_file


def main() -> None:
    load_dotenv_file(ROOT_DIR / ".env")
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    copilot_enabled = copilot_enabled_from_env()          # default OFF; server environment only

    service = ColdSenseService(copilot_enabled=copilot_enabled, semantic_search_available=True)
    server = ColdSenseHTTPServer(host=host, port=port, service=service)
    httpd = server.create_server()
    print(f"ColdSense Full-Stack Server running at http://{host}:{port}")
    print(f"Copilot: {'ENABLED' if copilot_enabled else 'DISABLED'} (COLDSENSE_COPILOT_ENABLED, default off)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ColdSense server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
