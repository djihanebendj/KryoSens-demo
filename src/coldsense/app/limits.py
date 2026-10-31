"""Transport and upload limits for the public demo.

Code constants, not environment variables: a limit that protects the process should not be
loosened by a deployment typo.
"""
from __future__ import annotations

UPLOAD_MAX_BYTES = 10 * 1024 * 1024      # 10 MiB of request body
UPLOAD_MAX_ROWS = 200_000                # data rows after parsing; the domain holds every row in memory
