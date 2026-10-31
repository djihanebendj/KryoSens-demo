"""Read-only access to the Phase 7 knowledge store for the Evidence page.

The Phase 7 pipeline remains the only retrieval implementation. This module only opens the already-built local
SQLite store, calls ``Retriever.answer`` and maps a missing/incompatible store to a clear result. It never sends
document text or a query to a model provider, and it never treats a quoted document as an instruction.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from coldsense.knowledge import EmbeddingError, KnowledgeStore, RetrievalParams, Retriever
from coldsense.knowledge.schemas import Answer

log = logging.getLogger(__name__)


class KnowledgeUnavailable(Exception):
    """The local knowledge index cannot answer right now. The message is safe to show to a user."""


class KnowledgeService:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def query(self, query: str, k: int = 5) -> Answer:
        if not self.path.is_file():
            raise KnowledgeUnavailable(
                "the knowledge base is not indexed; run scripts/index_knowledge.ps1 (or the documented "
                "`coldsense kb ingest` command) and try again"
            )
        try:
            with KnowledgeStore(self.path) as store:
                return Retriever(store, RetrievalParams(k=k)).answer(query)
        except EmbeddingError:
            log.warning("knowledge retrieval unavailable: the store's embedding provider could not be loaded")
            raise KnowledgeUnavailable(
                "the knowledge base was built with an embedding model that is not installed; re-index it with "
                "the lexical provider (see scripts/index_knowledge.ps1)"
            ) from None
        except (sqlite3.Error, OSError):
            log.warning("knowledge retrieval unavailable: the store could not be opened")
            raise KnowledgeUnavailable(
                "the knowledge base could not be opened; rebuild it with scripts/index_knowledge.ps1"
            ) from None
        except Exception:  # noqa: BLE001 - a retrieval bug must not turn the Evidence page into a 500
            log.exception("knowledge retrieval failed")
            raise KnowledgeUnavailable("the knowledge base could not answer this query; see the API log") from None
