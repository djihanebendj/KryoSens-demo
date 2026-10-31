"""Phase 7: technical knowledge and source-traceable retrieval (docs/KNOWLEDGE.md).

DOCUMENT → EXTRACTION → CHUNKING → METADATA → EMBEDDINGS → VECTOR STORE → RETRIEVAL → RANKING → CONTEXT → ANSWER.
Local only: nothing is sent outside this machine. Embeddings: lexical (default) or the pinned local semantic model.
"""

from .chunking import ChunkingParams
from .embeddings import SEMANTIC_BLOCKER, EmbeddingError, EmbeddingProvider, LexicalHashingProvider, get_provider
from .retrieval import AnswerGenerator, ExtractiveGenerator, GroundedContext, RetrievalParams, Retriever
from .schemas import Answer, AnswerStatus, ClaimType, DocumentMetadata, DocumentType, IngestOutcome
from .store import KnowledgeStore

__all__ = ["SEMANTIC_BLOCKER", "Answer", "AnswerGenerator", "AnswerStatus", "ChunkingParams", "ClaimType",
           "DocumentMetadata", "DocumentType", "EmbeddingError", "EmbeddingProvider", "ExtractiveGenerator",
           "GroundedContext", "IngestOutcome", "KnowledgeStore", "LexicalHashingProvider", "RetrievalParams",
           "Retriever", "get_provider"]
