"""SEMANTIC embedding provider: BAAI/bge-small-en-v1.5 at a pinned revision, run locally with onnxruntime.

TEXT -> TOKENIZATION (the model's own tokenizer.json: BERT WordPiece, lower case, [CLS] ... [SEP], at most 512
tokens) -> ONNX INFERENCE (input_ids, attention_mask, token_type_ids -> last_hidden_state [batch, seq, 384]) ->
CLS POOLING (1_Pooling/config.json: pooling_mode_cls_token true, mean pooling false) -> L2 NORMALISATION
(modules.json: Normalize) -> 384-d float32 vector.

Every asset is checked against the SHA-256 recorded below for the pinned revision before it is used, so a
missing, corrupt or different file is refused instead of silently changing the vectors. Nothing is downloaded
here: the assets must already be in the model directory (docs/KNOWLEDGE.md §1 says how they were obtained).
Inference is local CPU only. English only: the model was trained on English text.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from .embeddings import EmbeddingError

REPO = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
LICENSE = "MIT"
DIM = 384
MAX_TOKENS = 512
MODEL_DIR_ENV = "COLDSENSE_EMBEDDING_MODEL_DIR"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "coldsense" / "models" / "BAAI--bge-small-en-v1.5" / REVISION

# SHA-256 of each asset at REVISION (the ONNX hash is the hub's LFS hash; the others were checked against the
# revision's git blob ids when downloaded).
ASSETS = {
    "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    "1_Pooling/config.json": "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
}

# The model card's instruction for retrieval queries; never applied to passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QueryPolicy = Literal["plain", "instruct"]
BATCH = 16

# Frozen on dev-v1 (eval-corpus-v2) by evaluation.select_on_development (plain 25/30, instruction prefix 24/30);
# used with the extractive-v1 generator (D-059). Only the selected identity has parameters; any other identity must
# be given them explicitly (Retriever refuses to guess).
DEFAULT_QUERY_POLICY: QueryPolicy = "plain"
D059_RETRIEVAL = {"min_similarity": 0.74, "semantic_gate": "hybrid", "semantic_ranking": "cosine"}
SELECTED_RETRIEVAL = {"plain": dict(D059_RETRIEVAL)}
# answerability-v1 (D-060): selected on dev-v2 (27/30), then REJECTED as a default by the fresh heldout-v3 run
# (false support 6 of 13 unanswerable, against 2 of 13 for the default). Opt-in only, by passing these parameters.
ANSWERABILITY_V1 = {"min_similarity": 0.68, "semantic_gate": "hybrid", "semantic_ranking": "cosine",
                    "min_sentence_similarity": 0.74, "answer_contrast": True}


def model_id(query_policy: QueryPolicy) -> str:
    """The full embedding identity: model, revision, dimension, pooling, normalisation and query policy. Stores
    compare it exactly, so a change in any part needs a re-index."""
    return f"bge-small-en-v1.5@{REVISION[:12]}/{DIM}/cls-l2/q-{query_policy}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_assets(model_dir: Path) -> dict[str, str]:
    """Check every asset is present and has the pinned hash; return the hashes. Raises EmbeddingError."""
    out = {}
    for rel, expected in ASSETS.items():
        p = model_dir / rel
        if not p.is_file():
            raise EmbeddingError(f"{REPO} asset '{rel}' is missing from {model_dir} (set {MODEL_DIR_ENV})")
        got = _sha256(p)
        if got != expected:
            raise EmbeddingError(f"{REPO} asset '{rel}' does not match revision {REVISION[:12]} "
                                 f"(sha256 {got[:16]}…, expected {expected[:16]}…)")
        out[rel] = got
    pool = json.loads((model_dir / "1_Pooling/config.json").read_text())
    if not (pool.get("pooling_mode_cls_token") is True and pool.get("pooling_mode_mean_tokens") is False
            and pool.get("word_embedding_dimension") == DIM):
        raise EmbeddingError(f"{REPO} pooling configuration is not CLS pooling of {DIM} dimensions")
    if not any(m.get("type", "").endswith("Normalize") for m in json.loads((model_dir / "modules.json").read_text())):
        raise EmbeddingError(f"{REPO} module list has no Normalize step")
    return out


class BgeSmallOnnxProvider:
    kind: Literal["LEXICAL", "SEMANTIC"] = "SEMANTIC"
    dim = DIM

    def __init__(self, query_policy: QueryPolicy = DEFAULT_QUERY_POLICY, model_dir: Path | None = None) -> None:
        if query_policy not in ("plain", "instruct"):
            raise EmbeddingError(f"unknown query policy '{query_policy}'")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise EmbeddingError(f"{REPO} needs onnxruntime and tokenizers (pip install 'coldsense[semantic]')") from e
        self.model_dir = Path(model_dir or os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
        self.asset_sha256 = verify_assets(self.model_dir)
        self.query_policy: QueryPolicy = query_policy
        self.model_id = model_id(query_policy)
        self.selected_retrieval = SELECTED_RETRIEVAL.get(query_policy)
        try:
            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1          # one thread: identical float results on every run
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(self.model_dir / "onnx/model.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        except Exception as e:
            raise EmbeddingError(f"{REPO} assets could not be loaded ({type(e).__name__})") from e
        names = {i.name: i.type for i in sess.get_inputs()}
        outs = sess.get_outputs()
        if names != {n: "tensor(int64)" for n in ("input_ids", "attention_mask", "token_type_ids")} or \
                [o.name for o in outs] != ["last_hidden_state"] or outs[0].shape[-1] != DIM:
            raise EmbeddingError(f"{REPO} ONNX graph has an unexpected input/output contract")
        pad = tok.token_to_id("[PAD]")
        if pad != 0 or tok.token_to_id("[CLS]") != 101 or tok.token_to_id("[SEP]") != 102:
            raise EmbeddingError(f"{REPO} tokenizer special tokens are not the BERT ids the model expects")
        tok.enable_truncation(MAX_TOKENS)
        tok.enable_padding(pad_id=pad, pad_token="[PAD]")
        self._tok, self._sess = tok, sess
        self._queries: dict[str, np.ndarray] = {}

    def identity(self) -> dict:
        return {"provider": "BgeSmallOnnxProvider", "model_id": self.model_id, "repository": REPO,
                "revision": REVISION, "license": LICENSE, "dim": DIM, "max_tokens": MAX_TOKENS,
                "pooling": "CLS", "normalisation": "L2", "query_policy": self.query_policy,
                "query_instruction": QUERY_INSTRUCTION if self.query_policy == "instruct" else None,
                "selected_retrieval": self.selected_retrieval, "asset_sha256": self.asset_sha256}

    def tokenize(self, texts: list[str]):
        return self._tok.encode_batch(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Passage vectors: the text as given (no instruction), CLS pooled, L2-normalised."""
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for s in range(0, len(texts), BATCH):
            enc = self._tok.encode_batch(texts[s:s + BATCH])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                     "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64)}
            hidden = self._sess.run(["last_hidden_state"], feeds)[0]
            cls = hidden[:, 0, :].astype(np.float32)               # position 0 is [CLS]; padding is never pooled
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            out[s:s + len(enc)] = cls / np.where(norms > 0, norms, 1)
        return out

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Query vectors under the recorded query policy (instruction prefix or plain text)."""
        prefix = QUERY_INSTRUCTION if self.query_policy == "instruct" else ""
        missing = [t for t in dict.fromkeys(texts) if t not in self._queries]
        if missing:
            for t, v in zip(missing, self.embed([prefix + t for t in missing]), strict=True):
                if len(self._queries) < 4096:
                    self._queries[t] = v
                else:
                    return self.embed([prefix + t for t in texts])
        return np.stack([self._queries[t] for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)


def from_model_id(name: str) -> BgeSmallOnnxProvider:
    """The provider for a stored or requested identity: 'bge-small-en-v1.5' (default policy) or a full model id,
    which must name this exact revision, dimension, pooling and normalisation."""
    if name == "bge-small-en-v1.5":
        return BgeSmallOnnxProvider()
    for policy in ("plain", "instruct"):
        if name == model_id(policy):
            return BgeSmallOnnxProvider(policy)
    raise EmbeddingError(f"'{name}' is not a supported bge-small-en-v1.5 identity; supported: "
                         f"{model_id('plain')}, {model_id('instruct')}")
