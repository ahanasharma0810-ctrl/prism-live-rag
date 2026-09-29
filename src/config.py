"""Runtime configuration read from environment variables (see .env.example).

Every value has a CPU-only default so the repo runs without any model download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _path(name: str, default: str) -> Path:
    p = Path(_env(name, default))
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = field(default_factory=lambda: _path("PRISM_CORPUS_DIR", "data/corpus"))
    log_dir: Path = field(default_factory=lambda: _path("PRISM_LOG_DIR", "logs"))
    index_dir: Path = field(default_factory=lambda: _path("PRISM_INDEX_DIR", "indexes"))

    chunk_max_tokens: int = field(default_factory=lambda: int(_env("PRISM_CHUNK_MAX_TOKENS", "400")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(_env("PRISM_CHUNK_OVERLAP_TOKENS", "40")))

    dense_backend: str = field(default_factory=lambda: _env("PRISM_DENSE_BACKEND", "hashing"))
    dense_model: str = field(default_factory=lambda: _env("PRISM_DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_backend: str = field(default_factory=lambda: _env("PRISM_RERANK_BACKEND", "lexical"))
    rerank_model: str = field(
        default_factory=lambda: _env("PRISM_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rrf_k: int = field(default_factory=lambda: int(_env("PRISM_RRF_K", "60")))
    candidates: int = field(default_factory=lambda: int(_env("PRISM_CANDIDATES", "20")))
    top_k: int = field(default_factory=lambda: int(_env("PRISM_TOP_K", "5")))

    min_evidence_score: float = field(default_factory=lambda: float(_env("PRISM_MIN_EVIDENCE_SCORE", "0.34")))

    # suppression gate (phase 2)
    gate_threshold: float = field(default_factory=lambda: float(_env("PRISM_GATE_THRESHOLD", "0.5")))
    # stability probe (phase 2)
    probe_k: int = field(default_factory=lambda: int(_env("PRISM_PROBE_K", "3")))
    probe_min_content_terms: int = field(default_factory=lambda: int(_env("PRISM_PROBE_MIN_CONTENT_TERMS", "2")))
    probe_min_tokens: int = field(default_factory=lambda: int(_env("PRISM_PROBE_MIN_TOKENS", "4")))
    # retrieval controller (phase 2)
    controller_mode: str = field(default_factory=lambda: _env("PRISM_CONTROLLER_MODE", "rule_only"))
    stability_threshold: float = field(default_factory=lambda: float(_env("PRISM_STABILITY_THRESHOLD", "0.2")))
    stable_chunks: int = field(default_factory=lambda: int(_env("PRISM_STABLE_CHUNKS", "1")))
    max_provisional: int = field(default_factory=lambda: int(_env("PRISM_MAX_PROVISIONAL", "1")))

    # decomposition and fusion (phase 3)
    decomposer: str = field(default_factory=lambda: _env("PRISM_DECOMPOSER", "rules"))

    llm_provider: str = field(default_factory=lambda: _env("PRISM_LLM_PROVIDER", "none"))
    llm_model: str = field(default_factory=lambda: _env("PRISM_LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: _env("PRISM_LLM_BASE_URL", "http://localhost:11434"))
    llm_api_key: str = field(default_factory=lambda: _env("PRISM_LLM_API_KEY", ""))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("PRISM_LLM_TIMEOUT_S", "120")))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("PRISM_LLM_MAX_TOKENS", "512")))
    cost_per_1k_input: float = field(default_factory=lambda: float(_env("PRISM_COST_PER_1K_INPUT", "0")))
    cost_per_1k_output: float = field(default_factory=lambda: float(_env("PRISM_COST_PER_1K_OUTPUT", "0")))


def get_settings() -> Settings:
    return Settings()
