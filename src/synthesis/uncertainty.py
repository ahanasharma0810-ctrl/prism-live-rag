"""Uncertainty handling (baseline level).

The evidence score of a chunk is the fraction of the request's content terms that occur in
it (lexical coverage, independent of which reranker is configured). If no retrieved chunk
reaches `min_score`, the system states that the corpus lacks evidence and asks for
clarification instead of answering. Phase 4 adds claim-level verification failures.
"""

from __future__ import annotations

from src.retrieval.text import content_terms
from src.schemas import ScoredChunk


def evidence_score(query: str, text: str) -> float:
    q = content_terms(query)
    if not q:
        return 0.0
    return len(q & content_terms(text)) / len(q)


def supported_hits(query: str, hits: list[ScoredChunk], min_score: float) -> list[ScoredChunk]:
    return [h for h in hits if evidence_score(query, h.chunk.text) >= min_score]


def uncertainty_note(query: str, hits: list[ScoredChunk], min_score: float) -> str | None:
    """Return an explicit uncertainty / clarification note, or None when evidence suffices."""
    q = content_terms(query)
    if not q:
        return "The request contains no searchable content. Please restate what you would like to know."
    if not hits:
        return "No evidence for this request was found in the corpus. Please clarify or rephrase the request."
    best = max(evidence_score(query, h.chunk.text) for h in hits)
    if best >= min_score:
        return None
    covered = set().union(*(content_terms(h.chunk.text) for h in hits)) & q
    missing = sorted(q - covered)
    detail = f" Terms not found in the retrieved evidence: {', '.join(missing)}." if missing else ""
    return (
        "The corpus does not contain sufficient evidence to answer this request reliably"
        f" (best evidence coverage {best:.2f} < {min_score:.2f}).{detail}"
        " Please clarify or narrow the request."
    )
