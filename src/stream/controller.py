"""Retrieval controller (step 2.4): decide wait / retrieve / suppress on each transcript chunk.

Policy (mode `rule_stability`, the default):
* suppress  - the suppression gate classifies the turn as presentation-only
              (reason `presentation_restructure`), or at utterance end the transcript has no
              searchable content, or trails off unfinished below the content minimums
              (reason `insufficient_content`; the caller should ask for clarification).
* wait      - the fragment is incomplete, lacks content, or its stability score is below
              `stability_threshold`.
* retrieve  - trigger `provisional` once stability holds for `stable_chunks` consecutive
              chunks (at most `max_provisional` per utterance, to avoid thrashing);
              trigger `final` at utterance end if new content arrived since the last
              retrieval (or nothing was retrieved yet).

Mode `rule_only` drops the stability requirement: a complete, content-sufficient fragment
counts as stable. Both modes share the same gate and end-of-utterance rule so the ablation
isolates the effect of the stability probe. The `multi_intent` trigger belongs to Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import Settings, get_settings
from src.schemas import ControllerDecision, Trigger
from src.stream.stability import ProbeResult, StabilityProbe, is_incomplete
from src.stream.suppression import PresentationGate

Mode = Literal["rule_only", "rule_stability"]


@dataclass(frozen=True)
class ControllerConfig:
    mode: Mode = "rule_stability"
    stability_threshold: float = 0.5
    stable_chunks: int = 1
    max_provisional: int = 1

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> ControllerConfig:
        s = s or get_settings()
        return cls(
            mode=s.controller_mode,  # type: ignore[arg-type]
            stability_threshold=s.stability_threshold,
            stable_chunks=s.stable_chunks,
            max_provisional=s.max_provisional,
        )


class RetrievalController:
    def __init__(self, config: ControllerConfig, probe: StabilityProbe, gate: PresentationGate):
        if config.mode not in ("rule_only", "rule_stability"):
            raise ValueError(f"unknown controller mode {config.mode!r}")
        if config.stable_chunks < 1:
            raise ValueError("stable_chunks must be >= 1")
        self.config = config
        self.probe = probe
        self.gate = gate
        self.start_utterance(None)

    @property
    def name(self) -> str:
        return self.config.mode

    # ------------------------------------------------------------------ state

    def start_utterance(self, prior_output: str | None) -> None:
        """Reset per-utterance state. `prior_output` is the session's previous output, if any."""
        self.prior_output = prior_output
        self.probe.reset()
        self.streak = 0
        self.provisional_count = 0
        self.retrieved_terms: set[str] | None = None
        self.last_probe: ProbeResult | None = None

    def _terms(self, transcript: str) -> set[str]:
        slots, ents, _ = self.probe.content(transcript)
        return set(slots) | {e.lower() for e in ents}

    def _decision(self, action, reason, ts, trigger: Trigger | None = None) -> ControllerDecision:
        score = self.last_probe.stability_score if self.last_probe is not None else None
        return ControllerDecision(action=action, reason=reason, trigger=trigger, ts=ts, stability_score=score)

    def _retrieve(self, transcript: str, ts: float, trigger: Trigger, reason: str) -> ControllerDecision:
        self.retrieved_terms = self._terms(transcript)
        return self._decision("retrieve", reason, ts, trigger)

    # ------------------------------------------------------------------ events

    async def on_chunk(self, transcript: str, ts: float) -> ControllerDecision:
        gate = self.gate.evaluate(transcript, self.prior_output)
        if gate.presentation_only:
            self.streak = 0
            return self._decision("suppress", gate.reason, ts)

        p = self.last_probe = self.probe.probe(transcript, ts)
        if p.incomplete or not p.sufficient:
            self.streak = 0
            return self._decision("wait", "incomplete_fragment" if p.incomplete else "insufficient_content", ts)

        holds = True if self.config.mode == "rule_only" else p.stability_score >= self.config.stability_threshold
        self.streak = self.streak + 1 if holds else 0
        if not holds:
            return self._decision("wait", "low_stability", ts)
        if self.streak < self.config.stable_chunks:
            return self._decision("wait", "stabilizing", ts)
        if self.provisional_count >= self.config.max_provisional:
            return self._decision("wait", "provisional_limit", ts)
        self.provisional_count += 1
        return self._retrieve(transcript, ts, "provisional", "stable_intent")

    async def on_utterance_end(self, transcript: str, ts: float) -> ControllerDecision:
        gate = self.gate.evaluate(transcript, self.prior_output)
        if gate.presentation_only:
            return self._decision("suppress", gate.reason, ts)
        slots, ents, n_tokens = self.probe.content(transcript)
        n_content = len(slots) + len(ents)
        thin = n_content < self.probe.min_content_terms or n_tokens < self.probe.min_tokens
        if n_content == 0 or (is_incomplete(transcript) and thin):
            return self._decision("suppress", "insufficient_content", ts)
        if self.retrieved_terms is None:
            return self._retrieve(transcript, ts, "final", "utterance_end")
        new = self._terms(transcript) - self.retrieved_terms
        if new:
            return self._retrieve(transcript, ts, "final", "new_content_since_last_retrieval")
        return self._decision("wait", "no_new_content", ts)


def build_controller(stack, settings: Settings | None = None, mode: Mode | None = None, **overrides) -> RetrievalController:
    """Controller wired to the corpus BM25 index, configured from settings."""
    s = settings or get_settings()
    cfg = ControllerConfig.from_settings(s)
    if mode is not None:
        overrides["mode"] = mode
    if overrides:
        cfg = ControllerConfig(**{**cfg.__dict__, **overrides})
    probe = StabilityProbe(stack.bm25, k=s.probe_k, min_content_terms=s.probe_min_content_terms,
                           min_tokens=s.probe_min_tokens)
    gate = PresentationGate(stack.bm25.vocabulary, threshold=s.gate_threshold)
    return RetrievalController(cfg, probe, gate)
