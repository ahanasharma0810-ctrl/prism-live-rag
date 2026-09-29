"""Streaming engine (Phase 2 scope: controller + provisional retrieval wiring, step 2.5).

Consumes a transcript stream, asks the controller for a decision on every chunk and at the
utterance end, and when the decision is `retrieve` dispatches a hybrid search as a
background task (so the stream keeps flowing) and logs `retrieval_started` with the stream
timestamp, query and trigger. Decomposition (Phase 3) and session refinement / answer
synthesis (Phase 4) are not part of this module yet.

    python -m src.engine --utterance "first fragment | second fragment" [--clock wall] [--prior "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.retrieval.factory import RetrievalStack, build_retrieval
from src.retrieval.rerank import Reranker, make_reranker
from src.retrieval.text import STOPWORDS
from src.schemas import ControllerDecision, RetrievalEvent, ScoredChunk, TelemetryEvent, TranscriptChunk, Trigger
from src.stream.controller import RetrievalController, build_controller
from src.stream.simulator import Clock, chunks_from_text, replay
from src.stream.stability import FILLERS, surface_tokens
from src.telemetry import events as ev
from src.telemetry.logger import RequestTrace, TelemetryLogger


def to_search_query(transcript: str) -> str:
    """Keyword query from the transcript: content words in order, fillers/stopwords dropped."""
    seen: dict[str, str] = {}
    for w in surface_tokens(transcript):
        key = w.lower()
        if key not in STOPWORDS and key not in FILLERS and key not in seen:
            seen[key] = w
    return " ".join(seen.values())


@dataclass
class RetrievalRun:
    event: RetrievalEvent
    results: list[ScoredChunk] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class StreamTurnResult:
    utterance: str
    utterance_end_s: float
    decisions: list[ControllerDecision]
    retrievals: list[RetrievalRun]
    suppressed_reason: str | None
    telemetry: list[TelemetryEvent]

    @property
    def retrieval_events(self) -> list[RetrievalEvent]:
        return [r.event for r in self.retrievals]

    @property
    def first_retrieval_s(self) -> float | None:
        return min((r.event.timestamp_s for r in self.retrievals), default=None)

    @property
    def retrieved_early(self) -> bool:
        first = self.first_retrieval_s
        return first is not None and first < self.utterance_end_s

    @property
    def final_results(self) -> list[ScoredChunk]:
        return self.retrievals[-1].results if self.retrievals else []


class StreamingEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        stack: RetrievalStack | None = None,
        controller: RetrievalController | None = None,
        reranker: Reranker | None = None,
        telemetry: TelemetryLogger | None = None,
    ):
        self.settings = s = settings or get_settings()
        self.stack = stack or build_retrieval(s)
        self.controller = controller or build_controller(self.stack, s)
        self.reranker = reranker or make_reranker(s.rerank_backend, s.rerank_model)
        self.telemetry = telemetry or TelemetryLogger(s.log_dir / "telemetry.jsonl")

    def _search(self, query: str) -> list[ScoredChunk]:
        hits = self.stack.hybrid.search(query, k=self.settings.candidates)
        return self.reranker.rerank(query, hits, self.settings.top_k)

    async def _dispatch(self, run: RetrievalRun, trace: RequestTrace) -> None:
        t = time.perf_counter()
        run.results = await asyncio.to_thread(self._search, run.event.query)
        run.latency_ms = (time.perf_counter() - t) * 1000
        trace.emit(
            ev.RETRIEVAL_COMPLETED,
            stage_latency_ms=run.latency_ms,
            trigger=run.event.trigger,
            stream_ts=run.event.timestamp_s,
            chunk_ids=[h.chunk.chunk_id for h in run.results],
        )

    def _start(self, transcript: str, ts: float, trigger: Trigger, trace: RequestTrace,
               runs: list[RetrievalRun], tasks: list[asyncio.Task]) -> None:
        query = to_search_query(transcript)
        run = RetrievalRun(RetrievalEvent(timestamp_s=ts, query=query, trigger=trigger))
        runs.append(run)
        trace.emit(ev.RETRIEVAL_STARTED, trigger=trigger, stream_ts=ts, query=query)
        tasks.append(asyncio.create_task(self._dispatch(run, trace)))

    async def run_turn(
        self,
        chunks: list[TranscriptChunk],
        prior_output: str | None = None,
        clock: Clock = "simulated",
        speed: float = 1.0,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> StreamTurnResult:
        trace = self.telemetry.trace(request_id, session_id)
        trace.emit(ev.REQUEST_STARTED, system="streaming", controller=self.controller.name, clock=clock,
                   has_prior_output=bool(prior_output))
        self.controller.start_utterance(prior_output)
        decisions: list[ControllerDecision] = []
        runs: list[RetrievalRun] = []
        tasks: list[asyncio.Task] = []
        transcript, end_ts = "", 0.0

        async for event in replay(chunks, clock=clock, speed=speed):
            transcript = event.transcript
            if event.kind == "chunk":
                decision = await self.controller.on_chunk(transcript, event.ts)
            else:
                end_ts = event.ts
                trace.emit(ev.UTTERANCE_END, stream_ts=event.ts, implicit=event.implicit_end)
                decision = await self.controller.on_utterance_end(transcript, event.ts)
            decisions.append(decision)
            trace.emit(ev.CONTROLLER_DECISION, stream_ts=event.ts, action=decision.action, reason=decision.reason,
                       trigger_decided=decision.trigger, stability_score=decision.stability_score)
            if decision.action == "retrieve" and decision.trigger is not None:
                self._start(transcript, event.ts, decision.trigger, trace, runs, tasks)

        if tasks:
            await asyncio.gather(*tasks)
        last = decisions[-1] if decisions else None
        suppressed = last.reason if last is not None and last.action == "suppress" else None
        result = StreamTurnResult(transcript, end_ts, decisions, runs, suppressed, [])
        trace.emit(ev.REQUEST_COMPLETED, stage_latency_ms=trace.elapsed_ms(), retrievals=len(runs),
                   first_retrieval_s=result.first_retrieval_s, utterance_end_s=end_ts,
                   retrieved_early=result.retrieved_early, suppressed_reason=suppressed)
        result.telemetry = list(trace.events)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream one utterance through the controller and retrieval.")
    parser.add_argument("--utterance", required=True, help="Fragments separated by '|'.")
    parser.add_argument("--step", type=float, default=0.8, help="Seconds between fragments.")
    parser.add_argument("--clock", choices=["simulated", "wall"], default="simulated")
    parser.add_argument("--prior", default=None, help="Previous output in this session, if any.")
    args = parser.parse_args()
    parts = [p.strip() for p in args.utterance.split("|") if p.strip()]
    if not parts:
        parser.error("--utterance is empty")

    result = asyncio.run(StreamingEngine().run_turn(chunks_from_text(parts, step_s=args.step), args.prior, clock=args.clock))
    out = {
        "decisions": [d.model_dump(exclude_none=True) for d in result.decisions],
        "retrieval_events": [e.model_dump() for e in result.retrieval_events],
        "retrieved_chunk_ids": [h.chunk.chunk_id for h in result.final_results],
        "utterance_end_s": result.utterance_end_s,
        "retrieved_early": result.retrieved_early,
        "suppressed_reason": result.suppressed_reason,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
