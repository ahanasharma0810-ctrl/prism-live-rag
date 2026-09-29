"""Provisional retrieval wiring in the streaming engine (step 2.5)."""

import json

import pytest

from src.config import Settings
from src.engine import StreamingEngine, to_search_query
from src.retrieval.factory import build_retrieval
from src.stream.simulator import chunks_from_text
from src.telemetry.logger import TelemetryLogger


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


@pytest.fixture
def engine(stack, tmp_path):
    s = Settings(log_dir=tmp_path)
    eng = StreamingEngine(s, stack=stack, telemetry=TelemetryLogger(tmp_path / "t.jsonl"))
    eng.search_calls = 0
    original = eng._search

    def counting(query):
        eng.search_calls += 1
        return original(query)

    eng._search = counting
    return eng


def test_to_search_query_keeps_content_words():
    assert to_search_query("um so what is the uh Early retrieval target?") == "Early retrieval target"


async def test_provisional_retrieval_is_logged_before_utterance_end(engine, tmp_path):
    chunks = chunks_from_text(["Which gate covers", "early retrieval", "and how is it validated?"])
    result = await engine.run_turn(chunks, request_id="r1")
    assert result.retrieval_events[0].trigger == "provisional"
    assert result.retrieved_early and result.first_retrieval_s < result.utterance_end_s
    assert result.final_results and engine.search_calls == len(result.retrievals)

    names = [e.event for e in result.telemetry]
    assert names.index("retrieval_started") < names.index("utterance_end")
    started = next(e for e in result.telemetry if e.event == "retrieval_started")
    assert started.trigger == "provisional" and started.data["query"] and started.data["stream_ts"] == 0.8
    logged = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert {"retrieval_started", "retrieval_completed", "controller_decision", "request_completed"} <= {
        r["event"] for r in logged
    }


async def test_final_retrieval_at_end_when_nothing_stable_earlier(engine):
    result = await engine.run_turn(chunks_from_text(["what about grounding"]))
    assert [e.trigger for e in result.retrieval_events] == ["final"]
    assert not result.retrieved_early


async def test_presentation_turn_makes_zero_search_calls(engine):
    result = await engine.run_turn(chunks_from_text(["Could you rewrite that", "as a short list?"]),
                                   prior_output="Some earlier answer.")
    assert engine.search_calls == 0 and result.retrievals == []
    assert result.suppressed_reason == "presentation_restructure"


async def test_incomplete_utterance_is_suppressed_without_search(engine):
    result = await engine.run_turn(chunks_from_text(["so um", "can you"]))
    assert engine.search_calls == 0 and result.suppressed_reason == "insufficient_content"
