"""Retrieval controller policy (step 2.4)."""

import pytest

from src.config import Settings
from src.retrieval.factory import build_retrieval
from src.stream.controller import ControllerConfig, RetrievalController, build_controller


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


def ctrl(stack, **kw):
    return build_controller(stack, Settings(), **kw)


async def run(c, parts, prior=None):
    c.start_utterance(prior)
    out, text = [], ""
    for i, p in enumerate(parts):
        text = f"{text} {p}".strip()
        out.append(await c.on_chunk(text, i * 0.8))
    out.append(await c.on_utterance_end(text, len(parts) * 0.8))
    return out


async def test_incomplete_fragment_waits(stack):
    c = ctrl(stack)
    c.start_utterance(None)
    d = await c.on_chunk("I would like to know about the", 0.0)
    assert d.action == "wait" and d.reason == "incomplete_fragment"


async def test_rule_stability_retrieves_provisionally_once_stable(stack):
    decisions = await run(ctrl(stack, stability_threshold=0.3),
                          ["Explain citation hallucination and fabricated ids", "in the grounding gates"])
    assert decisions[0].action == "wait" and decisions[0].reason == "low_stability"  # first probe scores 0
    assert decisions[1].action == "retrieve" and decisions[1].trigger == "provisional"
    assert decisions[1].stability_score >= 0.3


async def test_rule_only_retrieves_on_first_sufficient_chunk(stack):
    decisions = await run(ctrl(stack, mode="rule_only"), ["Explain citation hallucination risks", "please"])
    assert decisions[0].action == "retrieve" and decisions[0].trigger == "provisional"


async def test_stable_chunks_n_delays_retrieval(stack):
    parts = ["describe telemetry trace coverage", "for the observability gate", "and token cost logging"]
    one = await run(ctrl(stack, mode="rule_only", stable_chunks=1), parts)
    two = await run(ctrl(stack, mode="rule_only", stable_chunks=2), parts)
    first = lambda ds: next(i for i, d in enumerate(ds) if d.action == "retrieve")  # noqa: E731
    assert first(two) == first(one) + 1


async def test_provisional_limit_prevents_thrashing(stack):
    parts = ["describe telemetry trace coverage", "for the observability gate", "and token cost"]
    decisions = await run(ctrl(stack, mode="rule_only", max_provisional=1), parts)
    provisional = [d for d in decisions if d.trigger == "provisional"]
    assert len(provisional) == 1
    assert any(d.reason == "provisional_limit" for d in decisions)


async def test_final_retrieval_only_when_new_content(stack):
    same = await run(ctrl(stack, mode="rule_only"), ["describe the reranker component", "please"])
    assert same[-1].action == "wait" and same[-1].reason == "no_new_content"
    more = await run(ctrl(stack, mode="rule_only"), ["describe the reranker component", "and the telemetry schema"])
    assert more[-1].action == "retrieve" and more[-1].trigger == "final"


async def test_final_retrieval_when_nothing_retrieved_during_stream(stack):
    decisions = await run(ctrl(stack), ["what about grounding"])
    assert decisions[-1].action == "retrieve" and decisions[-1].trigger == "final"


async def test_insufficient_content_at_end_is_suppressed(stack):
    decisions = await run(ctrl(stack), ["so um", "can you"])
    assert all(d.action != "retrieve" for d in decisions)
    assert decisions[-1].action == "suppress" and decisions[-1].reason == "insufficient_content"


async def test_presentation_turn_never_retrieves(stack):
    decisions = await run(ctrl(stack), ["Could you rewrite that", "as bullet points?"], prior="Earlier answer text.")
    assert all(d.action != "retrieve" for d in decisions)
    assert decisions[-1].action == "suppress" and decisions[-1].reason == "presentation_restructure"


def test_invalid_config_is_rejected(stack):
    c = ctrl(stack)
    with pytest.raises(ValueError):
        RetrievalController(ControllerConfig(mode="bogus"), c.probe, c.gate)
    with pytest.raises(ValueError):
        RetrievalController(ControllerConfig(stable_chunks=0), c.probe, c.gate)
