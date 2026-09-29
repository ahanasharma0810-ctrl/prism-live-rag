"""eval/run_eval.py (step 1.10)."""

import subprocess
import sys
from pathlib import Path

import pytest

from eval.run_eval import evaluate_baseline, recall_at_k
from eval.scenarios import load_scenarios
from src.schemas import CorpusChunk, ScoredChunk
from src.synthesis.generator import ExtractiveGenerator

ROOT = Path(__file__).resolve().parents[1]


def test_recall_at_k():
    assert recall_at_k(["Doc_01 §5", "Doc_01 §3"], ["Doc_01 §5"]) == 1.0
    assert recall_at_k(["Doc_01 §3"], ["Doc_01 §5", "Doc_01 §3"]) == 0.5
    with pytest.raises(ValueError):
        recall_at_k(["Doc_01 §3"], [])


async def test_evaluate_baseline_on_a_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PRISM_INDEX_DIR", str(tmp_path / "idx"))
    subset = [s for s in load_scenarios() if s.id in {"single_02", "noevid_01", "present_04"}]
    out = await evaluate_baseline(subset)
    summary = out["summary"]
    assert summary["turns"] == 4
    assert summary["citation_validity"]["fabricated"] == 0
    assert summary["recall_at_5"]["n"] == 2
    assert summary["no_retrieval_turns"] == {"n": 1, "retrieved_anyway": 1}
    assert set(summary["retrieval_only_recall_at_5"]) == {"bm25", "dense", "hybrid"}
    assert (tmp_path / "logs" / "telemetry.jsonl").exists()


def test_later_phase_systems_are_rejected():
    result = subprocess.run([sys.executable, "eval/run_eval.py", "--system", "full"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "Phase 4" in result.stderr


async def test_extractive_falls_back_to_leading_units_on_heading_match():
    chunk = CorpusChunk(chunk_id="D §1#1", doc_id="D", section="1", heading="Common Pitfalls",
                        text="T — §1 Common Pitfalls\n\nFirst sentence here. Second sentence here.")
    hit = ScoredChunk(chunk=chunk, score=1.0, rank=1, source="rrf")
    synth = await ExtractiveGenerator(per_chunk=1).synthesize("common pitfalls", [hit])
    assert [c.text for c in synth.claims] == ["First sentence here."]
