"""Run a system over the dev scenarios and report metrics (step 1.10).

    python eval/run_eval.py --system baseline [--out results/baseline_dev.json] [--verbose]

Phase 1 metrics (baseline):
* recall@5            share of gold supporting sections found among the top-5 reranked chunks
                      (turns that need retrieval and have gold evidence)
* citation validity   share of citations in answers that resolve to a real corpus section;
                      `fabricated` counts the ones that do not
* latency             end-to-end ms per turn (p50 / p95 / mean)
Informational:
* uncertainty on no-evidence turns / false uncertainty on answerable turns
* retrieval on no-retrieval turns (the baseline has no controller, so it always retrieves)
* retrieval-only recall@5 for bm25 / dense / hybrid (input for later ablations)

The baseline is non-streaming and stateless: every turn is answered independently from its
full utterance. Dev scenarios are evaluation data only; nothing in src/ imports them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scenarios import SCENARIO_DIR, Scenario, load_scenarios  # noqa: E402
from src.baseline import BaselinePipeline  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.retrieval.factory import build_retrieval  # noqa: E402
from src.synthesis.citations import CitationIndex, extract_citations  # noqa: E402

K = 5
SYSTEMS = {"baseline": 1, "controller_only": 2, "decompose_fusion": 3, "full": 4}


def recall_at_k(retrieved_citations: list[str], gold: list[str]) -> float:
    if not gold:
        raise ValueError("recall is undefined without gold evidence")
    return sum(g in retrieved_citations for g in gold) / len(gold)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


async def evaluate_baseline(scenarios: list[Scenario]) -> dict:
    settings = get_settings()
    stack = build_retrieval(settings)
    pipeline = BaselinePipeline(settings, stack=stack)
    index = CitationIndex(stack.chunks)

    turns_out = []
    for scenario in scenarios:
        session_id = f"eval-{scenario.id}-{int(time.time() * 1000)}"
        for t in scenario.turns:
            utterance = t.utterance
            if not utterance:
                turns_out.append({"scenario": scenario.id, "turn": t.turn, "category": scenario.category,
                                  "skipped": "empty utterance"})
                continue
            result = await pipeline.answer(
                utterance, utterance_end_s=t.utterance_end_s,
                request_id=f"{scenario.id}-t{t.turn}", session_id=session_id,
            )
            top_citations = [h.chunk.citation for h in result.reranked[:K]]
            cited = extract_citations(result.record.answer)
            gold = t.gold_supporting
            row = {
                "scenario": scenario.id,
                "turn": t.turn,
                "category": scenario.category,
                "utterance": utterance,
                "retrieval_required": t.retrieval_required,
                "expect_uncertainty": t.expect_uncertainty,
                "gold_supporting": gold,
                "top5": top_citations,
                "recall_at_5": recall_at_k(top_citations, gold) if (t.retrieval_required and gold) else None,
                "cited": cited,
                "invalid_citations": [c for c in cited if not index.is_valid(c)],
                "retrieved": bool(result.record.retrieval_events),
                "uncertainty": result.record.uncertainty,
                "latency_ms": round(result.latency_ms, 3),
                "record": result.record.model_dump(),
            }
            turns_out.append(row)

    # retrieval-only recall for each retriever (no rerank), same turns
    ablation = {}
    for name in ("bm25", "dense", "hybrid"):
        retriever = stack.by_name(name)
        vals = []
        for row in turns_out:
            if row.get("recall_at_5") is None:
                continue
            hits = retriever.search(row["utterance"], k=K)
            vals.append(recall_at_k([h.chunk.citation for h in hits], row["gold_supporting"]))
        ablation[name] = round(statistics.mean(vals), 4) if vals else None

    answered = [r for r in turns_out if "skipped" not in r]
    recalls = [r["recall_at_5"] for r in answered if r["recall_at_5"] is not None]
    all_cites = [c for r in answered for c in r["cited"]]
    invalid = [c for r in answered for c in r["invalid_citations"]]
    lat = [r["latency_ms"] for r in answered]
    no_ev = [r for r in answered if r["expect_uncertainty"]]
    answerable = [r for r in answered if r["retrieval_required"] and not r["expect_uncertainty"]]
    no_retr = [r for r in answered if not r["retrieval_required"]]

    summary = {
        "system": "baseline",
        "scenarios": len(scenarios),
        "turns": len(turns_out),
        "turns_skipped": len(turns_out) - len(answered),
        "recall_at_5": {"mean": round(statistics.mean(recalls), 4) if recalls else None, "n": len(recalls),
                        "perfect": sum(v == 1.0 for v in recalls), "zero": sum(v == 0.0 for v in recalls)},
        "citation_validity": {
            "citations": len(all_cites),
            "valid": len(all_cites) - len(invalid),
            "fabricated": len(invalid),
            "rate": round((len(all_cites) - len(invalid)) / len(all_cites), 4) if all_cites else None,
        },
        "latency_ms": {"p50": round(percentile(lat, 0.5), 3), "p95": round(percentile(lat, 0.95), 3),
                       "mean": round(statistics.mean(lat), 3) if lat else None, "n": len(lat)},
        "uncertainty": {
            "expected_turns": len(no_ev),
            "emitted_when_expected": sum(r["uncertainty"] is not None for r in no_ev),
            "answerable_turns": len(answerable),
            "false_uncertainty_on_answerable": sum(r["uncertainty"] is not None for r in answerable),
        },
        "no_retrieval_turns": {"n": len(no_retr), "retrieved_anyway": sum(r["retrieved"] for r in no_retr)},
        "retrieval_only_recall_at_5": ablation,
        "config": {
            "dense_backend": settings.dense_backend, "rerank_backend": settings.rerank_backend,
            "llm_provider": settings.llm_provider, "rrf_k": settings.rrf_k, "top_k": settings.top_k,
            "min_evidence_score": settings.min_evidence_score,
        },
    }
    return {"summary": summary, "turns": turns_out}


def print_summary(s: dict) -> None:
    r, c, lat, u, nr = s["recall_at_5"], s["citation_validity"], s["latency_ms"], s["uncertainty"], s["no_retrieval_turns"]
    print(f"\nSystem: {s['system']}   scenarios={s['scenarios']} turns={s['turns']} (skipped {s['turns_skipped']})")
    print("| metric | value |")
    print("|---|---|")
    print(f"| recall@5 (mean over {r['n']} turns) | {r['mean']} (perfect {r['perfect']}, zero {r['zero']}) |")
    print(f"| citation validity | {c['valid']}/{c['citations']} = {c['rate']} (fabricated {c['fabricated']}) |")
    print(f"| latency ms p50 / p95 / mean | {lat['p50']} / {lat['p95']} / {lat['mean']} |")
    print(f"| uncertainty emitted on no-evidence turns | {u['emitted_when_expected']}/{u['expected_turns']} |")
    print(f"| false uncertainty on answerable turns | {u['false_uncertainty_on_answerable']}/{u['answerable_turns']} |")
    print(f"| no-retrieval turns that retrieved anyway | {nr['retrieved_anyway']}/{nr['n']} |")
    ab = s["retrieval_only_recall_at_5"]
    print(f"| retrieval-only recall@5 bm25 / dense / hybrid | {ab['bm25']} / {ab['dense']} / {ab['hybrid']} |")
    print(f"config: {json.dumps(s['config'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="baseline", choices=sorted(SYSTEMS))
    parser.add_argument("--metrics", default="phase1", help="Only 'phase1' is implemented.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--out", type=Path, help="Write full per-turn results as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print one line per turn.")
    args = parser.parse_args()

    if args.system != "baseline":
        print(f"--system {args.system} is not implemented until Phase {SYSTEMS[args.system]}.", file=sys.stderr)
        return 2
    if args.metrics != "phase1":
        print(f"--metrics {args.metrics} is not implemented in Phase 1.", file=sys.stderr)
        return 2

    results = asyncio.run(evaluate_baseline(load_scenarios(args.scenarios)))
    if args.verbose:
        for row in results["turns"]:
            if "skipped" in row:
                print(f"{row['scenario']:<12} t{row['turn']} skipped: {row['skipped']}")
                continue
            print(f"{row['scenario']:<12} t{row['turn']} recall@5={row['recall_at_5']} "
                  f"cited={row['cited']} uncertainty={'yes' if row['uncertainty'] else 'no'} "
                  f"{row['latency_ms']:.1f}ms")
    print_summary(results["summary"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if results["summary"]["citation_validity"]["fabricated"] else 0


if __name__ == "__main__":
    sys.exit(main())
