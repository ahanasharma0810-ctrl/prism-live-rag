# PRISM Streaming Live RAG

An event-driven retrieval-augmented generation engine that listens to a live transcript,
retrieves early, decomposes multi-intent requests, refines answers when late constraints
arrive, and grounds every claim in a `[Doc_ID §Section]` citation.
The problem statement is the Theme 4 guide (transcribed into `data/corpus/Doc_01.md`).
The build plan is `docs/agent_playbook.md`.

> **Status: Phase 2 (Controller & Live Stream Simulation) complete.**
> * Phase 1: a clean indexed corpus, schemas, a non-streaming **baseline**, dev scenarios and metrics.
> * Phase 2: a stream simulator, a presentation-only suppression gate, a BM25 stability probe,
>   a wait/retrieve/suppress **controller**, and a streaming engine that starts retrieval
>   before the utterance ends (G2 = 0.97 on the dev set).
>
> Decomposition, session refinement, grounding verification and full telemetry are
> Phases 3-5 and are **not** implemented yet (their modules are placeholders).
> See `reports/PHASE_1_REPORT.md`, `reports/PHASE_2_REPORT.md` and
> `reports/PHASE_2_CONTROLLER_ABLATION.md`.

## Quick start (CPU only, no model downloads)

Requires Python 3.11.

```bash
python -m venv .venv && . .venv/bin/activate
make install          # pinned runtime + dev dependencies (pydantic, rank-bm25, numpy, pytest, jsonschema)
make test             # full test suite
make audit            # corpus audit
make index            # build BM25 + dense indexes, write indexes/chunks.jsonl
make baseline Q="What are the technical evaluation gates and their target thresholds?"
make eval             # baseline over eval/dev_scenarios: recall@5, citation validity, latency
make stream U="Which gate covers | early retrieval | and how is it validated?"   # streaming controller demo
make controller       # G2 early retrieval / false triggers / seconds gained over the dev scenarios
make ablate           # controller ablation + threshold grid -> reports/PHASE_2_CONTROLLER_ABLATION.md
```

Add `--clock wall` to `python -m src.engine ...` to replay fragments at real speed, and
`--prior "..."` to simulate an earlier answer in the session (needed for suppression).

Without `make`: `python -m pytest -q`, `python -m src.corpus.build_index`,
`python -m src.baseline --query "..."`, `python eval/run_eval.py --system baseline`.

Output of the baseline is the record defined in the problem statement:

```json
{"retrieval_events": [{"timestamp_s": 0.0, "query": "...", "trigger": "final"}],
 "sub_queries": ["..."], "answer": "... [Doc_01 §5]", "citations": ["Doc_01 §5"], "uncertainty": null}
```

Telemetry for every request is appended to `logs/telemetry.jsonl`
(schema: `schemas/telemetry_event.schema.json`).

## Architecture

Streaming path (Phase 2):

```
transcript chunks ─► simulator ─► suppression gate ─► stability probe ─► controller ─► retrieve? ─► hybrid search + rerank
 (ts, text)          (simulated     presentation-only?   BM25 top-k Jaccard  wait / retrieve     (background task, logs
                      or wall)      -> suppress          + content check     (provisional|final) retrieval_started)
                                                                             / suppress
```

Baseline path (Phase 1):

```
utterance ─► hybrid retrieval ─► rerank ─► evidence threshold ─► synthesis ─► citation check ─► OutputRecord
             BM25 + dense, RRF    lexical     (uncertainty)       extractive      ids must exist      + telemetry
             (k=60)               (or cross-                      or LLM (JSON)   in the corpus
                                   encoder)
```

| Path | Role |
|---|---|
| `data/corpus/` | read-only corpus (`Doc_01.md`) + `MANIFEST.json` (SHA-256) |
| `data/PROVENANCE.md` | how the corpus was produced, what was excluded, stable id table |
| `src/schemas.py` | pydantic models: `TranscriptChunk`, `ControllerDecision`, `SubQuery`, `CorpusChunk`, `Claim`, `AnswerVersion`, `OutputRecord`, `TelemetryEvent` |
| `schemas/*.schema.json` | JSON Schemas exported from `src/schemas.py` (`make schema`) |
| `src/config.py` | all settings from environment variables (`.env.example`) |
| `src/corpus/` | `loader.py` (sections), `chunker.py` (section-aware chunks), `audit.py`, `build_index.py` |
| `src/retrieval/` | `bm25.py`, `dense.py` (embedder interface), `rrf.py`, `hybrid.py`, `rerank.py`, `factory.py` |
| `src/llm/client.py` | **the only place that talks to an LLM** (ollama / openai-compatible / mock) |
| `src/synthesis/` | `generator.py` (extractive or LLM), `citations.py`, `uncertainty.py` |
| `src/telemetry/` | JSONL event logger and event types |
| `src/baseline.py` | non-streaming reference pipeline |
| `src/stream/` | `simulator.py` (replay), `suppression.py` (presentation-only gate), `stability.py` (probe), `controller.py` (policy + LLM controller option) |
| `src/engine.py` | streaming engine: controller decisions, background retrieval dispatch, telemetry |
| `eval/` | dev scenarios, `run_eval.py`, `controller_eval.py`, `ablate_controller.py`. **Never imported by `src/`** (enforced by a test) |
| `src/decompose/`, `src/session/`, `src/retrieval/cache.py`, `src/synthesis/grounding.py` | placeholders for Phases 3-4 |

### Citations and ids

* Chunk id: `Doc_01 §4.1.2#1` (document, section, part). Stable across runs.
* Citation: `[Doc_01 §4.1.2]` in answer text, `"Doc_01 §4.1.2"` in `citations`.
* A citation is emitted only for evidence chunks the pipeline actually used. The citation
  check fails the request if any cited id does not exist in the corpus.
* If no retrieved chunk covers enough of the request (`PRISM_MIN_EVIDENCE_SCORE`), the answer
  is empty and `uncertainty` explains what is missing and asks for clarification.

## Configuration

Copy `.env.example` to `.env` (or export the variables). The defaults are CPU-light and need
no downloads:

| Setting | Default | Alternatives |
|---|---|---|
| `PRISM_DENSE_BACKEND` | `hashing` (numpy feature hashing, **not semantic**) | `sentence_transformers` (+ `PRISM_DENSE_MODEL`, default `BAAI/bge-small-en-v1.5`) |
| `PRISM_RERANK_BACKEND` | `lexical` (query-term coverage) | `cross_encoder` (+ `PRISM_RERANK_MODEL`), `none` |
| `PRISM_LLM_PROVIDER` | `none` (extractive answers, no LLM) | `ollama`, `openai_compatible` (+ `PRISM_LLM_MODEL`, `PRISM_LLM_BASE_URL`, `PRISM_LLM_API_KEY`) |
| `PRISM_RRF_K` / `PRISM_CANDIDATES` / `PRISM_TOP_K` | 60 / 20 / 5 | |
| `PRISM_MIN_EVIDENCE_SCORE` | 0.34 | not tuned; see report |
| `PRISM_CHUNK_MAX_TOKENS` / `PRISM_CHUNK_OVERLAP_TOKENS` | 400 / 40 | |
| `PRISM_CONTROLLER_MODE` | `rule_only` (tuned) | `rule_stability`, `llm` (needs an LLM) |
| `PRISM_STABILITY_THRESHOLD` / `PRISM_STABLE_CHUNKS` / `PRISM_MAX_PROVISIONAL` | 0.2 / 1 / 1 | tuned in step 2.7 |
| `PRISM_PROBE_K` / `PRISM_PROBE_MIN_CONTENT_TERMS` / `PRISM_PROBE_MIN_TOKENS` | 3 / 2 / 4 | tuned in step 2.7 |
| `PRISM_GATE_THRESHOLD` | 0.5 | suppression-gate classifier fallback |

## Teammate setup (infrastructure, not done in Phase 1)

Nothing in this list has been run or verified yet.

1. **Docker.** `Dockerfile` and `docker-compose.yml` are configuration only. They have never
   been built or run (no Docker daemon in the build environment). Run `docker compose up` on a
   clean machine and fix what breaks (gate G1). The default service needs no LLM.
2. **LLM.** Choose a model that fits CPU-only / ~4 GB RAM, then set `PRISM_LLM_PROVIDER`,
   `PRISM_LLM_MODEL` and `PRISM_LLM_BASE_URL`.
   * Local Ollama: `docker compose --profile llm up`, pull a small instruct model, set
     `PRISM_LLM_PROVIDER=ollama`. Pin the `ollama/ollama` image tag in the compose file.
   * Hosted / OpenAI-compatible endpoint: `PRISM_LLM_PROVIDER=openai_compatible`,
     `PRISM_LLM_BASE_URL=https://.../v1`, `PRISM_LLM_API_KEY=...` (never commit `.env`).
   * The HTTP request/response handling has only been tested against a local stub server.
     Run one real request and check that token counts arrive in `logs/telemetry.jsonl`.
   * Set `PRISM_COST_PER_1K_INPUT/OUTPUT` if the endpoint is billed.
   * Then run the missing Phase 2 ablation arm: `PRISM_LLM_PROVIDER=... make ablate`, which
     adds the measured `llm` controller row to `reports/PHASE_2_CONTROLLER_ABLATION.md`.
3. **Dense model / reranker (optional).** `make install-optional` installs
   sentence-transformers + faiss-cpu (pulls PyTorch). Verify RAM on the 4 GB target, pin the
   versions in `requirements-optional.txt`, then set `PRISM_DENSE_BACKEND=sentence_transformers`
   and/or `PRISM_RERANK_BACKEND=cross_encoder`. Dense embeddings are cached in `indexes/`.
4. **Lockfile.** `uv.lock` was generated with `uv lock` and covers the optional extras. Regenerate
   it with `make lock` after changing `pyproject.toml`. `requirements*.txt` are kept in sync by hand.

## Rules this repo enforces (from the problem statement, [Doc_01 §3])

* Corpus isolation: answers come only from `data/corpus/`; the extractive path quotes it verbatim.
* No hardcoded prompts, queries or answers in `src/`; dev scenarios live in `eval/` only.
* Every claim carries a citation that exists in the corpus, or the system emits uncertainty.
* No cross-session state (the baseline is stateless; the controller's state is reset per utterance).
* Presentation-only turns are suppressed and never reach the corpus.
* No multi-agent frameworks.
