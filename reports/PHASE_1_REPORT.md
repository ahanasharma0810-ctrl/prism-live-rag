# Phase 1 report

_Status: in progress. This file is completed at the end of Phase 1._

## Intake (step 1.1)

| Question | Answer (source) |
|---|---|
| Corpus location and format | No separate corpus was supplied. The owner instructed that the initial corpus is built **only** from `Theme_4_Guide_RAG.pdf`. It is transcribed to `data/corpus/Doc_01.md` (Markdown with explicit `[§N]` section headers). See `data/PROVENANCE.md`. |
| Allowed models | Not fixed yet. Owner: "do not require a large local model"; the final LLM infrastructure is decided by a teammate. The playbook suggests Ollama with a local instruct model, `BAAI/bge-small-en-v1.5`, and `cross-encoder/ms-marco-MiniLM-L-6-v2`. None of them were downloaded or verified in Phase 1. |
| Hardware | CPU-only, about 4 GB RAM (owner). |
| Rules beyond the PDF | None supplied. The hard constraints are PDF §3 and the playbook's standing rules. |
| Infrastructure | Docker, WSL, Ollama and cloud are **out of scope** for this phase (owner). Dockerfile and compose are configuration only. |
