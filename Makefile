PYTHON ?= python3
Q ?= What are the technical evaluation gates and their target thresholds?

.PHONY: help install install-optional test audit index baseline eval replay controller ablate stream schema manifest lock clean

help:
	@echo "install           install pinned runtime + dev dependencies"
	@echo "install-optional  install model-based backends (torch; not tested in Phase 1)"
	@echo "test              run the test suite"
	@echo "audit             print corpus audit (docs, sections, chunk lengths, duplicates)"
	@echo "index             build BM25 + dense indexes and write indexes/chunks.jsonl"
	@echo "baseline          answer Q=\"...\" with the non-streaming baseline"
	@echo "eval / replay     run the baseline over eval/dev_scenarios and print metrics"
	@echo "controller        G2 controller metrics over eval/dev_scenarios (phase 2)"
	@echo "ablate            controller ablation + threshold grid -> reports/PHASE_2_CONTROLLER_ABLATION.md"
	@echo "stream            stream U=\"frag 1 | frag 2\" through the controller (phase 2 demo)"
	@echo "schema            regenerate schemas/*.schema.json from src/schemas.py"
	@echo "manifest          recompute data/corpus/MANIFEST.json hashes"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-optional:
	$(PYTHON) -m pip install -r requirements-optional.txt

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) -m src.corpus.audit

index:
	$(PYTHON) -m src.corpus.build_index

baseline:
	$(PYTHON) -m src.baseline --query "$(Q)"

eval:
	$(PYTHON) eval/run_eval.py --system baseline

controller:
	$(PYTHON) eval/run_eval.py --system controller_only --metrics g2

ablate:
	$(PYTHON) eval/ablate_controller.py --out reports/PHASE_2_CONTROLLER_ABLATION.md

U ?= Which gate covers | early retrieval | and how is it validated?
stream:
	$(PYTHON) -m src.engine --utterance "$(U)"

# Phase 1: the replay suite is the baseline over the dev scenarios.
# Phase 5 replaces this with the full streaming replay runner.
replay: eval

schema:
	$(PYTHON) scripts/export_schemas.py

manifest:
	$(PYTHON) scripts/corpus_manifest.py --write

lock:
	uv lock

clean:
	rm -rf indexes results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
