"""Repository layout and the eval/ isolation rule (playbook standing rule 7)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "Dockerfile", "docker-compose.yml", "Makefile", "README.md", "pyproject.toml",
    "requirements.txt", ".gitignore", ".env.example",
    "data/corpus", "data/PROVENANCE.md", "eval/dev_scenarios",
    "reports", "logs", "schemas", "scripts",
    "src/config.py", "src/retrieval/cache.py",
    "src/stream/simulator.py", "src/stream/controller.py", "src/stream/stability.py",
    "src/decompose/decomposer.py", "src/decompose/dedupe.py",
    "src/session/store.py", "src/session/ledger.py", "src/session/delta.py",
    "src/synthesis/grounding.py", "src/engine.py",
]


def test_required_layout_exists():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_src_never_imports_eval():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for mod in _imported_modules(path):
            if mod == "eval" or mod.startswith("eval."):
                offenders.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not offenders, offenders


def test_all_llm_calls_live_in_llm_client():
    """No module other than src/llm/client.py may talk HTTP or import an LLM SDK."""
    banned = ("urllib.request", "http.client", "requests", "httpx", "openai", "anthropic", "ollama")
    allowed = ROOT / "src" / "llm" / "client.py"
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path == allowed:
            continue
        for mod in _imported_modules(path):
            if mod in banned or any(mod.startswith(b + ".") for b in banned):
                offenders.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not offenders, offenders
