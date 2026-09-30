"""Presentation-only path (Phase 4). Minimal in step 4.6: re-render the current answer
unchanged, with no retrieval. Step 4.7 adds the transforms."""

from __future__ import annotations

from src.schemas import Claim
from src.synthesis.generator import render_answer


async def present(instruction: str, claims: list[Claim], client=None) -> tuple[str, str | None]:
    return render_answer(claims), None
