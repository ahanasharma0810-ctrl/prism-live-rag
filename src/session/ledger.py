"""Claim ledger (Phase 4). Minimal holder in step 4.1; completed in step 4.2."""

from __future__ import annotations


class ClaimLedger:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.claims: dict = {}
