"""Standalone auth shim for the isolated aim-ml-ops repo.

The production dashboard resolves the user from an ALB-injected OIDC JWT and
enforces per-page authorization against a Google Sheet. None of that belongs in
this Milestone 1 deliverable, which runs the revenue page locally against
pipeline artifacts on disk. So `page` is a passthrough decorator: it preserves
the `@page(SLUG)` call site in `src/pages/revenue.py` verbatim (no source edit
needed) while doing no authorization.

If this page is ever folded back into the full dashboard, delete this shim and
import the real `src.auth.page` instead.
"""

from __future__ import annotations

from collections.abc import Callable


def page(slug: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """No-op stand-in for the dashboard's authz/audit page decorator."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        return fn

    return decorator
