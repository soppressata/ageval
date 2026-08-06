# Contributing to ageval

Thanks for your interest! `ageval` is a small, focused, stdlib-only harness and we
welcome bug reports, fixes, and new adapters/scorers.

## Filing issues

Open an issue at [github.com/soppressata/ageval/issues](https://github.com/soppressata/ageval/issues).
Include:

- A clear title and description.
- Steps to reproduce (or a minimal failing task suite / command).
- Expected vs actual behavior.

## Pull-request process

1. Fork the repository and create a branch off `master`.
2. Make your changes — keep them scoped to the feature or fix.
3. Run `python3 -m pytest -q` — the suite must pass.
4. Open the PR with a short description of the change and the motivation.

## Rules

- **`SPEC.md` is law.** Never rename or remove fields it defines. If you think the
  spec is wrong, open an issue — don't "fix" it yourself.
- **Standard library only** in `ageval/`. No `pip install` of third-party packages,
  no `asyncio`. Use `urllib.request` for HTTP.
- **File ownership** matters. `AGENTS.md` describes which engineer owns which files.
  Respect it — don't reformat or "improve" code someone else is working on.
- Every module starts with `from __future__ import annotations`.
- Type hints on every public function. Concise docstrings.
- Fail loudly in constructors; fail softly (return an error-carrying object) at
  runtime boundaries — see `SPEC.md` sections 5 and 6.
- Never use bare `except:`; use `except Exception as e:`.

## Adding an agent or scorer

Decorate your class with `@register_agent("name")` or `@register_scorer("name")`
(see `ageval/core.py`). Ensure `predict` / `score` never raise — catch every
exception and return an error-carrying result.

Happy hacking!
