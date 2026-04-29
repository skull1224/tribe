# Repository Instructions

## Project Context

- This is the TRIBE v2 Python package for multimodal brain-response inference and training.
- The supported runtime is Python 3.11+.
- Main source code lives in `tribev2/`.
- Local helper scripts live in `scripts/`.
- Read `RUN_LOCAL.md` before changing local setup, inference, or transcription behavior.

## Using Codex And Claude Code Together

- Treat this file as the shared source of truth for AI coding agents in this repository.
- `CLAUDE.md` points Claude Code back to this file so both tools follow the same project rules.
- Work one task at a time. Before editing, check `git status --short` and avoid overwriting changes you did not make.
- Keep edits focused on the requested behavior. Do not reformat unrelated files.
- When handing work from one agent to the other, include the changed files, verification commands, and any skipped checks.
- Do not commit generated outputs, local virtualenvs, model caches, or downloaded checkpoints.

## Environment

- Create the base environment with:

  ```bash
  ./scripts/setup_env.sh
  ```

- This installs the package in editable mode into `./.venv`.
- If WhisperX is needed, use the separate environment described in `RUN_LOCAL.md`:

  ```bash
  ./scripts/setup_whisperx.sh
  ```

- Full text/audio/video inference may require Hugging Face authentication and access to `meta-llama/Llama-3.2-3B`.
- Prefer `--skip-text` for small smoke tests unless text behavior is the target of the change.

## Common Commands

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall tribev2 scripts
.venv/bin/python scripts/run_inference.py --audio /absolute/path/to/sample.wav --skip-text --device auto --extractor-device auto
```

Use the inference command only when suitable local media and model access are available. Note skipped large-download or gated-model checks in the final summary.

## Code Style

- Follow the existing Python style.
- `black` is configured with line length 88.
- `isort` uses the `black` profile.
- Prefer structured parsing and existing project helpers over ad hoc string handling.
- Add comments only when they clarify non-obvious behavior.

## Verification

- For narrow code changes, run the most targeted test or smoke check available.
- If no relevant tests exist, run:

  ```bash
  .venv/bin/python -m compileall tribev2 scripts
  ```

- For setup or documentation-only changes, verify by reading the changed files and checking `git status --short`.
