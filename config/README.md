# config/

Four files. Two describe the model panel, two describe the tasks. Loaded via
`src/config_loader.py`.

## `panel.yaml`

The model roster. Top-level `experiment:` holds run-wide constants
(`reasoning_effort` — must stay identical across every model in a comparison
set, changing it invalidates cross-model comparisons). Each entry under
`models:` describes one model:

- `provider` — key into `src/adapters/PROVIDER_MAP` (`src/adapters/__init__.py`).
- `role` — `scored` (counts toward panel results), `anchor` (Gemma 4 — a
  stable low-cost reference point run alongside every comparison), or
  `judge` (Phase 2 legibility judge, not itself scored).
- `trace_exposure` — `raw` (full reasoning text available), `summarized`
  (provider returns a summary, not the raw trace — e.g. Claude), or
  `count_only` (only a token count, no text — e.g. OpenAI reasoning
  models). This is the harness's *expected* classification; the smoke test
  (`run.py --smoke`) verifies it empirically per run and flags mismatches.
- `model_id` — the direct-provider API model identifier (used only if that
  provider's key is present in `.env`).
- `openrouter_model_id` — the OpenRouter slug. Where reproducibility
  matters, this is a dated pin (`...-20260708` etc.) with a comment
  documenting how the pin was confirmed (live catalog listing, or — for
  models where OpenRouter never lists dated snapshots — a small number of
  real completion calls probing which suffixes return HTTP 200 vs 400).
  Re-verify pins against provider docs before a production run; they can
  and do go stale.
- `thinking_budget` — the reasoning token budget sent to models that accept
  one.

Read the per-row comments before trusting a row at face value — several
document open questions (e.g. whether a model's `reasoning_effort` levels
are behaviorally distinct, or whether a size estimate is an official spec
vs. a community estimate) that matter for interpreting that model's results.

## `pricing.yaml`

USD-per-million-token pricing per model, keyed identically to `panel.yaml`.
`snapshot_date` at the top records when this was last verified against
OpenRouter's live catalog — **re-verify before trusting absolute cost
figures from an old run**, and especially before running anything real:
prices change. Rows marked `# Verify:` were not independently confirmed at
snapshot time for that specific pricing dimension (e.g. cache pricing).

Cost formula (`src/cost.py`):
`input*p_in + cache_read*p_cache_read + cache_write*p_cache_write + (reasoning + output)*p_out`

## `prompts.yaml` — light suite (P1–P10)

Ten Danish prompts. Each entry has:

- `id`, `type`, `reasoning_load` (low/medium/high/very_high),
  `language_probe`, `carries_correctness` (whether `facit` is meaningful
  for this prompt or `null`).
- `prompt` — the only field sent to the model.
- `facit` — the answer key. `null` for open-ended prompts (P1, P2, P9, P10)
  that have no single correct answer; a string (sometimes with a short
  justification) for the six prompts that carry correctness (P3–P8).

## `prompts_multilang.yaml` — language-cost supplement (M1–M6)

Six tasks, each with `da`/`en`/`zh` variants expressing the same content
(English is the translation pivot; Chinese variants are machine-translated
and unverified by a fluent speaker — see the file's header). `facit` is
again the blind answer key, `null` for the two open-ended tasks (M1, M6).

## The facit contract

**`facit` must never reach a model.** Both `load_prompts()` and
`load_multilang_prompts()` in `src/config_loader.py` strip the `facit` key
before returning, and each asserts twice per call — once that `facit` was
present in the source (so a missing key is caught early, not silently
treated as "no answer"), and once that it does not survive the strip (so a
bug in the strip logic itself hard-stops the run instead of leaking).

If you add a new prompt: always include a `facit` key, even if it's
`null`. If you write code that reads these YAML files directly instead of
going through `config_loader.py`, you are responsible for stripping `facit`
yourself before it touches any outgoing request — don't bypass the loader
on the request path.

Grading code (`src/grader.py`, `src/heavy_grader.py`, `run_phase3.py`) is
the one place `facit` is legitimately used — always after the model's
answer has already been generated, never before or during.
