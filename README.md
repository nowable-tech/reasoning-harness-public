# Reasoning-Economy Benchmark Harness

A single-turn benchmark harness for measuring how frontier LLMs spend
reasoning tokens — not just whether they get the right answer, but how much
thinking (and how much money) it costs them to get there.

Built for the Nowable essay series on the economics of LLM reasoning
(link: TBD).

## What this is

- **12+1 frontier models**: 12 scored models (DeepSeek V4, GLM 5.2, Kimi K2.7,
  Kimi K3, GPT-5.5, GPT-5.6-Sol, Claude Sonnet 4.6, Claude Opus 4.8, Claude
  Opus 5, Claude Fable 5, Mistral Medium 3.5, Inkling) plus 1 anchor model
  (Gemma 4) used as a stable low-cost reference point across runs. See
  `config/panel.yaml`.
- **13-task set**: 10 Danish light-suite prompts (P1–P10, one pass each) + 3
  heavy tasks (H1–H3: code, finance-calculation, finance-interpretation, run
  at 5 passes per condition) — this is the set every scored model runs in
  the main evaluation. A separate 6-task multilingual supplement (M1–M6,
  Danish/English/Chinese) adds language *options* for an independent
  language-cost experiment: three of the six (M2/M3/M5) are the same
  content as three of the light-suite prompts above, just in translation,
  not new tasks. Counting every distinct task-content across both sets
  gives 16 — that figure describes content variants, not the size of the
  main 13-task evaluation. See [Task set](#task-set).
- **Tools / no-tools**: every scored model runs both a plain baseline and a
  tool-offload condition (`python_exec` sandbox always available,
  `web_search` if `SEARCH_API_KEY` is set), to measure whether reasoning
  tokens get replaced by tool calls or just added on top.
- **5-pass heavy cells**: heavy-task cells are repeated (default 5 passes) to
  distinguish genuine model behavior from run-to-run variance.
- **Single-turn only**: no agentic loops, no multi-turn conversations, no
  memory across calls. Every measurement is one prompt → one response.

The harness measures reasoning-token volume, cost, correctness, tool-use
propensity, and (for a subset of raw-trace models) reasoning-trace
legibility, and it does all of this per-model, per-task, per-condition, so
you can see where the reasoning spend actually goes instead of only a single
aggregate cost number.

## Quickstart

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Fill in the keys you have. You do not need every key — each entrypoint skips
(or, where a key is strictly required, fails early with a clear error)
models whose credentials are absent. At minimum:

- **OpenRouter** (`OPENROUTER_API_KEY`) covers most scored models through a
  single gateway key and is the easiest way to get started.
- **Direct provider keys** (`ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`,
  `OPENAI_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`) are used instead of
  OpenRouter when present, for models where a direct route is offered.
- **Gemma 4** runs via OpenRouter — no separate key needed beyond
  `OPENROUTER_API_KEY`.
- Phase 2 (legibility judging) needs `OPENROUTER_API_KEY` regardless of
  which provider keys you have for the scored models.
- `SEARCH_API_KEY` is optional — without it, `web_search` is simply dropped
  from the tool list; `python_exec` is unaffected.

### Run

Smoke test first, always — one call per available model, confirms
credentials, model IDs, and (for raw-trace models) that the reasoning trace
actually comes through:

```bash
python3 run.py --smoke
```

Light suite (10 prompts × all scored+anchor models, one pass each):

```bash
python3 run.py --full
```

Heavy suite (3 tasks × baseline/tool-invited × all models × 5 passes):

```bash
python3 run.py --heavy
```

Variance repro (repeat the light suite `--passes` times to measure
run-to-run spread):

```bash
python3 run.py --variance --passes 2
```

Grading + judges (run in this order):

```bash
python3 run_phase3.py                 # Phase 3: correctness (P3–P8 facit grading)
python3 run.py --validate-judges      # Phase 2 gate: judges on Gemma traces only
python3 run.py --judge                # Phase 2 full: judges on all raw-trace models
```

`--validate-judges` intentionally stops after generating an HTML review —
read it and confirm judge quality before running `--judge` on the full set.
Likewise `run_phase3.py` stops after the P3/P4 (legal) HTML review — those
two are graded by an LLM and worth a manual read before trusting the
verdicts.

Other entrypoints: `run.py --tools` (tool-offload experiment), `run.py
--langcost` / `--langcost-full` (language-cost experiment over
`config/prompts_multilang.yaml`), `run_phase2.py` (anchored-rubric variant
of the Phase 2 judge). Run any entrypoint with `--help` for its full flag
list.

## Task set

- **Light suite** (`config/prompts.yaml`, P1–P10): Danish natural-language,
  legal reasoning, math, logic, code-structure, code-bug, and open-analysis
  prompts. One pass per model. `facit` (the answer key) is present for the
  six prompts that carry correctness (P3–P8) and is stripped before any
  request is sent — see [config/README.md](config/README.md).
- **Heavy suite** (`src/heavy_tasks.py`): three tasks from established,
  external, single-turn benchmarks, downloaded once and cached under
  `data/heavy/` (gitignored):
  - **code** — HumanEval task `HumanEval/94`. HumanEval is © OpenAI,
    [MIT-licensed](https://github.com/openai/human-eval/blob/master/LICENSE);
    redistributed here under that license.
  - **finance_calc**, **finance_interp** — [FinQA](https://github.com/czyssrs/FinQA)
    (Chen et al., EMNLP 2021) records `CDNS/2015/page_30.pdf-3` and
    `AMAT/2013/page_37.pdf-2`. FinQA is CC BY 4.0-licensed; redistributed
    here under that license with attribution.
  - `finance_interp`'s facit encodes a documented correction: FinQA's own
    `qa.answer` string for that record is missing a trailing zero (a known
    dataset annotation issue), and the question is separately ambiguous
    between two defensible readings — both are accepted. See the comment in
    `src/heavy_tasks.py::load_heavy_tasks`.
- **Language-cost supplement** (`config/prompts_multilang.yaml`, M1–M6):
  the same content in Danish/English/Chinese, used to isolate how much of a
  model's reasoning spend is language-dependent rather than task-dependent.
  The Chinese variants are machine-translated (English pivot) and are **not**
  verified by a fluent Chinese speaker — see the caveat at the top of that
  file before trusting zh-language conclusions.

## Conventions

These are load-bearing and easy to get subtly wrong when re-deriving results
from the raw JSONL. Follow them exactly:

1. **Heavy numbers = median of per-task medians, baseline condition; light
   numbers = a single flat median — the two suites are NOT aggregated the
   same way.** Heavy: when you see a single reported heavy-suite number for
   a model, it is the median across that model's three tasks' own per-task
   medians (each task's median taken across its repeated passes), not a
   median across every raw row pooled together. Light: there is no
   task-grouping to nest within, so a single reported light-suite number is
   the flat median across that model's prompt-level rows directly — 10 rows
   (one pass each) in the main suite, 20 rows (10 prompts × 2 passes) in
   `--variance`. Applying the heavy suite's two-stage median to light data
   (or vice versa) will silently produce a different, wrong number.
2. **Pass-level dedup: the latest run wins per cell.** If a task/model/
   condition cell was re-run (e.g. a recap correcting a truncation or
   token-cap artifact), the later run's rows replace the earlier run's rows
   for that exact cell — never pool both.
3. **Three run-modes, not two.** `--full` (light suite, 1 pass per prompt),
   `--heavy` (5 passes per cell), and `--variance` — a dedicated repro run,
   separate from both: 2 passes on the full 10-prompt light suite (baseline
   only) plus 5 passes on the heavy suite's cells, run independently of the
   main `--full`/`--heavy` runs. Variance-suite rows are not the same rows
   as the main light/heavy runs and should not be silently pooled with
   them — see `run.py`'s `run_variance()`.
4. **Efficiency = correct answers / actual summed `cost_usd`, never
   median-cost × row-count.** Cost distributions are right-skewed: a
   handful of expensive outlier rows pull the true total well above what
   `median × n` predicts. This was confirmed empirically on
   `mistral_medium_3_5`: `median × n` predicted $0.56 in total spend against
   an actual $0.84 — a 34% error. See `src/metrics.py::correct_per_dollar`,
   the single source of truth for this metric — use it rather than
   re-deriving the ratio inline.
5. **Light-suite numbers are n=1.** Each light prompt runs once per model
   in the main `--full` suite (see convention 3 above for `--variance`'s 2
   passes). Read main-suite light numbers as an indicative spread across
   the panel, not as a statistically robust per-model estimate — do not
   compute variance or confidence intervals on n=1 data.
6. **Closed-model reasoning-token counting.** Reasoning tokens are read from
   the provider's own billed usage metadata where it reports one. Where the
   API reports no count at all (Anthropic on some routes), the harness falls
   back to a proportional estimate — reasoning share of the response, split
   from the summarized thinking text length versus total output tokens. This
   is never silent: every row persists which path produced its number as
   `reasoning_source` (`"api"` or `"text_estimate"`). Check this field before
   treating a closed-model reasoning-token figure as directly comparable to
   an open-model one measured from real usage metadata.

## Caveats

- **Single-turn scope.** This harness measures one-shot reasoning economy.
  It says nothing about agentic loops, multi-turn conversations, or
  memory/context accumulation across calls — those are different cost
  regimes entirely.
- **Sampling is not pinned across providers.** Temperature, top-p, and
  similar sampling parameters are each provider's own defaults, not
  harmonized across the panel. Cross-model comparisons hold
  `reasoning_effort`/`thinking_budget` constant (see `config/panel.yaml`)
  but not low-level sampling.
- **Open models via OpenRouter are a backend lottery.** OpenRouter may route
  a given open-weight model across multiple hosting backends (see the
  `served_by`-logging convention in `config/panel.yaml` and
  `src/adapters/base.py`). Two calls to the "same" model can be served by
  different infrastructure with different latency and, occasionally,
  different behavior. Every result row logs `served_by` — do not silently
  pool across backends without checking it.
- **Prices are snapshots**, dated in `config/pricing.yaml`
  (`snapshot_date`). Re-verify against each provider's live pricing before
  trusting absolute cost figures from an old run — several rows are
  annotated `# Verify:` for pricing dimensions that were not independently
  confirmed at snapshot time.
- **Model pinning is inconsistent by necessity.** Some `openrouter_model_id`
  values are dated pins (for exact reproducibility); others are undated
  because OpenRouter never listed a dated snapshot for that model. See the
  per-row comments in `config/panel.yaml` — they document exactly how each
  pin was confirmed (or why it couldn't be).

## Cost warning

**A full panel run costs real money.** Concrete measured data points from
single-addition runs on this panel (light + heavy phases, ~40–80 calls):
roughly **$0.10–$1.10** per run, depending on which models were included
(premium models like Opus/Fable/GPT-5.6-Sol cost substantially more per call
than the open-weight models). A full run across the entire 12+1-model panel
and every phase (`--full`, `--heavy`, `--variance`, `--tools`, `--langcost`)
multiplies that by roughly 13 models and several passes each — budget on the
order of **tens of dollars**, not cents, before running everything
unattended.

`run.py`'s own phases (`--full`, `--heavy`, `--variance`, `--tools`) have
**no built-in spend cap** — they print a running cost total only at the end.
The standalone one-off scripts (`run_opus5_panel.py`,
`run_langcost_k3_inkling.py`, `run_auto_router.py`) each implement a
`PRICE_CAP_USD` guard that stops (not crashes) once cumulative spend crosses
a threshold — use that pattern if you add your own unattended runs.
Recommended before any real run: `--smoke` first (one call per model), then
`--pilot` (a small prompt subset) before `--full`, and keep an eye on the
per-model cost breakdown each phase prints at the end.

## Reproducing / extending

- **Recompute headline figures from raw data**: `scripts/compute_findings.py
  <results_dir>` reads a results tree (this repo's own `results/` if you've
  run the harness yourself, or a data bundle someone shared with you in this
  same row schema) and deterministically prints per-model reasoning medians
  (light + heavy), reasoning share, correct/actual-$, tool grab-rate,
  variance quartiles, and a `trace_status` inventory — applying every
  convention above exactly. No API calls, no dependency on any curated or
  hand-reviewed dataset. If a phase subdirectory (e.g. `tools/`, `variance/`)
  is absent from the data you point it at, that section just reports no
  data rather than failing.
- **Add a model**: add a block to `config/panel.yaml` (provider, role,
  `trace_exposure`, model IDs, `thinking_budget`) and a matching block to
  `config/pricing.yaml`. If the provider isn't already in
  `src/adapters/PROVIDER_MAP` (`src/adapters/__init__.py`), add an adapter
  module following the existing `*_adapter.py` pattern — `BaseAdapter` in
  `src/adapters/base.py` documents the required interface, including the
  construction-time credential check.
- **Add a task**: append an entry to `config/prompts.yaml` (light suite) or
  `config/prompts_multilang.yaml` (multilingual supplement), following the
  existing `id`/`facit` structure. See
  [config/README.md](config/README.md) for the facit contract — it must
  never enter the request path.
- **Add a judge rubric**: `src/judge.py` holds the default legibility rubric;
  `src/judge_rubric.py` holds an alternate ("anchored") rubric used by
  `run_phase2.py`. Follow the existing floor-principle / counts-vs-does-not
  structure if you add a new one — the redundancy scale is only meaningful
  if judges are anchored against the same reference examples.

## License and citation

Code and configuration in this repository are licensed under the [MIT
License](LICENSE), © 2026 Lars Harder / Nowable — with the exception of
the HumanEval task content (© OpenAI, MIT-licensed, redistributed under its
own license) and the FinQA task content (© Chen et al., EMNLP 2021,
CC BY 4.0-licensed, redistributed under its own license).

If you use this harness or its results, please cite:

> Lars Harder / Nowable, "Reasoning-Economy Benchmark Harness" (2026).
> https://github.com/nowable-tech/reasoning-harness-public
