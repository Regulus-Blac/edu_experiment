# Edu Experiment

This package orchestrates the Edu routing screen and task-level surface ablation without modifying the sibling DataMind checkout. Run commands from the repository root; all paths in `settings.json` and command arguments are repository-relative.

Use a Python interpreter with DataMind's dependencies installed (the local checkout used `.venv-edu/bin/python`; the examples below use `python3` as a placeholder).

## Setup

```bash
cp edu_experiment/settings.example.json edu_experiment/settings.json
# Fill in llm.api_key and embedding.api_key.
python3 -m edu_experiment init --settings edu_experiment/settings.json --reindex
```

`settings.json` is local secret material and should not be committed. The orchestrator converts it to DataMind's process-local nested settings and never writes raw keys to artifacts.

## Two phases

Run one task-specific FULL routing query per selected task. A task only enables its own `required_surfaces`.

```bash
python3 -m edu_experiment screen \
  --settings edu_experiment/settings.json \
  --tasks edu_sample/tasks/tasks_big.jsonl \
  --count 20
```

After reviewing `screen/results.jsonl`, choose task-level modes in a copied `ablation_plan.json` and continue:

```bash
python3 -m edu_experiment continue \
  --settings edu_experiment/settings.json \
  --run-id <run_id> \
  --plan edu_experiment/ablation_plan.json
# continue also writes final_report.json/.md and human_review_tasks.csv
python3 -m edu_experiment report \
  --settings edu_experiment/settings.json \
  --run-id <run_id>
```

`full` expands to the task's FULL condition and all strict non-empty subsets. `selected` runs only listed conditions, while `none` keeps the routing result only. Reports separate `fully_ablated`, `partially_ablated`, and `screen_only` tasks.

Normally `continue` requires the same settings fingerprint as the screen run. To switch API URL, key, protocol, or model, create a new configuration branch explicitly:

```bash
python3 -m edu_experiment continue \
  --settings edu_experiment/other_settings.json \
  --run-id <screen_run_id> \
  --new-run-id <new_run_id> \
  --plan edu_experiment/ablation_plan.json
```

Use `screen --dry-run` to validate selection and create a manifest without contacting an API. Use `screen --prepare-workspace` when the DataMind profile has not yet been imported; `--clean-workspace` is required to replace a non-empty profile.
