# Know When You Don't

`Know When You Don't` is an authoring repo for a metacognition benchmark suite built on Kaggle Benchmarks.

The core idea is simple:

- keep the **same underlying task pool**
- evaluate it under **different metacognitive protocols**
- score each family with a rubric that matches the behavior we are trying to measure

This repo is where we define task families, shared items, schemas, scoring logic, and Kaggle-ready notebooks.

## Current Families

The repo currently includes four families built on the same shared task set:

1. `boundaries_clarification`
   - natural boundary management
   - asks whether a model answers, clarifies, abstains, or challenges appropriately

2. `prospective_monitoring`
   - confidence before answering
   - asks the model to predict how likely it is to respond appropriately and correctly

3. `retrospective_monitoring`
   - confidence after answering
   - asks the model to assess whether its response was likely appropriate and correct

4. `self_correction`
   - revision after reflection
   - asks the model to review and improve its own response

## Shared Task Design

The current shared pool is centered on boundary management and includes five subtypes:

- `clarify_ambiguity`
- `clarify_missing_detail`
- `abstain_underdetermined`
- `challenge_false_premise`
- `answer_safe_control`

The working shared item set lives in:

- [items_current.yaml](/C:/Users/risha/Documents/New%20project/know-when-you-dont/task_families/boundaries_clarification/items_current.yaml)

Other families reuse that source through their `family.yaml` configuration.

## Judge Design

The current notebooks use a fixed judge handle from Kaggle Benchmarks:

- `kbench.judge_llm`

The judge path is intentionally lightweight:

- judge variants vote on behavior
- the notebooks parse judge output defensively instead of relying on brittle strict schema parsing
- if required judge fields are missing for a case, the notebook logs a warning and assigns that item a score of `0.0` instead of crashing

This makes large Kaggle runs much more stable.

## Repo Layout

- [task_families](/C:/Users/risha/Documents/New%20project/know-when-you-dont/task_families)
  - family configs and authored YAML items
- [datasets](/C:/Users/risha/Documents/New%20project/know-when-you-dont/datasets)
  - rendered JSONL datasets for Kaggle notebooks
- [notebooks](/C:/Users/risha/Documents/New%20project/know-when-you-dont/notebooks)
  - generated Kaggle-ready notebooks
- [src/know_when_you_dont](/C:/Users/risha/Documents/New%20project/know-when-you-dont/src/know_when_you_dont)
  - schemas, family loading, and notebook rendering logic

## Workflow

1. Author or refine items in a family YAML file.
2. Render the dataset and notebook for that family.
3. Upload or sync the repo-backed dataset to Kaggle.
4. Run the generated notebook on Kaggle.
5. Inspect row-level outputs before trusting aggregate scores.

For shared-item families, update the source item file once and rerender each family.

## Local Commands

Use the package with `PYTHONPATH=src` if it is not installed in the environment.

```powershell
$env:PYTHONPATH='src'
python -m know_when_you_dont.render boundaries_clarification
python -m know_when_you_dont.render prospective_monitoring
python -m know_when_you_dont.render retrospective_monitoring
python -m know_when_you_dont.render self_correction
```

## Current Status

- Boundary Management is the most mature family.
- The shared prompt bank has been expanded into a larger current item set.
- Additional families now render from the same shared pool with family-specific prompting and scoring.
- Kaggle runs are producing differentiated scores across models, which is exactly what we want at this stage.

## Notes

- This repo is an authoring and rendering repo, not a standalone benchmark runner.
- Kaggle is the source of truth for actual task execution and leaderboard behavior.
- Judge-model pinning can be added separately if Kaggle exposes a stable explicit judge selection path.
