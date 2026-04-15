# Know When You Don't

`Know When You Don't` is a local authoring repo for Kaggle Benchmarks tasks that evaluate metacognitive boundaries.

The repo is designed around task families, not around a local benchmark runtime. We author task items, schemas, scoring rules, and Kaggle-ready notebook code here, then publish tasks from Kaggle notebooks and combine them into a benchmark in the Kaggle UI.

## V1 Family

The first task family is `boundaries_clarification`, which measures whether a model:

- detects ambiguity
- detects insufficient information
- asks a targeted clarification instead of guessing

## Workflow

1. Author or refine items in `task_families/<family>/items.yaml`.
2. Validate the family locally.
3. Render a Kaggle notebook for the family.
4. Open the rendered notebook in Kaggle, run it, and publish the task.

## Commands

```powershell
python scripts\validate_family.py boundaries_clarification
python scripts\render_family.py boundaries_clarification
python scripts\preview_family.py boundaries_clarification
```

## Reference Repo

The cloned `kaggle-benchmarks-ref` repo in the parent workspace is used as a reference only. This repo does not vendor or fork it.
