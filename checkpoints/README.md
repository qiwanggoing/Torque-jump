# Checkpoints

Three checkpoints are committed here on purpose: a GitHub **fork copies repository contents but not
Release assets**, so putting them in the repo is what makes them reach whoever forks this. Total
~23 MB; GitHub's hard per-file limit is 100 MB.

| File | What it is | Actor input |
|---|---|---|
| `refprev2_25k_s1_model_23500.pt` | current best (meanHit **0.954** under mass DR) | **1060** |
| `sstill6_s1_model_19500.pt` | the `stand_still` working point | **1024** |
| `inertiafix_s1_model_14700.pt` | **paper baseline**, and source of `commanded_ref_14700.npz` | **1012** |

**A checkpoint will not load unless the observation geometry matches.** The exact exports each one
needs, the measurements behind each number, and the two structural flaws in the reference table are
in [`../RUNBOOK.md`](../RUNBOOK.md) §5. Read [`../RUNBOOK.md`](../RUNBOOK.md) §1 trap 1 before
evaluating any of them: without `PD_OFF_PHASE=all` you are handing the policy back PD it never
trained with, which measurably fabricates differences between runs.
