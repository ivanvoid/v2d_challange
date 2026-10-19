# Experiment ledger — Track 1 submissions

One row per Kaggle submission. Change ONE thing per submission (about 5 submissions per week).
Kaggle scores: lower is better for all five. Local columns come from
`track1/eval_local.py` run on the exact parquet that was uploaded (scored frames only):

```bash
.venv-submit/bin/python track1/eval_local.py extract --parquet <file> --out logs/eval/<name>
CUDA_VISIBLE_DEVICES=0 /opt/venvs/sam3d_body/bin/python track1/eval_local.py proxies --out logs/eval/<name>
/opt/venvs/sam3d_body/bin/python track1/eval_local.py compare logs/eval/<before> logs/eval/<name>
```

Local measurements (no reference needed): `PEN` is the scorer's PEN computed locally (512 sampled MHR
hand vertices, alignment scale 1); `gap` = median over scored frames of the closest hand-object
distance (scored frames are contact frames, so this should be ~0); `contact` = fraction of scored
frames with gap < 2 cm; `accH`/`accO` = our own mean |second difference| of the 22 scored joints /
the object translation per scored stretch (jitter, not the error).

| # | Date | Name | Commit | Change (only this) | CD-H | CD-O | ACC-H | ACC-O | PEN | local PEN | gap cm | contact | accH | accO | Files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-10-15 | baseline | `20b2d5c` | stages 1–8 as in HANDOFF.md | 15.716 | 37.420 | 0.1745 | 0.1274 | 0.1007 | 0.0966 | 12.38 | 0.454 | 0.131 | 0.115 | `submission/track1.parquet` |
| 1 | 2026-10-19 | p1_global | `899c80e` | episodes 0 1 8 14 25 27 28 29: stage-8 scale variant "none" → "global" (object translation and size × s_global 1.07–1.16) | | | | | | 0.0850 | 16.06 | 0.417 | 0.131 | 0.118 | `experiments/p1_global/track1.parquet` |

## Notes per experiment

**1 · p1_global** — question: should the object live in the body's depth scale ("global") or in
MoGe's ("none")? The baseline chose by the smallest hand–object gap over the whole first..last
range, which includes non-contact frames. Projective consistency and CARI4D (which anchors
monocular depth to the human body, `models/CARI4D/prep/align_monod2hum.py`) both argue for the body
scale. Read CD-O first. If CD-O improves, follow-ups: "global" everywhere vs per-frame (CARI4D
aligns per frame with a robust scale + shift); choose the variant on scored (contact) frames only.

Local prediction (`eval_local.py compare logs/eval/baseline logs/eval/p1_global`): the two local
signals disagree, which is what makes this submission informative.
* contact on scored frames gets WORSE on most changed episodes (28: gap 0.3 → 9.7 cm, contact
  58 → 31 %; 29: 0.0 → 3.3 cm; 8: 4.2 → 5.9 cm; 25: 0.0 → 2.1 cm); better only on 14 (12.3 → 8.7 cm)
* object size agrees BETTER across the 3 videos of each object (largest ratio 1.28 → 1.21; carts
  0.88/0.72/0.80 → 0.96/0.82/0.87 m, hoop 0.72/0.72/0.80 → 0.83/0.84/0.80 m)
* local PEN 0.0966 → 0.0850
If CD-O improves: trust the body's scale (and size consistency); the hands are what is misplaced.
If CD-O gets worse: the hand-contact signal is the better guide for object placement.
