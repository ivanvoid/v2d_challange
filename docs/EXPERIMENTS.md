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
| 1 | 2026-10-19 | p1_global | `899c80e` | episodes 0 1 8 14 25 27 28 29: stage-8 scale variant "none" → "global" (object translation and size × s_global 1.07–1.16) | 15.716 | **43.294** | 0.1745 | 0.1306 | 0.0889 | 0.0850 | 16.06 | 0.417 | 0.131 | 0.118 | `experiments/p1_global/track1.parquet` |
| 2 | 2026-10-22 | stage67_v2 | `8873d07` | stages 5e-8 re-run: new 5e meshes (iron forced to ep 6 frame 723), new stage 6 tracker (re-find, states), stage 7 hidden frames interpolated, stage 8 fixed/follow scale | 15.716 | 30.956 | 0.1742 | 0.1235 | 0.0451 | 0.0555 | 4.65 | 0.489 | 0.130 | 0.112 | `experiments/stage67_v2/track1.parquet` |
| 3 | 2026-10-27 | penfix_smooth | `fd79f96` | stage-8-only bounded smooth object translation on foam episodes 15–17; scores reported by user, receipt/upload hash pending | 15.9737* | 30.9978 | 0.17421 | 0.12355 | 0.03349 | 0.0396 | 4.70 | 0.487 | 0.130 | 0.112 | `experiments/penfix_smooth/track1.parquet` |
| 6 | 2026-10-28 | phys_v1 | `7ad2b63` | object_translation only: lam_t 150 on all moving episodes (depth placement pinned), PEN push-out where it passes fit+holdout, stools static; control = stool_static_plus_smooth_t150 | 15.9737 | 31.04288 | 0.17421 | **0.10316** | **0.02742** | 0.0321 | 4.79 | 0.482 | 0.130 | 0.086 | `experiments/phys_v1/track1.parquet` |
| 7 | 2026-10-28 | phys_v2 | `35fa94d` | phys_v1 + body smoothing sigma 4 (smooth_body.py) + PEN squeeze (push-out <= 5 cm, foam 15/16 scale x0.92) | **15.85675** | 31.04884 | **0.14125** | 0.10539 | **0.01246** | 0.0135 | 4.92 | 0.479 | 0.065 | 0.089 | `experiments/phys_v2/track1.parquet` |
| 8 | 2026-10-28 | phys_v3 | `23468f3` | phys_v2 + object lam_t 1000 + object depth x1.04 (not 7 8 15-17) + PEN squeeze redo; tuned on the Track 2 local test set | 15.85675 | **37.06158** | 0.14125 | **0.08468** | 0.01517 | 0.0191 | 6.87 | 0.459 | 0.065 | 0.058 | `experiments/phys_v3/track1.parquet` |
| 9 | 2026-10-28 | phys_v4 | `da54a17` | phys_v3 without the depth x1.04 (lam 1000 + PEN squeeze at production placement) | 15.85675 | **30.87113** | 0.14125 | **0.08382** | 0.01499 | 0.0179 | 4.97 | 0.477 | 0.065 | 0.056 | `experiments/phys_v4/track1.parquet` |
| 10 | 2026-10-29 | phys_v5 | `af4895c` | phys_v4 + second PEN push-out pass (cap 7 cm) on 11 episodes | pending | pending | pending | pending | pending | 0.0089 | 5.03 | 0.474 | 0.065 | 0.057 | `experiments/phys_v5/track1.parquet` |
| 11 | 2026-10-29 | phys_v6 | `f496c69` | phys_v4 + hand-only depenetration (H4; body pose of hands/arms only, objects identical) | 15.85531 | **30.78576** | 0.14129 | 0.08383 | **0.00908** | 0.0096 | 5.02 | 0.477 | 0.065 | 0.056 | `experiments/phys_v6/track1.parquet` |
| 12 | 2026-10-29 | phys_v7 | `363831d` | phys_v6 + ep0 hoop at the feet (x1.193) + ep1 hoop from image ellipse + desk locks 10/11 | 15.85531 | **27.91838** | 0.14129 | 0.08385 | 0.00921 | 0.0097 | 5.80 | 0.474 | 0.065 | 0.057 | `experiments/phys_v7/track1.parquet` |

## Pre-registered hypotheses (written 2026-10-29, BEFORE running the tests)

Control: phys_v4 (Kaggle CD-H 15.85675, CD-O 30.87113, ACC-H 0.14125, ACC-O 0.08382, PEN 0.01499).
Each result is recorded under its hypothesis as PASS / FAIL against the numbers below; no
re-interpretation after the fact.

* **H1 — test-set alignment.** The test set's "objects 4 % too close" came from aligning by the
  pelvis path over time, which the scorer does not do. Test: redo the depth-factor sweep with a
  scorer-like alignment (rigid, once, on the first scored frame, from the root pose). Prediction:
  best factor 1.00 +- 0.01. FAIL if the best factor stays >= 1.03 (then Track 1 / Track 2 differ
  and test-set placement results stay banned).
* **H-depth — the depth model is not the bottleneck.** On test-set frames where the object is
  visible, MoGe's object-surface depth (median over our object mask), expressed in our body's
  frame, is within 5 cm of the reference surface depth over the same pixels (median |error| per
  episode). FAIL if > 10 cm: then a better depth model would help.
  * **Result 2026-10-29: INCONCLUSIVE — and the test set cannot settle placement at all.** Best factor
    under the first-frame emulation: per episode 0.98 / 1.08 / 0.97 / 1.04 / 1.04 (mean 1.02); under
    the trajectory alignment 1.03-1.05. The emulation aligns on the root pose of one frame (no
    reference body surface exists) and is far noisier than the scorer's whole-body alignment: CD-O
    35-135 cm per episode vs Kaggle's 31 cm mean. The fixed convention between our body frame and
    the reference root also varies 4-14 deg per frame. Conclusion: neither alignment reproduces
    the scorer, so test-set object-placement results stay banned; smoothing results (validated
    twice on Kaggle) remain usable. Plot: v2d_t2eval/diag/h1_depth_factor.png (`track1/t2_diag.py`).
* **H-depth result 2026-10-29: FAIL as stated, but confounded.** MoGe object-surface depth over our
  mask vs the reference, in our body frame: median |error| 24.1 cm (signed -21.2: closer). Our
  final placement (contact-chosen k): 10.7 cm (signed -8.3). Two caveats: (1) "our body frame"
  carries the registration scale 0.957, i.e. ~4.5 % (about -15 cm at 3.3 m) of the MoGe error is
  the known body-vs-MoGe scale disagreement, not MoGe; (2) errors drift over time within an
  episode (desk ep 10: 0 -> -40 cm), which no per-episode constant can fix. So per-frame depth
  consistency between body and object matters more than the depth model; our contact placement
  already removes about half of MoGe's error. Plot: v2d_t2eval/diag/hdepth_per_frame.png.
* **H4 — PEN is a hand-pose problem.** Adjusting only MHR wrist/finger parameters (objects =
  phys_v4) reaches local PEN < 0.005 with mean hand-point movement < 1 cm, contact not lower than
  phys_v4 - 0.01. Kaggle: PEN <= 0.005, CD-O 30.87 +- 0.05, CD-H <= 15.87, ACC-H <= 0.1445.
  FAIL if PEN > 0.006 or CD-O > 30.95.
  * Pilot on foam ep 16 (`track1/hand_depen.py`, local, 2026-10-29): fingers+wrist, w_reg 1: PEN
    0.1289 -> 0.0010, hand move mean 0.92 cm, but the overlay showed fingers flipped upward out of the
    block (`experiments/hand_pilot/plots/episode_000016_hands.png`). w_reg 10: same PEN, mean move
    0.69 cm, parameter change p99 0.25 rad (chosen). With arm parameters free: mean move 5.2 cm,
    contact 0.61 -> 0.50 — rejected (moving the arm swings the whole hand).
  * Visual review (user, 2026-10-29): the finger-only fix flips fingers out of the object — the
    "before" hands match the video better. Variant with arm + a 2D anchor (hand vertices keep their
    image position): realistic in the image, but the top-down view showed the solver hiding a hand
    26 cm deeper behind the desk (ep 9); mean hand move 0.8-2.5 cm > the 1 cm gate. Final variant:
    arm + 2D anchor + 3D motion cap 2 cm (`--arm --w_reg 10 --w_reg_arm 2 --w_2d 1 --w_move 3e4
    --move_tol_cm 2`): foam 16 PEN 0.129 -> 0.036, desk 9 0.030 -> 0.009, p99 hand move < 2.8 cm,
    hands keep their shape (`experiments/hand_pilot_cap/plots/`). Penetration deeper than ~2 cm
    is left in place on purpose: it is a hand/object placement error, not a pose error.
* **Assumption falsified (2026-10-29, review sheet `experiments/review/first_frames.png`):
  "scored frames are hand-contact frames" (stage8_scale.py docstring) is not true at the start of
  the span.** On the first scored frame our hands are 15-62 cm from the object in most episodes,
  and in the video the person is often still reaching (iron 7, paint roller 19, stool 21). 15
  episodes start at exactly video frame 90, which looks like a fixed 3 s cut, not contact onset.
  Consequences: (1) stage 8's contact-based depth choice also scores non-contact frames, a
  plausible source of its bias; (2) Fix 2 ("bring the object to the hands on every scored frame")
  would be wrong on approach frames and needs a contact detector before use.
  * Two more fixes before shipping (gates caught them): (1) arm/hand corrections jittered per
    frame — 22-joint self-acceleration 0.065 -> 0.131; fixed by Gaussian-smoothing the corrections
    (sigma 4). (2) The contact term pulled hands IN on every touching frame (moved arms where there
    was no penetration); now it only forbids moving away.
  * **H4 local result: PARTIAL.** Gates on hand motion (0.16-0.53 cm mean), contact (unchanged),
    body jitter (0.0652 -> 0.0654), objects (identical) PASS. Local PEN 0.0179 -> 0.0096: the
    pre-registered < 0.005 is NOT reached — the realism constraints (2D anchor, 2 cm cap,
    smoothing) leave deep penetrations (foam 15/16 ~0.06) in place. Kaggle prediction for phys_v6
    (pre-registered): PEN 0.0080-0.0095, CD-O 30.84-30.90, ACC-O 0.0835-0.0841, CD-H 15.84-15.88,
    ACC-H 0.1405-0.1420. If PEN lands in range with CD-O at phys_v4's level, phys_v6 dominates
    phys_v5 (same PEN, CD-O -0.11).
  * **Kaggle (user-reported 2026-10-29): CD-H 15.85531, CD-O 30.78576, ACC-H 0.14129, ACC-O 0.08383,
    PEN 0.00908.** PEN, ACC-O, CD-H, ACC-H inside the pre-registered ranges. **CD-O outside its range
    (30.84-30.90), in the good direction (-0.085 vs phys_v4) although every object array is
    identical**: CD-O depends on the body through the scorer's first-frame alignment. New
    hypothesis H9 below. phys_v6 dominates phys_v4 and phys_v5. CARI4D baseline PEN 0.00423.
  * User review: hand overlays ~90 % fine (some slightly misaligned); first scored frames all good
    except episode 1; desk moves in ep 9, is static in the first half of ep 10 and all of ep 11;
    carts 27-29 move.
* **H9 — the first-frame body pose drives CD-H and CD-O.** The scorer aligns each episode on our
  body at its first scored frame, so a body error there shifts every frame. Evidence: phys_v6
  changed only hands and moved CD-O by -0.085. Test: replace ONLY episode 1's first-scored-frame
  body (user: the only bad first frame) by a better estimate; prediction: CD-H and CD-O both
  improve, by >= 0.05 each (>= 1.5 cm on that episode). FAIL if either changes by < 0.02.
  * Episode 1 inspected (`experiments/review/ep1/first_frames_zoom.png`): the BODY at the first
    scored frame is fine; the HOOP is wrong (on the body; the real hoop hangs from the extended
    hand ~60 cm to the left). H9 as stated does not apply to episode 1.
* **H10 — hoop pose from the image ellipse (`track1/hoop_conic.py`).** A circle of known radius
  seen by a calibrated camera is fixed by its image ellipse up to a two-fold tilt; the hand that
  holds the rim picks the tilt. Local on episode 1: 189 frames replaced (72/174 scored, where the
  tracker had lost the hoop or IoU < 0.3); chosen tilt passes the hand at 1.6 cm median vs 9.9 cm
  for the other; hand-object gap p90 on scored frames 24.5 -> 6.7 cm; review sheet
  `experiments/hoop_conic_ep1/episode_000001_hoop.png` (green on the real hoop, old pose on the body).
  **Kaggle prediction (phys_v7 = phys_v6 + episode 1 (variant c) + desk locks 10/11): CD-O 30.10-30.60 (episode 1 improves
  by ~6-20 cm); PEN, CD-H, ACC-H within +-0.001/+-0.02/+-0.001; ACC-O within +-0.003. FAIL if
  CD-O > 30.70.**
  * Iterations on episode 1 (local, all with review sheets): (a) replace only lost / IoU<0.3 frames:
    45 cm jumps where it switches to the old pose (object self-acc 0.09 -> 3.43) — rejected;
    (b) replace every frame + lam 1000: smooth, but where the hoop is half hidden the ellipse fits
    a partial arc and gives a thin edge-on hoop on frames the tracker had right — rejected;
    (c) **per frame keep whichever pose explains the mask better (`--choose_by_image`) + lam 1000**:
    lost frames fixed (hoop moves from the body to the hand), tracked frames unchanged; 140/174
    scored frames use the ellipse pose; hand-object gap p90 24.5 -> 16.5 cm. Episode 2 with its
    mesh radius (0.41 m = 83 cm hoop) made gaps worse (22 -> 83 cm median): depth from size
    scales with the radius, so it is re-run with the user's 67-72 cm prior (`--radius 0.35`).
* **H12 — episode 0's hoop belongs at the feet.** Episode 0 is "hula_hoop_inside_ground": on the scored
  frames the hoop lies on the floor and the person stands in it (overlay
  `experiments/hoop_conic_ep0/episode_000000_hoop.png`). Our hoop is 22-29 cm above the foot soles and
  50-70 cm in front of the body; stage 8 chose its depth (k 0.925) from HAND contact on frames where
  the contact is with the FEET. Moving the hoop along the camera rays by x1.193 (per-frame estimates
  1.172-1.198, p10-p90) puts its lowest point at the soles; foot points inside the ring 0 % -> 56 %
  (top-down `experiments/review/ep0_hoop_feet_topdown.png`: person stands inside the hoop as in the
  video). The ellipse refit (H10) is NOT used for episode 0 (old and new agree there).
* **phys_v7 = phys_v6 + H10 (ep 1) + H11 (desk 10/11) + H12 (ep 0 x1.193). Pre-registered Kaggle
  prediction: CD-O 29.0-30.5 (ep 0 -0.5..-1.3, ep 1 -0.2..-0.6, desks ~0); ACC-O 0.0830-0.0845;
  PEN 0.0085-0.0100; CD-H 15.84-15.87; ACC-H 0.1405-0.1420. FAIL if CD-O > 30.70.**
* **phys_v7 Kaggle (user-reported 2026-10-29): CD-O 30.78576 -> 27.91838 (-2.87), below the predicted
  29.0-30.5 (good direction); ACC-O 0.08385, PEN 0.00921, CD-H 15.85531, ACC-H 0.14129 in range.**
  CD-O is a mean over 30 episodes, so ~86 cm of one episode's CD-O disappeared — almost surely
  episode 0's hoop moving from 50-70 cm in front of the person to the feet. Lesson: gross
  misplacements in single episodes dominate CD-O, and physical evidence other than hand contact
  (feet inside the hoop) finds them. Next: the same audit for body contact (sitting on stools
  24/26), the jump-rope hoop (ep 2) and the paint roller (18/19).
* **J pilot, episode 6 (local, `track1/joint_fit.py`, 2026-10-29).** Full losses (silhouette + hand
  contact + no penetration): IoU 0.765 -> 0.748, object self-acc 0.025 -> 0.286 — REJECT (hand terms
  shake the object per frame). Heavy smoothing + weak hand terms: IoU 0.770, depth change p99 4.8 cm
  (contact-driven depth moves, the phys_v3 failure mode), self-acc 0.049 — REJECT. Silhouette only:
  IoU 0.765 -> 0.791, self-acc 0.025 -> 0.021, shift p99 0.9 cm; failed only "PEN not higher"
  (0.0048 -> 0.0060). Decision: hand terms off (w_contact = w_pen = 0); the PEN gate gets a 0.002 cm
  tolerance because pen_squeeze / hand_depen run after this step in postprocess.py. Expectation
  revised: silhouettes cannot correct depth, so the gain is a small refinement, larger only where
  tracking was poor.
* **J / H5 test-set check (2026-10-29): FAIL — silhouette refinement does not move objects toward
  the reference.** Silhouette-only joint fit on test-set episodes 9 (bowl) and 21 (bar stool):
  IoU 0.838 -> 0.864 and 0.786 -> 0.827, shifts <= 2.5 cm. Against the multi-view reference
  (ep 9): CD-O 23.5 -> 23.5 cm, lateral error 12.1 -> 12.2 cm. The lateral error is not a 2D
  misfit (H5 falsified), so the joint fit's silhouette term cannot fix it, and its hand-contact
  term moves objects in depth (the phys_v3 failure mode). J is not submitted; joint_fit stays
  disabled in postprocess_config.json. Remaining CD-O error is placement relative to the body
  (depth and registration), not image fit.
* **H11 — the desk is static where the user says so (`track1/static_lock.py`).** User: ep 10 first half
  and ep 11 static. The raw tracker agrees until video frame ~375 (ep 10) and ~715 (ep 11; the
  sequence is "white_desk_lower_to_ground", so the late motion is probably real — asked the user).
  Locked: ep 10 video 80-364 (held-out IoU 0.764 -> 0.770, shift <= 2.7 cm), ep 11 0-704
  (0.852 -> 0.865, <= 6.0 cm). Kaggle prediction (alone): ACC-O -0.0001..-0.0008 (these episodes are
  already smooth at lam 1000), CD-O -0.05..+0.02. FAIL if ACC-O rises or CD-O rises > 0.05.
* **R1 (RHINO contact detector vs scored frames) — NOT TESTABLE on Track 1, dropped before running.**
  Our packed outputs cover the scored span; frames between scored stretches are also contact
  frames (invalid reference, hand still touching), so only the 10 margin frames per side would be
  negatives. An F1 on that would be meaningless.
* **J — joint whole-video object fit (5-episode pilot: 3, 6, 9, 18, 19).** Fitting object R, t, scale
  per episode to SAM2 masks + MoGe depth + hand contact on contact frames + no penetration +
  smoothness lowers Kaggle CD-O by >= 0.6 (<= 30.27) with the other 25 episodes byte-identical.
  FAIL if CD-O drops < 0.2 or rises.

## Notes per experiment

**9 · phys_v4 (Kaggle, user-reported 2026-10-28): CD-H 15.85675, CD-O 30.87113 (best so far; phys_v2
31.04884), ACC-H 0.14125, ACC-O 0.08382 (best; phys_v2 0.10539), PEN 0.01499 (phys_v2 0.01246).**
phys_v3 minus the depth x1.04: lam 1000 object smoothing helps BOTH ACC-O (-0.0216) and CD-O (-0.18),
as the test set said; the local PEN rise (heavier smoothing pushes foam/iron into the hands) cost
+0.0025 PEN, as predicted.

**8 · phys_v3 (prepared 2026-10-28, not uploaded)** — first candidate tuned on the Track 2 local
test set (`track1/t2_testset.py`; 6 Track 2 episodes run through the unchanged pipeline, scored
against the multi-view reference; episode 7 excluded: the person barely moves, so the camera
cannot be registered). Findings there: CD-O 20.7 cm, of which only 3.6 cm remains after centring
the object -> placement dominates; objects sit 12.9 cm too close to the camera (stage 8 tie-break
takes the camera-side end of the contact band); object jitter ACC-O 0.150 vs 0.102 for a
constant-velocity guess.
* lam_t 1000 (all moving episodes except 12), depth x1.04 (all except 7 8 15 16 17, where it
  drove the hands into the object), PEN squeeze redone. Test set: CD-O 20.7 -> 16.1, ACC-O 0.150 -> 0.102.
* local: PEN 0.0135 -> 0.0191, accO(self) 0.089 -> 0.058, gap 4.92 -> 6.87 cm. The gap proxy also
  worsens on the test set while CD-O improves there, so it is not the guide at this scale.
* **Kaggle (user-reported): CD-O 31.04884 -> 37.06158 (+6.0, WRONG direction), ACC-O 0.10539 -> 0.08468
  (-0.021, as the test set predicted), PEN 0.01246 -> 0.01517, CD-H/ACC-H unchanged.** The depth x1.04
  did not transfer: same sign and size as p1_global (+5.9 for 8 episodes x1.07-1.16). The local
  gap/contact proxy (worse) was right; the test-set depth result was wrong. Suspected cause: the
  test set's scored frames (motion span +-15 frames) are not contact frames, so stage 8's contact
  placement there is biased in a way Track 1 (true contact frames) is not. lam 1000 is confirmed.
  Next: phys_v4 = phys_v3 without the depth change; test-set depth conclusions are on hold until
  its scored frames behave like Track 1's.


**7 · phys_v2 (prepared 2026-10-28, not uploaded)** — control phys_v1.
* body: `smooth_body.py --sigma 4` (MHR pose; Euler-switch-safe). Chosen on the Track 2 local test set
  (6 episodes, pelvis vs reference root only — no reference body model for the other joints): pelvis
  ACC error 0.149 -> 0.122, position 2.46 -> 2.39 cm. Two naive variants were rejected there
  (spikes at Euler switches); see the module docstring.
* PEN: `pen_squeeze.py` push-out up to 5 cm; foam 15/16 object x0.92 (two hands squeezing).
  Local PEN 0.0321 -> 0.0135, accH(self) 0.130 -> 0.065, accO(self) 0.086 -> 0.089, gap 4.79 -> 4.92.
* **Kaggle (user-reported): CD-H 15.9737 -> 15.85675 (-0.117), CD-O 31.04288 -> 31.04884 (+0.006),
  ACC-H 0.17421 -> 0.14125 (-0.033), ACC-O 0.10316 -> 0.10539 (+0.0022), PEN 0.02742 -> 0.01246 (-55%).**
  Body smoothing helped BOTH ACC-H and CD-H (the pelvis test-set evidence held for all joints).
  PEN matched the local prediction (Kaggle/local 0.92). The ACC-O cost equals the local self-acceleration
  rise from the larger push-outs (+0.0024): smooth the push-out more next time.


**6 · phys_v1 (prepared 2026-10-28, not uploaded)** — the physical-plausibility half of the score
(ACC-O, PEN), extending two fixes that Kaggle already confirmed on a few episodes to all of them.
Control: `stool_static_plus_smooth_t150` (Kaggle ACC-O 0.11673, PEN 0.04496). Only object
translations change (8,727 rows); body, rotation, scale and meshes are identical, so CD-H/ACC-H
must not move.
* stage 7 `lam_t 150` on every moving episode; stage 8 reuses production's depth placement
  exactly (`stage8_scale.py --pin`, via `ablate_pose_scale.py --pin_source_scale`)
* episodes 1, 7, 12: lam 150 moved scored-frame translations p99 22.5 / 19.5 / 6.7 cm (fast
  motion), so they keep the control's lam-50 poses; episode 7 still gets the PEN push-out
* stage 8b PEN push-out (validated `penfix_smooth` settings) where it lowers PEN on the fit AND
  held-out hand points and raises the episode's object self-acceleration by <= 0.015
  (`track1/assemble_phys.py`); accepted on 6 7 8 9 10 11 15 16 17 20 25 27 28 29
* local: PEN 0.0556 -> 0.0321 cm, accO(self) 0.104 -> 0.086, gap 4.71 -> 4.79 cm, contact 0.487 -> 0.482
* prediction: PEN ~0.026 on Kaggle (control's Kaggle/local ratio 0.81), ACC-O lower, CD-O within ~0.1
* **Kaggle (user-reported 2026-10-28): PEN 0.04496 -> 0.02742 (-39%), ACC-O 0.11673 -> 0.10316 (-0.0136),
  CD-O 30.95619 -> 31.04288 (+0.087), CD-H / ACC-H unchanged as required.** Kaggle/local PEN ratio 0.854.
  The small CD-O cost matches penfix_smooth (+0.042 for 3 episodes): the push-out moves objects
  off the hands by up to 3 cm. Physical-plausibility fixes work as predicted; the remaining ACC-O gap
  to the leaders (0.069) is the next target, tuned on the Track 2 local test set.


**1 · p1_global** — question: should the object live in the body's depth scale ("global") or in
MoGe's ("none")? The baseline chose by the smallest hand–object gap over the whole first..last
range (the contact span; gaps between scored stretches are frames with invalid ground truth, not
non-contact frames — see v2dlb/mhr_submission.py:34). Projective consistency and CARI4D (which anchors
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

**Result: CD-O 37.42 → 43.29 (worse by 5.87), ACC-O 0.1274 → 0.1306, PEN 0.1007 → 0.0889, CD-H and
ACC-H unchanged.** The contact signal predicted the direction; size consistency did not. Lessons:
* place the object where the hands touch it on scored (contact) frames — not by the body's scale
* object depth placement is a big lever: 8 episodes moved by 7–16 % in depth changed the
  30-episode CD-O by ~6 cm (≈ 22 cm per changed episode)
* local PEN is calibrated: Kaggle / local = 1.046 here, 1.042 for the baseline
Next: choose the object's depth scale per episode continuously (not 3 variants) by minimising the
hand–object gap on scored frames only, and check the gain locally before submitting.

**2 · stage67_v2 (submitted Oct 22; scores from the Oct 27 public leaderboard)** — a bundle, not a one-variable change: everything from stage 5e on was
re-run (see HANDOFF.md). Local proxies vs the baseline (`eval_local.py compare logs/eval/baseline logs/eval/stage67_v2`):
* local PEN 0.0966 -> 0.0555, median hand-object gap 12.4 -> 4.65 cm, contact 0.454 -> 0.489,
  object steps > 10 cm per frame 0.233 -> 0, accO(self) 0.1145 -> 0.1122, accH(self) unchanged (0.131 -> 0.130)
* worse: iron PEN ep 7 0.041 -> 0.093, ep 8 0.021 -> 0.128; foam block ep 15 0.22 -> 0.42 (ep 16/17 better); contact on
  bowls 0.94 -> 0.57 in ep 4 (gap still < 1 cm)
* unresolved: hula hoop (ep 0 gap 42 cm / ep 2 22 cm, tracker median IoU 0.28-0.39), paint roller ep 18/19 gap 11-16 cm,
  stool ep 26 gap 15.6 cm (seated); size spread across videos of one object up to 1.28x
* all 30 episodes packed: 740,780 rows, row ids/order equal to the sample, no non-finite values
Local proxies are not CD-H/CD-O/ACC errors (see AGENT_HANDOFF.md section 6); only Kaggle scores say whether CD-O improved.

## Beta models (local only, not submitted) — `run_beta.sh`, branch `beta-models`

Candidates: Video Depth Anything depth (`BETA_DEPTH=vda`), TRELLIS.2 / Hunyuan3D-2 meshes (`BETA_MESH`), CoTracker3
pose refinement (`BETA_TRACK=cotracker`). Run on the iron episodes 6 7 8 only.

**Compare against the control `logs/eval/beta/prodnow`** (current `submission/episodes`, i.e. experiment 2 = stage67_v2,
all 30 episodes), NOT against `logs/eval/baseline`: that one is experiment 0 (Oct 15), and `run_beta.sh eval` still
compares with it by default (set `BETA_BASELINE=logs/eval/beta/prodnow`). Against the stale baseline CoTracker looked like a
big win (gap 12.4 -> 0.1 cm); that was the pipeline changes since experiment 0, not the model. Also `eval_local.py compare` prints
the mean over each run's own episodes, so its last line is not comparable when the runs cover different episode sets.

Mean over episodes 6 7 8 (iron):

| run | gap cm | contact | PEN | accO | object size m |
|---|---|---|---|---|---|
| exp 0 baseline (Oct 15) | 18.08 | 0.316 | 0.0455 | 0.0969 | 0.305 |
| prodnow = exp 2 | 0.00 | 0.704 | 0.0924 | 0.1000 | 0.285 |
| warp (CoTracker3) | 0.10 | 0.695 | 0.0839 | 0.1234 | 0.285 |
| ember (Hunyuan3D-2 meshes) | 1.63 | 0.548 | 0.0896 | 0.1142 | 0.390 |

* CoTracker3: reprojection error of the refined poses drops (1.99 -> 1.21, 1.88 -> 1.08, 3.68 -> 1.60 px), but no proxy improves
  clearly over exp 2 (PEN -9 %, accO +23 % worse). No evidence for including it.
* Hunyuan3D-2: real 3D but lumpy, oversized meshes (0.35-0.44 m; iron prior 0.18-0.40); worse than exp 2 on episodes 6 and 7.
* TRELLIS.2: flat sheets (thickness 0.2-5 % of width) for the iron (47 x 82 px in the frame); the official example image works,
  so it is the input, not the install. Larger objects (stool, cart) are probed separately.
* Environment notes: `transformers` must be 4.56.2 (5.x breaks both DINO loaders); Hunyuan also needs `diffusers==0.35.2`.

**CoTracker3 on 12 episodes** (iron 6 7 8, foam block 15 16 17, stool 24 25 26, cart 27 28 29; `warp` vs the control `prodnow`,
means over the same 12 episodes):

| run | gap cm | contact | PEN | accH | accO | object size m |
|---|---|---|---|---|---|---|
| prodnow (exp 2) | 1.851 | 0.595 | 0.1283 | 0.1352 | 0.1263 | 0.491 |
| warp (CoTracker3) | 1.856 | 0.596 | 0.1122 | 0.1352 | 0.1403 | 0.489 |

PEN better on 9 of 12 episodes (-12.5 % mean), accO worse on 9 of 12 (+11 %), gap/contact unchanged (gap worse on 5, better on 2).
It does not repair the weak episodes (stool ep 26 gap 15.6 -> 15.9 cm). A trade (PEN vs ACC-O), not a gain: not included
in the submission. If a submission slot is spare it could be run as its own experiment (exp 2 + CoTracker, one change).

## PEN correction trial (local, October 27; not uploaded)

The active stage-8 runner uses `poses_final`, so the pre-existing `stage8_depen.py` was not in any submitted
pipeline. Tests in `track1/tests/test_penetration_contract.py` verify the scorer kernel on a synthetic solid,
the 40-frame trim offset in foam episode 16, the zero-gap-inside-object ambiguity, and a hard total-shift cap.
No production output or scorer code was modified by this experiment.

An isolated stage-8-only trial under `experiments/penfix_smooth` uses the existing MHR fit, budgeted mesh
and final poses. It fits object translations on 256 sampled hand vertices, applies Gaussian smoothing
(sigma=5) without unsmoothed residual repairs, and caps total shifts at 3 cm. It keeps all object rotations,
scale, body parameters and mesh unchanged. It then rebuilds and evaluates the actual packed Parquet, not
just the fitting objective. One initial unsmoothed attempt reduced PEN but raised object self-acceleration
from 0.184 to 0.528 cm/frame^2 and was rejected. An initial preview packed with frame_offset=0 was invalid;
the scored video in episode 16 starts at local offset 40. All reported results below use the correct offset.

| foam episode | local PEN (control → trial, cm) | gap (cm) | contact | accO self (cm/frame²) |
|---|---|---|---|---|
| 15 | 0.4170 → 0.2879 | 0 → 0.2 | 0.58 → 0.56 | 0.236 → 0.234 |
| 16 | 0.4584 → 0.2886 | 0 → 0.7 | 0.68 → 0.65 | 0.184 → 0.186 |
| 17 | 0.2637 → 0.0862 | 1.0 → 1.5 | 0.56 → 0.55 | 0.176 → 0.180 |

Independent anatomical hand samples from an MHR-model-hash-matched CARI4D spec (not the secret Kaggle
hand indices) also improve on all three: ep 15 0.326→0.235, ep 16 0.367→0.227, ep 17 0.159→0.049 cm.
These are local proxies; only Kaggle evaluates reference alignment and the hidden hand indices. Do not
interpret zero median contact gap as no penetration. A one-episode trial with stronger shifts had gap 3.7 cm
and contact 40 % and was rejected. Recheck the full packed candidate before any upload, and keep it separate
from both `submission/track1.parquet` and `experiments/stage67_v2/track1.parquet`.

**Full packed trial:** `experiments/penfix_smooth/track1.parquet` (740,780 rows), containing only 691 changed
`object_translation` rows in episodes 15-17; every other array and episode matches experiment 2 exactly.
Matched 30-episode local mean PEN is **0.0555 -> 0.0396 cm**; contact 0.4889 -> 0.4872, object
self-acceleration 0.1122 -> 0.1123 cm/frame². All three changed episodes pass the guardrails in
`experiments/penfix_smooth/full_gate.json`. The kit's local schema validator passes. The packed file
references code commit `fd79f965f212345b9bca3ca15c2587a2bb243ccf`, which is currently on local
`beta-models` only: **do not upload until the commit is pushed and the GitHub URL is accessible**.
The file SHA-256 is `415332f98891baf3af8e7fbb6c6ec9559af06dfacfe9b6be7d1e2c91bfafc652`.
The local trial alone did not establish its Kaggle scores or guarantee improvements on the other four metrics;
the subsequently user-reported Kaggle result is recorded below.

**3 · penfix_smooth user-reported Kaggle result (Oct 27, receipt not yet inspected):** PEN 0.04506 →
0.03349 cm (−25.7 %), CD-O 30.95621 → 30.99780 cm (+0.04159 cm), ACC-O 0.12346 → 0.12355
cm/frame² (+0.00009), ACC-H unchanged at 0.17421. User reports CD-H 15.97370 versus
15.71611 for experiment 2 (+0.25759 cm). **This CD-H difference is not attributable to this
local artifact:** compared against `experiments/stage67_v2/track1.parquet`, its only changed
`x/y/z` values are 691 `object_translation` rows in episodes 15–17. The Track 1 scorer's
`cd_h_cm` evaluates human surface points only and uses an episode-indexed sampling seed.
Check that all five score entries belong to the same Kaggle upload and compare the submitted
file's SHA-256 with `415332f98891baf3af8e7fbb6c6ec9559af06dfacfe9b6be7d1e2c91bfafc652`.
Do not treat the reported CD-H change as an algorithmic regression until provenance is verified.

**CARI4D follow-up feasibility (no pretrained inference yet):** the native-MHR implementation under
`/workspace/v2d_cari4d` has a model-matched anatomical hand surface asset and differentiable
`object_sdf_penetration_loss`; `track1/tests/test_cari4d_feasibility.py` confirms the loss produces
the expected gradient for hand points inside an analytic solid. However, the default parity
refinement penalizes **object samples inside the human**, not Kaggle's **hand points inside the
object**. This must be evaluated or augmented before drawing any conclusion about PEN. A real
CoCoNet experiment needs the separate gated `nvidia/cari4d_commercial` step-200000 checkpoint,
the CARI4D image/runtime, and accepted model terms. Neither this checkpoint nor a Docker runtime
is currently present in this container; `HF_TOKEN` is unset. The public upstream step-031397
checkpoint in `weights/cari4d/hf/` is not a drop-in replacement for the native-MHR model.
As a separate one-frame feasibility probe on the existing episode-16 submission, the installed
TorchScript MHR model produced nonzero gradients in 53 of its 54 finger-control columns from
a **box-proxy** hand-inside-object loss (53/54, gradient L2 0.00452); this is not a real-mesh
penetration improvement and did not change any submission. It indicates that bounded, temporal
MHR finger-pose refinement is technically possible without installing another model. The proper
next experiment needs actual budgeted-mesh signed depth, finger-pose and reprojection priors,
and the same packed-output guardrails as the object-only trial.

**Stop test for a finger-only pilot:** `track1/tests/test_foam_hand_overlap.py` checks
episode 16, video frame 237, against the *packed* `penfix_smooth` mesh/body and the
model-matched anatomical hand sample. **239 of 256 left-hand vertices are inside**;
mean left-hand depth is about 4.3 cm, maximum about 10 cm. Perturbing all finger
controls by 0.3 rad still moves 63 penetrating left-hand points no more than
0.2 cm (many palm/wrist points). Only 14/256 right-hand points penetrate.
Consequently, a small finger-only pose correction cannot remove this collision;
the next physical-consistency experiment should jointly diagnose body-hand/object
placement and geometry under image constraints. A full CARI4D run remains blocked
by its separate gated checkpoint/runtime and cannot be inferred from this diagnostic.

## VDA depth ablation (local only; stool episodes 24–26)

`ripple_stool`: Video Depth Anything depth with production SAM3D mesh and no CoTracker. All
stages 1–8 completed; the stage-8 review initially crashed on obsolete `variants`
in `scale_match.json`. The generator was subsequently fixed and its episode pages rebuilt.
Compare to the **current** control `logs/eval/beta/prodnow`, not the runner's old-baseline compare log.

| matched mean over ep 24–26 | current production | VDA |
|---|---:|---:|
| local PEN (cm) | 0.0181 | 0.0022 |
| hand–object gap (cm) | 6.95 | **12.57** |
| contact fraction | 0.328 | **0.167** |
| object size (m) | 0.552 | 0.629 |
| body minus object depth (cm) | −8.8 | −28.1 |

Episode 25's gap worsened 0 → 9.7 cm and contact 63% → 26%; episode 26's gap
15.6 → 19.3 cm. The PEN reduction is not evidence of improvement when the object moves
away from the hands. Do **not** substitute VDA into a submission on this evidence.
Review: `logs/beta/ripple_stool/07_smooth/index.html` and
`logs/beta/ripple_stool/08_final/review/episode_000025_result.png`.

**Scale-boundary follow-up (isolated stage-8-only ablation):** the VDA stool search
selected its lower bound k=0.800 on all three episodes despite a better logged
off-grid diagnostic on episode 25. Tests in `track1/tests/test_pose_scale_ablation.py`
confirm this before running. Re-searching k=0.65..1.25 in steps of 0.025 on the
same VDA outputs (`experiments/scale_rescue_vda_stool/`, partial/non-uploadable):

| matched mean, episodes 24–26 | production | original VDA | VDA + expanded search |
|---|---:|---:|---:|
| PEN cm | 0.0181 | 0.0022 | 0.0144 |
| hand–object gap cm | **6.95** | 12.57 | 8.03 |
| contact fraction | **0.328** | 0.167 | 0.268 |
| object self-acceleration | 0.118 | 0.116 | 0.104 |
| max object size m, mean | 0.552 | 0.629 | 0.582 |

Episode 25 gap 9.73→0.53 cm and contact 26%→53%, but production remains at
0 cm/63%; episode 26 remains disconnected (15.9 cm gap). The expanded scale
search **rescued a real integration bug but did not make VDA competitive as a
whole**. Its 745 changed packed rows are exclusively object translation/scale;
body, mesh, rotation and frame IDs are unchanged. Object self-acceleration is
not official ACC-O error. The stage-8 review generator was repaired and its
VDA stool page rebuilt at `logs/beta/ripple_stool/08_final/index.html`.

## Stage-7 translation smoothing ablation (local only; episodes 8, 16, 25, 28)

Current `stage7_smooth.py` uses `--lam_t 50`. Isolated `smooth_t15` and
`smooth_t150` trials use the same production stage-6 poses, the same rotation
smoothing, and re-run stage 8; their partial Parquets are **not uploadable**.
Matched control is `logs/eval/beta/prodnow` on exactly these four episodes.

| matched mean | production λ=50 | weaker λ=15 | stronger λ=150 |
|---|---:|---:|---:|
| object self-acceleration cm/frame² | 0.1672 | **0.2300 (worse)** | 0.1225 |
| PEN cm | 0.1639 | 0.1582 | 0.1665 |
| gap cm | 0.10 | 0.29 | 0.13 |
| contact fraction | 0.668 | 0.645 | 0.665 |

Weaker smoothing changed episode 25's object size by −0.075 (stage-8 scale
reselection) and moved its object up to 26 cm on scored frames. **Reject it.**
Stronger smoothing preserved all object sizes, body parameters, meshes,
rotations and frame IDs; only object translations changed. It lowered local
self-acceleration on all four episodes, but this is **not the reference-based
Kaggle ACC-O error**. Episode 25 still has a 7.6 cm maximum displacement
(3.8 cm scored-frame p95) relative to production, so review transitions.
The six-frame sheet and stage-7 checks are under `experiments/smooth_t150/review7/`;
episodes 8, 16, 28 pass; episode 25 has a warning from a large correction to
already-bad stage-6 outlier frames. Result sheets for 16 and 25 are under
`experiments/smooth_t150/review_final/`. No change is promoted or submitted yet.

A *separate* full-file ACC-O ablation is packed at
`experiments/smooth_t150_full/track1.parquet` (SHA-256
`4cdab16befc4dfa725c643a5479493d6c0b4756b8a91de2bd7b633ee64ff430a`).
It uses **experiment 2 as its control**, not the later PEN correction: all
740,780 rows are present; exactly 1,060 object-translation rows differ, in
episodes 8, 16, 25 and 28 only. The body, meshes, object rotations, scales
and all other episodes are byte-equivalent to experiment 2. The kit's local
schema validator passes and tests verify the episode NPZs. It embeds commit
`1b915846561405db43156418f104caad051eceab`, which must be pushed before
any submission. At the time of local validation this ablation had no Kaggle
score; the proxy decrease did not establish an ACC-O gain against the hidden reference.

**Full 30-episode check completed** (`experiments/smooth_t150_full/compare_production.json`):
local object self-acceleration 0.112164→0.106191 cm/frame² (−0.005973), PEN
0.055463→0.055814 cm (+0.000352), contact 0.488920→0.488524 and gap
4.651899→4.656379 cm. Only the four intended episodes change. This is a
clean ACC-O benchmark ablation, but self-acceleration is not the scored ACC-O
error. The subsequently user-reported Kaggle result appears below; keep it
separate from `penfix_smooth` when attributing that score.

**Later user-reported Kaggle result for `smooth_t150_full` (receipt not inspected):**
CD-H 15.97370, CD-O 30.92902, ACC-H 0.17421, ACC-O **0.11826**, PEN 0.04505.
Against experiment 2 ACC-O is 0.12346→0.11826, CD-O 30.95621→30.92902,
PEN approximately unchanged; against `penfix_smooth` ACC-O improves but PEN
returns to the experiment-2 level. The local CD-H-provenance anomaly remains.

## Stationary-stool multi-frame pose trial (local only; episodes 24 and 26)

The user confirmed that the stool does **not move throughout either video**;
episode 25 is not assumed stationary, and the camera is fixed only per video.
`track1/stationary_stool.py` uses 5 high-visibility anchors distributed across
each clip to select one existing final mesh pose by median mask IoU at *other*
anchor frames, then validates it on 66/71 held-out clear frames. It does not
change the camera or any body parameter, object mesh or scale. It writes one
constant object R,t for the whole video directly into isolated submission NPZs,
so later stage-8 scale selection cannot undo the stationarity.

| Episode | held-out IoU original → stationary | max translation change | local object self-acceleration |
|---|---:|---:|---:|
| 24 | 0.819 → 0.833 | 5.48 cm | 0.034 → 0 |
| 26 | 0.774 → 0.786 | 7.41 cm | 0.039 → 0 |

The episode-24 hand-gap proxy rises 5.19→7.02 cm, but continuous hand contact
is **not** a valid stool prior when sitting, so do not retune the stool position
to minimize it. Episode 25 is untouched. The result is a *physical-stationarity
hypothesis*, not proof of improvement against hidden ACC-O/CD-O reference data;
check packed structure, the reviewed overlays and Kaggle metrics before promotion.
Files and frame-index/holdout tests: `experiments/stool_static_v1/` and
`track1/tests/test_stationary_stool.py`.

**Full-file stationary-stool candidate based on the separately scored λ=150
smoothing variant:** `experiments/stool_static_plus_smooth_t150/track1.parquet`
(SHA-256 `db83a85d5379b78934287b21493149d0ff5a0215a9b17ba43aed6c16bc795f0a`).
Compared with `experiments/smooth_t150_full/track1.parquet`, only object
rotation and translation rows in episodes 24 and 26 differ; the other 28
episodes, all human parameters, object meshes, scales and scored frame IDs
are identical. The full 740,780-row kit validator and tests pass. Matched
30-episode local self-acceleration 0.106191→0.103777 cm/frame², PEN
0.055814→0.055558 cm; contact 0.488524→0.486457, gap 4.656379→4.710636.
Because sitting is not continuous hand contact and ACC-O scores error against
the hidden reference, these proxies alone do not prove a benchmark gain.
The file embeds commit `126c6c86a655ea9645ae6869786d7d8667baac89`;
push that commit so the organizers can inspect it.

**Later user-reported Kaggle result (receipt not inspected):** CD-H 15.97370,
CD-O 30.95619, ACC-H 0.17421, ACC-O **0.11673**, PEN 0.04496. Relative to
`smooth_t150_full`, ACC-O 0.11826→0.11673 (−0.00153 cm/frame²), CD-O
30.92902→30.95619 (+0.02717 cm), PEN 0.04505→0.04496 (−0.00009 cm),
human metrics unchanged. This **is** an incremental ACC-O gain, not evidence
of a large overall reconstruction improvement. For a larger effect, investigate
the hula hoop's poor stage-6 tracking and the filled-in late SAM2 masks.

**Camera-prior check:** Stage 1 already uses AnyCalib predictions from 16
frames evenly distributed across each video (episode 24 logged samples 0,
17, ..., 261), aggregates intrinsics/distortion by a robust median, and
stores fixed per-video intrinsics. No single-frame camera-initialization
bug has been demonstrated, so do not rerun camera calibration without an
independent residual or image evidence of a calibration error.

## Hula hoop: geometry versus segmentation (isolated local probes)

The user estimates the same rigid hoop's outer diameter at roughly **67–72 cm
from the person's shoe** (a soft prior, not a measured caliper value). The
episode-0 hoop lies on the floor in the first clear second, is subsequently
lifted, and lies on the floor again at the end; episode 1 is held throughout;
episode 2 is held until its final seconds. `floor.json` supplies an independently
fitted plane per video. AnyCalib already uses 16 distributed frames per video.

**Do not blindly replace the object mesh with a torus.**
`track1/hoop_torus_probe.py` generated clean watertight genus-one meshes of
one shared *physical* diameter and evaluated them against multi-frame masks
with the original camera and pose held fixed. No high-fit stage-6 training
anchors exist for episode 1. The best common ring (diameter 0.66 m, tube
0.03 m) gave held-out IoU ep 0 **0.668→0.721** but ep 2 **0.226→0.000**.
Ep 2 alone prefers a hoop near 0.82–0.86 m in its CURRENT pose/scale.
This violates the shared-size prior and implies that pose/scale also needs
repair. The geometry-only candidate failed its held-out guard, produced no
submission, and the original meshes/poses remain unchanged.

**Stage-4 mask root cause discovered in episode 0:** early floor-frame mask
000000 has an inner hole covering 64% of its outer contour; late floor-frame
000600 has **no hole**—SAM2 produced a filled disk. `masks_raw/1/000600.png`
and the final mask are identical, so downstream mask repair did not introduce
the fill. A conservative offline repair of high-fill masks only in the
user-confirmed final floor interval changed **48/630 frames**, never adds
pixels or affects the opening masks, and raised IoU of the existing saved
ring poses on those frames from **0.279→0.636** (48 improved, 0 worsened).
The masks reside only under `experiments/hoop_mask_v1/masks/1/`; the test is
not ground truth and the independent gate is an isolated stage-6 rerun,
logged at `experiments/hoop_mask_v1/stage6.log`. The rerun **failed the
predeclared unaffected-frame gate**: repaired scored frames 38/169 improved
IoU 0.210→0.623 and lost share 28.9%→0%, but untouched scored frames 131/169
worsened lost share 2.3%→22.1%. Newly tracked late poses are physically
almost the same as production (only ~0.2–0.3 cm translation difference;
~180° symmetric ring orientation, ~0.4–0.6 cm surface difference). A shape-
corrected mask helped the metric's target, not the 3D trajectory. **No hoop
submission assembled** from this rejected run. Full verdict in
`docs/STAGE_AUDIT_2026-10-27.md`.

**Held-hoop episode-1 visibility-anchor pilot:** no mask or mesh was changed.
Seven anchors were proposed, but the first three early attempts had best IoU
0.00/0.13/0.23 and were rejected. Scored-frame lost share worsened
32.8%→33.9% with no meaningful IoU improvement. The independent gate at
`experiments/hoop_vis_v1/stage6_gate.json` rejected the run; no submission
file was produced. Future anchor search must validate registration fit rather
than rely on maximum visible mask area.

**Quality-ranked 30-frame-window anchor follow-up, episode 1:** new fresh
anchors registered at video frames 141/302/360/448/460; many others failed
IoU<0.4. Scored median IoU 0.3545→0.3890, but lost share 32.8%→33.9%
and translation-step p99 13.84→19.04 cm. Gate failed; there is **no**
supported new hoop Parquet to submit from these overnight experiments.
See `docs/STAGE_AUDIT_2026-10-27.md` for the full diagnosis.
