"""Per-frame object-mask quality for stage 4 (used by stage4_prompts.py --mode fill/assess and the review).

Inputs per frame (stage4_prompts.frame_stats + masks_conf.json from sam2_box_neg.py):
  area           mask pixels
  prob           mean SAM2 probability inside the mask (average of the backward and forward passes)
  uncertain      share of pixels with probability 0.3-0.7, relative to the mask area (fuzzy outline)
  fb_iou         IoU of the backward-pass and forward-pass masks (None where only one pass reached)
  size_m, dz_m   largest side of the mask in metres, and the mask's depth minus the actor's depth, both from
                 the NEAREST 20 % of the mask's depth (thin or edge-on objects pick up the wall's depth)
  de             colour distance (Lab) to the prompt frames
  wrist_cm       distance from the nearest wrist to the mask, in cm at the actor's depth
  person_frac    share of the mask lying on the actor's person mask (info: high whenever the object is held
                 in front of the body, because person masks include held objects)
  body_cover     share of the actor's person mask covered by the object mask (a leak over the torso / legs)
  hidden_hint    for an empty mask: the object's expected place (interpolated from its neighbours) is
                 covered by the actor's person mask (the body is in front of it)

Status, first match wins:
  rejected    emptied by the repair step (stray blob)                     -> the object is unknown here
  hidden      empty, and the actor's body covers where the object should be
  lost        empty otherwise (SAM2 lost the object)
  leak        the mask covers > LEAK_BODY of the actor's body, or area > LEAK_AREA x the local median
  suspicious  wrong region: TWO or more weak signs, or one strong sign. Weak: size outside x SIZE of the
              episode's median mask size, |dz| > DZ_M, colour > DE, > WRIST_CM from both wrists during the contact
              span. Strong: size outside x SIZE_STRONG. A mask within AT_HAND_CM of a wrist with the object's
              colour (<= COLOUR_OK) is the object: weak signs then only lower confidence. A single weak sign
              -> low_conf (an iron on the floor before the pick-up, a rotated iron's other colour, an edge-on pan)
  low_conf    prob < PROB_MIN, or fb_iou < FB_MIN, or uncertain > UNC_MAX
  ok
"""
from __future__ import annotations

import numpy as np

SIZE, SIZE_STRONG, DZ_M, DE, WRIST_CM = 2.0, 3.0, 1.0, 30.0, 35.0
AT_HAND_CM, COLOUR_OK = 10.0, 15.0      # at the hand with the object's colour: identity confirmed
LEAK_BODY, LEAK_AREA = 0.25, 1.6
PROB_MIN, FB_MIN, UNC_MAX = 0.85, 0.7, 0.35
ORDER = ["ok", "low_conf", "suspicious", "leak", "hidden", "lost", "rejected"]
COLOURS = {"ok": (0.20, 0.70, 0.20), "low_conf": (0.95, 0.80, 0.10), "suspicious": (0.95, 0.45, 0.00),
           "leak": (0.75, 0.20, 0.75), "hidden": (0.55, 0.55, 0.55), "lost": (0.90, 0.10, 0.10),
           "rejected": (0.30, 0.10, 0.10)}


def classify(r: dict, typical_size: float | None, local_area: float | None, in_contact: bool) -> tuple[str, list[str]]:
    """Status and reasons of one frame (see the module docstring)."""
    if r.get("rejected"):
        return "rejected", [r["rejected"]]
    if r.get("area", 0) == 0:
        return ("hidden", ["empty; the actor's body covers the object's place"]) if r.get("hidden_hint") \
            else ("lost", ["empty mask"])
    why = []
    if r.get("body_cover") is not None and r["body_cover"] > LEAK_BODY:
        why.append(f"covers {100 * r['body_cover']:.0f}% of the actor's body")
    if local_area and r["area"] > LEAK_AREA * local_area:
        why.append(f"area x{r['area'] / local_area:.1f} the local median")
    if why:
        return "leak", why
    # identity: no single measurement is reliable on its own (an iron lying on the floor before the pick-up is
    # far from the hands; a rotated iron shows another colour; a pan edge-on gets the wall's depth). A WEAK sign
    # alone only lowers confidence; SUSPICIOUS needs two weak signs or one strong one.
    weak, strong = [], []
    if typical_size and r.get("size_m"):
        k = max(r["size_m"] / typical_size, typical_size / r["size_m"])
        if k > SIZE_STRONG:
            strong.append(f"size {r['size_m']:.2f} m (x{k:.1f} the typical {typical_size:.2f} m)")
        elif k > SIZE:
            weak.append(f"size {r['size_m']:.2f} m (x{k:.1f} the typical {typical_size:.2f} m)")
    if r.get("dz_m") is not None and abs(r["dz_m"]) > DZ_M:
        weak.append(f"depth {r['dz_m']:+.2f} m from the actor")
    if r.get("de") is not None and r["de"] > DE:
        weak.append(f"colour distance {r['de']:.0f}")
    if in_contact and r.get("wrist_cm") is not None and r["wrist_cm"] > WRIST_CM:
        weak.append(f"{r['wrist_cm']:.0f} cm from both wrists during contact")
    confirmed = (r.get("wrist_cm") is not None and r["wrist_cm"] <= AT_HAND_CM
                 and r.get("de") is not None and r["de"] <= COLOUR_OK)
    if (strong or len(weak) >= 2) and not (confirmed and not strong):
        return "suspicious", strong + weak
    if weak or strong:
        why = ([f"at the hand with the object's colour, but {'; '.join(strong + weak)}"] if confirmed
               else [f"one weak sign: {'; '.join(weak)}"])
    if r.get("prob") is not None and r["prob"] < PROB_MIN:
        why.append(f"SAM2 probability {r['prob']:.2f}")
    if r.get("fb_iou") is not None and r["fb_iou"] < FB_MIN:
        why.append(f"backward/forward passes agree only IoU {r['fb_iou']:.2f}")
    if r.get("uncertain") is not None and r["uncertain"] > UNC_MAX:
        why.append(f"fuzzy outline ({100 * r['uncertain']:.0f}% uncertain pixels)")
    if why:
        return "low_conf", why
    return "ok", []


def local_median_area(areas: np.ndarray, f: int, half: int = 7) -> float | None:
    nb = [areas[g] for g in range(max(0, f - half), min(len(areas), f + half + 1)) if g != f and areas[g] > 0]
    return float(np.median(nb)) if nb else None


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True values."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out
