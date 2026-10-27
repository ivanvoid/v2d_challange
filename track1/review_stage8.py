#!/usr/bin/env python3
"""Stage 8 review (final result): one sheet per episode + checks + index.html.

Sheet (review/episode_XXXXXX_final.png):
  top     the submitted body (blue) + object (orange) on 6 frames (render_results.py --frames)
  plots   MoGe/body depth ratio over the scored span; current fixed/follow scale-search score
          over k, with the chosen result and out-of-grid global/per-frame diagnostics marked
  table   the QA report (submission/reports/episode_XXXXXX.json)

Checks: submission file built; QA flags; chosen variant improves the contact/PEN objective;
gap median <= 5 cm; penetration <= 1 cm; body and object jitter <= 0.5 cm/frame^2.

    /opt/venvs/sam3d_body/bin/python track1/review_stage8.py --episodes 6 8 17 26 28 --out_root out \
        --sub_dir submission --stage_dir logs/stages/08_final
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLS = {"fixed": "tab:blue", "follow": "tab:green"}


def scale_summary(sm):
    """Normalize current and legacy stage-8 scale reports for review, without changing scoring."""
    if "best" in sm and "legacy_variants" in sm:
        return sm["best"], sm["legacy_variants"]["none"]
    return sm["variants"][sm["chosen"]], sm["variants"]["none"]


def review_episode(W, sub_dir, review_dir):
    meta = json.loads((Path(W) / "track1_meta.json").read_text())
    e = meta["episode"]
    ep = f"episode_{e:06d}"
    sm = json.loads((Path(W) / "track" / "scale_match.json").read_text())
    qa_path = os.path.join(sub_dir, "reports", f"{ep}.json")
    qa = json.loads(Path(qa_path).read_text()) if os.path.exists(qa_path) else {}
    built = os.path.exists(os.path.join(sub_dir, "episodes", f"{ep}.npz"))
    ch, none = scale_summary(sm)
    contact = ch.get("contact_frac", ch.get("gap_frac_under_2cm", 0))
    size_factor = ch.get("size_factor", ch.get("k", 1))

    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    built = built and os.path.exists(os.path.join(sub_dir, "episodes", f"{ep}_object.obj"))
    add("submission file built", built, f"{ep}.npz + _object.obj in {sub_dir}/episodes", "FAIL")
    improves = (ch["score"] <= none["score"] + 0.05) if "score" in ch and "score" in none else (
        ch["gap_median_cm"] <= none["gap_median_cm"] + 0.5)
    add("scale match helps", improves,
        f"'{sm['chosen']}': gap {none['gap_median_cm']:.1f} -> {ch['gap_median_cm']:.1f} cm, "
        f"PEN {none.get('pen_cm', float('nan')):.3f} -> {ch.get('pen_cm', float('nan')):.3f} cm, "
        f"object x{size_factor:.3f}")
    add("hands touch the object", ch["gap_median_cm"] <= 5, f"median hand-object gap {ch['gap_median_cm']:.1f} cm, "
        f"{100 * contact:.0f}% of frames < 2 cm")
    if qa:
        add("penetration", qa.get("pen_cm", 0) <= 1.0, f"{qa.get('pen_cm', float('nan')):.3f} cm")
        add("body jitter", qa.get("acc_h_self_cm", 0) <= 0.5, f"{qa.get('acc_h_self_cm', float('nan')):.3f} cm/frame^2")
        add("object jitter", qa.get("acc_o_self_cm", 0) <= 0.5, f"{qa.get('acc_o_self_cm', float('nan')):.3f} cm/frame^2")
        add("QA flags", not qa.get("flags"), ", ".join(qa.get("flags", [])) or "none")
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.6, 0.6])
    res = os.path.join(review_dir, f"{ep}_result.png")
    ax = fig.add_subplot(gs[0, :])
    if os.path.exists(res):
        ax.imshow(mpimg.imread(res))
    ax.axis("off"); ax.set_title("submitted body (blue) + object (orange) on 6 frames of the scored span", fontsize=14)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(sm["video_frames"], sm["ratio_moge_over_body"], color="gray", lw=0.8, label="per frame")
    ax.plot(sm["video_frames"], sm["ratio_smoothed"], color="tab:green", lw=2, label="smoothed")
    ax.axhline(1 / sm["s_global"], color="tab:blue", ls="--", label="median (global)")
    ax.legend(fontsize=9); ax.set_title("MoGe depth / body depth on the body's pixels")
    ax = fig.add_subplot(gs[1, 1])
    if "table" in sm:
        for mode, color in COLS.items():
            rows = sorted((r for r in sm["table"] if r["mode"] == mode), key=lambda r: r["k"])
            if rows:
                ax.plot([r["k"] for r in rows], [r["score"] for r in rows], color=color, marker=".",
                        label=f"{mode} search")
        ax.scatter([ch["k"]], [ch["score"]], color="red", s=85, zorder=5,
                   label=f"chosen {sm['chosen']}: gap {ch['gap_median_cm']:.1f} cm, PEN {ch['pen_cm']:.3f} cm")
        for mode in ("global", "per_frame"):
            diagnostic = sm["legacy_variants"].get(mode)
            if diagnostic:
                ax.scatter([diagnostic["k"]], [diagnostic["score"]], marker="x", s=60,
                           label=f"{mode} diagnostic (outside search if x is off-grid)")
        ax.set_xlabel("object position/size factor k")
        ax.set_ylabel("contact + penetration objective (lower is better)")
        ax.set_title("Stage 8 search — boundary choices may exclude better diagnostic scales")
    else:
        for k, v in sm["variants"].items():
            ax.plot(sm["eval_video_frames"], v["gaps"], lw=2.5 if k == sm["chosen"] else 1,
                    label=f"{k}: median {v['gap_median_cm']:.1f} cm")
        ax.set_title("hand-object gap (cm) over evaluated video frames")
    ax.set_yscale("symlog", linthresh=1); ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    keys = ["acc_h_self_cm", "acc_o_self_cm", "pen_cm", "contact_gap_median_cm", "body_vs_moge_dz_cm",
            "object_vs_moge_dz_cm", "obj_iou_mean", "body_in_bbox", "object_scale", "object_extent_m", "flags"]
    lines = [f"{k:24s} {qa.get(k) if not isinstance(qa.get(k), float) else round(qa.get(k), 3)}" for k in keys if k in qa]
    ax.text(0, 1, "QA report (submission/reports)\n" + "\n".join(lines), va="top", family="monospace", fontsize=13)
    fig.suptitle(f"{ep} — stage 8 final (scale match + build + QA) — {status}", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    sheet = os.path.join(review_dir, f"{ep}_final.png")
    fig.savefig(sheet, dpi=50)
    plt.close(fig)
    return {"episode": e, "status": status, "checks": checks, "sheet": os.path.basename(sheet)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--sub_dir", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()
    review = os.path.join(args.stage_dir, "review")
    os.makedirs(review, exist_ok=True)
    results = []
    for e in args.episodes:
        W = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.exists(os.path.join(W, "track", "scale_match.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_final.png")):
                os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 8 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(args.sub_dir, "reports", f"episode_{e:06d}.json")
        scale_path = os.path.join(W, "track", "scale_match.json")
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > max(os.path.getmtime(key_out), os.path.getmtime(__file__),
                                                         os.path.getmtime(scale_path))
                    and os.path.exists(os.path.join(review, json.loads(Path(chk_path).read_text()).get("sheet", "-"))))
        if reuse_ok:                      # sheet already up to date: reuse it
            results.append(json.loads(Path(chk_path).read_text()))
            continue
        r = review_episode(W, args.sub_dir, review)
        (Path(args.stage_dir) / f"episode_{e:06d}.checks.json").write_text(json.dumps(r, indent=1))
        results.append(r)
        print(f"episode {e:3d}: {r['status']:4s}  " + "; ".join(
            f"{c['check']}: {c['detail']}" for c in r["checks"] if c["status"] != "PASS"))
    color = {"PASS": "#4c4", "WARN": "#fb3", "FAIL": "#f55"}
    out = ["<html><head><meta charset='utf-8'><title>Stage 8 final</title><style>body{font-family:sans-serif;"
           "background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-collapse:collapse}"
           " img{max-width:100%}</style></head><body><h1>Stage 8 — final result (scale match, submission files, QA)"
           "</h1><p>Check: blue body on the person, orange object on the object, hands on the object while it is "
           "held.</p><table><tr><th>episode</th><th>status</th><th>checks</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> "
                          f"{html.escape(c['check'])}: {html.escape(c['detail'])}" for c in r["checks"])
        out.append(f"<tr><td><a href='#e{r['episode']}'>{r['episode']}</a></td><td style='color:"
                   f"{color[r['status']]}'>{r['status']}</td><td>{det}</td></tr>")
    out.append("</table>")
    for r in results:
        out.append(f"<h2 id='e{r['episode']}'>episode {r['episode']} — {r['status']}</h2>")
        if r.get("sheet"):
            made = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(review, r["sheet"]))))
            out.append(f"<p style='color:#999'>sheet made at {made}</p>"
                       f"<a href='review/{r['sheet']}'><img src='review/{r['sheet']}'></a>")
    out.append("</body></html>")
    (Path(args.stage_dir) / "index.html").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
