#!/usr/bin/env python3
"""Stage 5e review page: one section per object — the cross-video choice sheet, then the FINAL model's
validation on held-out frames in each of its videos (out/<ep>/mesh/validate.json, the acceptance summary).

    /opt/venvs/sam3d_body/bin/python track1/review_stage5e.py --out_root out --stage_dir logs/stages/05e_object
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(paths.META)]
    objects = sorted({m["object"] for m in meta})
    rows, sections = [], []
    for o in objects:
        vids = sorted(m["episode_index"] for m in meta if m["object"] == o)
        rp = os.path.join(args.stage_dir, f"{o}.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        ch = r["chosen"]
        val = {}
        for e in vids:
            vp = os.path.join(args.out_root, f"episode_{e:06d}", "mesh", "validate.json")
            val[e] = json.load(open(vp)) if os.path.exists(vp) else None
        n_ok = sum(1 for v in val.values() if v and v["accepted"])
        status = "PASS" if n_ok == len(vids) else "FAIL"
        col = "#4c4" if status == "PASS" else "#f55"
        vtxt = []
        for e, v in val.items():
            if v is None:
                vtxt.append(f"ep {e}: not validated")
            else:
                d = "n/a" if v["median_depth_cm"] is None else f"{v['median_depth_cm']:.1f} cm"
                vtxt.append(f"ep {e}: {'ACCEPTED' if v['accepted'] else 'REJECTED'} — IoU {v['median_iou']:.2f}, "
                            f"coverage {v['median_coverage']:.2f}, depth {d}, size x{v.get('rel_scale') or 1:.2f}, "
                            f"extent {v.get('extent_m') or 0:.2f} m")
        rows.append(f"<tr><td><a href='#{o}'>{html.escape(o)}</a></td><td style='color:{col}'>{status}</td>"
                    f"<td>{html.escape(ch['candidate'])} of ep {ch['source_episode']} (score {ch['score']:.2f})</td>"
                    f"<td>{'<br>'.join(html.escape(t) for t in vtxt)}</td></tr>")
        sec = [f"<h2 id='{o}'>{html.escape(o)} — {status}</h2><p>logs: <a href='{o}.log'>progress</a> · "
               f"<a href='{o}.debug.log'>debug</a> · <a href='{o}.json'>scores</a></p>"]
        sheet = os.path.join(args.stage_dir, "review", f"{o}.png")
        if os.path.exists(sheet):
            made = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(sheet)))
            sec.append(f"<h3>choice across videos</h3><p style='color:#999'>sheet made at {made}</p>"
                       f"<a href='review/{o}.png'><img src='review/{o}.png'></a>")
        for e, v in val.items():
            if v and os.path.exists(os.path.join(args.stage_dir, "review", v["sheet"])):
                sec.append(f"<h3>episode {e}: final model on held-out frames</h3>"
                           f"<a href='review/{v['sheet']}'><img src='review/{v['sheet']}'></a>")
        sections.append("\n".join(sec))
    out = ["<html><head><meta charset='utf-8'><title>Stage 5e object mesh</title><style>body{font-family:"
           "sans-serif;background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-"
           "collapse:collapse} img{max-width:100%}</style></head><body><h1>Stage 5e — one mesh per object</h1>"
           "<p>Every candidate of an object (from all 3 videos) is tested on frames of all its videos; the winner "
           "is written to each video at that video's best size, then validated on held-out clear frames. "
           "Check: the render looks like the object (handle, wheels...), the blue model outline matches the green "
           "mask. Per-video candidates: <a href='../05c_check/index.html'>05c_check</a>.</p>"
           "<table><tr><th>object</th><th>status</th><th>chosen mesh</th><th>final validation per video</th></tr>",
           *rows, "</table>", *sections, "</body></html>"]
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
