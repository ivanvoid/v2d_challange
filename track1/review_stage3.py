#!/usr/bin/env python3
"""Stage 3 review (object selection): one sheet per episode + checks + index.html.

Per episode (review/episode_XXXXXX_select.png):
  top     the chosen seed: full undistorted frame with the box, and a zoom on the box
  grid    each candidate frame as the detector saw it (depth-covered), cropped around the
          actor: actor box (yellow), never-greyed region (cyan dashed), wrists (R red, L green),
          every detection (green thick = chosen, orange = accepted, red = rejected) with an id
  legend  per detection: prompt, confidence, size (m), depth vs actor, gap, wrist gap, verdict

Checks: object found; chosen size inside the prior; object on >= 3 of 5 frames;
size consistent across frames (max/min <= 1.6); seed close to a wrist (<= 0.15 heights).

    /opt/venvs/sam3d_body/bin/python track1/review_stage3.py --episodes 6 8 17 26 28 \
        --out_root out --stage_dir logs/stages/03_select
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

YELLOW, CYAN, RED, ORANGE, GREEN, WR, WL = ((255, 220, 0), (0, 230, 230), (235, 50, 50), (255, 165, 0),
                                            (40, 230, 60), (255, 40, 40), (40, 220, 40))


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def dashed(draw, b, color, step=20):
    for x in range(int(b["x0"]), int(b["x1"]), step):
        for y in (b["y0"], b["y1"]):
            draw.line([x, y, min(x + step // 2, b["x1"]), y], fill=color, width=3)
    for y in range(int(b["y0"]), int(b["y1"]), step):
        for x in (b["x0"], b["x1"]):
            draw.line([x, y, x, min(y + step // 2, b["y1"])], fill=color, width=3)


def tile_frame(W: str, cand: dict, f: int, chosen: dict | None) -> Image.Image:
    a, nf = cand["actor"][str(f)], cand["near"].get(str(f))
    tag = chosen["tag"] if chosen and "tag" in chosen else "near"
    src = os.path.join(W, "select", "near_frames", f"{f:06d}.png") if tag == "near" \
        else os.path.join(W, "frames", f"{f:06d}.png")
    img = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(img)
    pb = a["box"]
    d.rectangle([pb["x0"], pb["y0"], pb["x1"], pb["y1"]], outline=YELLOW, width=3)
    for p, col in ((a["wrist_r"], WR), (a["wrist_l"], WL)):
        d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=col)
    dets = [c for c in cand["candidates"] if c["frame"] == f and c["tag"] == tag]
    uniq = []
    for c in dets:  # same box found by several prompts -> draw once, list once
        if not any(all(abs(c["box"][k] - u["box"][k]) < 4 for k in ("x0", "y0", "x1", "y1")) for u in uniq):
            uniq.append(c)
    legend = []
    for i, c in enumerate(uniq):
        b = c["box"]
        is_ch = chosen is not None and f == chosen["frame"] and all(abs(b[k] - chosen["box"][k]) < 4 for k in b)
        col = GREEN if is_ch else (ORANGE if c["accepted"] else RED)
        d.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=col, width=7 if is_ch else 3)
        d.text((b["x0"] + 3, b["y0"] + 2), f"#{i}", fill=col, font=font(24))
        verdict = "CHOSEN" if is_ch else ("ok" if c["accepted"] else "; ".join(c["reasons"]))
        legend.append((col, f"#{i} '{c['prompt']}' conf {c['conf']:.2f} size {c['size_m']:.2f}m"
                            f"{'' if not c.get('ref_size_m') else ' (median %.2f)' % c['ref_size_m']} "
                            f"dz {c['dz_m']:+.2f}m gap {c['gap']:.2f} wrist {c['wrist_gap']:.2f} "
                            f"[{'+'.join(c.get('views', ['image']))}] — {verdict}"))
    # crop around the actor and all detections
    xs = [pb["x0"], pb["x1"]] + [c["box"][k] for c in uniq for k in ("x0", "x1")]
    ys = [pb["y0"], pb["y1"]] + [c["box"][k] for c in uniq for k in ("y0", "y1")]
    ph = pb["y1"] - pb["y0"]
    x0, x1 = max(0, min(xs) - 0.3 * ph), min(img.width, max(xs) + 0.3 * ph)
    y0, y1 = max(0, min(ys) - 0.2 * ph), min(img.height, max(ys) + 0.2 * ph)
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    s = 560 / crop.height
    crop = crop.resize((int(crop.width * s), 560))
    band = 30 * (len(legend) + 2) + 10
    out = Image.new("RGB", (max(crop.width, 1150), 560 + band), (20, 20, 20))
    out.paste(crop, (0, 0))
    td = os.path.join(W, "select", f"topdown_{f:06d}.png")
    if tag == "near" and os.path.exists(td):
        tdi = Image.open(td).resize((380, 380))
        out = out.crop((0, 0, max(out.width, crop.width + 390), out.height))
        out.paste(tdi, (crop.width + 10, 0))
    od = ImageDraw.Draw(out)
    note = f"video frame {f + cand['frame_offset']}  actor depth {a['depth_m']:.2f} m"
    if nf and tag == "near":
        note += (f"  V greyed {100 * nf['greyed_fraction']:.0f}%, actor pixels hidden "
                 f"{100 * nf['actor_hidden_fraction']:.1f}%")
    od.text((8, 566), note, fill=CYAN, font=font(20))
    for j, (col, t) in enumerate(legend):
        od.text((8, 566 + 30 * (j + 1)), t[:150], fill=col, font=font(17))
    return out


def review_episode(W: str, review_dir: str) -> dict:
    cand = json.load(open(os.path.join(W, "select", "candidates.json")))
    rec = json.load(open(os.path.join(W, "select", "object_track.json")))
    ep = f"episode_{rec['episode']:06d}"
    lo, hi = cand["size_accept_m"]
    checks = []
    def add(name, ok, detail, level="WARN"):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": detail})
    ok = rec["status"] == "ok"
    hid = max(v["actor_hidden_fraction"] for v in cand["near"].values()) if cand["near"] else 0
    add("actor not hidden", hid <= 0.05, f"at most {100 * hid:.1f}% of the actor's own pixels greyed by the V")
    add("object found", ok, f"mode {rec.get('mode')}, prompt {rec.get('prompt')!r}" if ok
        else "nothing passed the gates — add seeds/" + ep + ".json", "FAIL")
    chosen = None
    if ok and rec["mode"] != "manual":
        chosen = next(c for c in cand["candidates"] if c["frame"] == rec["reference_frame"]
                      and abs(c["box"]["x0"] - rec["box"]["x0"]) < 1 and abs(c["box"]["y0"] - rec["box"]["y0"]) < 1)
        add("size in prior", lo <= rec["size_m"] <= hi, f"{rec['size_m']:.2f} m (accept {lo:.2f}-{hi:.2f})")
        add("seen on frames", len(rec["frames_with_object"]) >= 3, f"{len(rec['frames_with_object'])}/5 frames")
        add("size consistent", abs(math.log(rec["size_m"] / rec["ref_size_m"])) <= math.log(1.4),
            f"chosen {rec['size_m']:.2f} m vs median {rec['ref_size_m']:.2f} m of all accepted boxes")
        add("near a wrist", rec["wrist_gap"] <= 0.15, f"{rec['wrist_gap']:.2f} actor heights")
    elif ok:
        chosen = {"frame": rec["reference_frame"], "box": rec["box"], "tag": "full"}

    tiles = [tile_frame(W, cand, f, chosen) for f in cand["frames"]]
    top = None
    if ok:
        f = rec["reference_frame"]
        full = Image.open(os.path.join(W, "frames", f"{f:06d}.png")).convert("RGB")
        b = rec["box"]
        zoom = full.crop((int(max(0, b["x0"] - 40)), int(max(0, b["y0"] - 40)),
                          int(min(full.width, b["x1"] + 40)), int(min(full.height, b["y1"] + 40))))
        ImageDraw.Draw(full).rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=GREEN, width=8)
        full = full.resize((full.width // 2, full.height // 2))
        zoom = zoom.resize((int(zoom.width * full.height / zoom.height), full.height))
        top = Image.new("RGB", (full.width + zoom.width + 10, full.height), (0, 0, 0))
        top.paste(full, (0, 0)); top.paste(zoom, (full.width + 10, 0))
    cols = 2
    tw = max(t.width for t in tiles)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rh = [max(t.height for t in r) for r in rows]
    width = max(tw * cols, top.width if top else 0)
    sheet = Image.new("RGB", (width, 70 + (top.height + 10 if top else 0) + sum(rh)), (0, 0, 0))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else \
             "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
    title = (f"{ep} — stage 3 object selection — {status} — {cand['object']}: "
             + (f"seed video frame {rec['reference_video_frame']}, '{rec['prompt']}', {rec['mode']}" if ok else "NOT FOUND"))
    ImageDraw.Draw(sheet).text((10, 15), title, fill=(255, 255, 255), font=font(30))
    y = 70
    if top:
        sheet.paste(top, (0, y)); y += top.height + 10
    for r, h in zip(rows, rh):
        for c, t in enumerate(r):
            sheet.paste(t, (c * tw, y))
        y += h
    os.makedirs(review_dir, exist_ok=True)
    path = os.path.join(review_dir, f"{ep}_select.png")
    sheet.save(path)
    return {"episode": rec["episode"], "status": status, "checks": checks, "sheet": os.path.basename(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage_dir", required=True)
    args = ap.parse_args()
    review = os.path.join(args.stage_dir, "review")
    results = []
    for e in args.episodes:
        W = os.path.join(args.out_root, f"episode_{e:06d}")
        if not os.path.exists(os.path.join(W, "select", "candidates.json")):
            for old in glob.glob(os.path.join(review, f"episode_{e:06d}_*.png")):   # stale sheet removed
                if not any(t in os.path.basename(old) for t in ("_before_", "_v2_", "_v3_", "_old", "_refine")):
                    os.remove(old)
            results.append({"episode": e, "status": "FAIL", "checks": [{"check": "stage 3 output", "status": "FAIL",
                            "detail": "missing (see the episode log)"}]})
            continue
        chk_path = os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json")
        key_out = os.path.join(W, "select", "object_track.json")
        reuse_ok = (os.path.exists(chk_path) and os.path.exists(key_out)
                    and os.path.getmtime(chk_path) > os.path.getmtime(key_out)
                    and os.path.exists(os.path.join(review, json.load(open(chk_path)).get("sheet", "-"))))
        if reuse_ok:                      # sheet already up to date: reuse it
            results.append(json.load(open(chk_path)))
            continue
        r = review_episode(W, review)
        json.dump(r, open(os.path.join(args.stage_dir, f"episode_{e:06d}.checks.json"), "w"), indent=1)
        results.append(r)
        print(f"episode {e:3d}: {r['status']:4s}  " + "; ".join(
            f"{c['check']}: {c['detail']}" for c in r["checks"] if c["status"] != "PASS"))
    color = {"PASS": "#4c4", "WARN": "#fb3", "FAIL": "#f55"}
    out = ["<html><head><meta charset='utf-8'><title>Stage 3 object selection</title><style>body{font-family:"
           "sans-serif;background:#111;color:#eee} td,th{border:1px solid #444;padding:3px 7px} table{border-"
           "collapse:collapse} img{max-width:100%}</style></head><body><h1>Stage 3 — object selection</h1>"
           "<p>Check: the GREEN box is the object the person handles, and only that object. Grey = hidden by the "
           "V wedge (top view on the right of each frame: red V, yellow actor, white camera). If wrong, write seeds/episode_XXXXXX.json "
           "with {\"video_frame\": N, \"box\": [x0, y0, x1, y1]}.</p><table><tr><th>episode</th><th>status</th>"
           "<th>checks</th></tr>"]
    for r in results:
        det = "<br>".join(f"<span style='color:{color[c['status']]}'>{c['status']}</span> "
                          f"{html.escape(c['check'])}: {html.escape(c['detail'])}" for c in r["checks"])
        out.append(f"<tr><td><a href='#e{r['episode']}'>{r['episode']}</a></td><td style='color:"
                   f"{color[r['status']]}'>{r['status']}</td><td>{det}</td></tr>")
    out.append("</table>")
    for r in results:
        out.append(f"<h2 id='e{r['episode']}'>episode {r['episode']} — {r['status']}</h2><p>logs: "
                   f"<a href='episode_{r['episode']:06d}.log'>progress</a> · "
                   f"<a href='episode_{r['episode']:06d}.debug.log'>debug</a></p>")
        if r.get("sheet"):
            made = __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(
                os.path.getmtime(os.path.join(args.stage_dir, "review", r["sheet"]))))
            out.append(f"<p style='color:#999'>sheet made at {made}</p>")
            out.append(f"<a href='review/{r['sheet']}'><img src='review/{r['sheet']}'></a>")
    out.append("</body></html>")
    open(os.path.join(args.stage_dir, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {os.path.join(args.stage_dir, 'index.html')}")


if __name__ == "__main__":
    main()
