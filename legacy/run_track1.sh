#!/usr/bin/env bash
# =============================================================================
# run_track1.sh — Track 1 batch: object + MHR body -> submission/ -> parquet.
#
# Fresh run: everything is computed from the videos into out/<episode>/
# (nothing is read from out_old/ or older runs).
#
# Two lanes run IN PARALLEL on the same GPU, then a sequential finish pass:
#   object lane  per episode: trim -> undistort -> frames -> MoGe depth -> object
#                selection (actor-centred, background greyed) -> SAM2 -> SAM3D mesh
#                -> scale -> FoundationPose -> EKF               (track1/run_object.py)
#   body lane    per episode, as soon as that episode's frames + intrinsics exist:
#                person track -> SAM3D-Body -> one-identity MHR fit
#   finish pass  per episode: retry anything that failed in a lane (sequentially, so a
#                GPU out-of-memory from running side by side gets a clean second try),
#                then build submission files, QA, review sheets; then pack.
#
# Every run writes to logs/<RUN_ID>/ (logs/latest points at the newest run):
#   batch.log, object_lane.log, body_lane.log     what each lane printed, timestamped
#   episode_XXXXXX.{object,body,finish}.log       full output of every stage
#   summary.tsv, qa_report.tsv                    one line per episode
#   review/index.html                             open this to review every episode
#   review/episode_XXXXXX_object_candidates.png   object selection, all candidates
#   review/episode_XXXXXX_masks.png               SAM2 object mask over the scored span
#   review/episode_XXXXXX_result.png              submitted body + object on 6 frames
#   review/episode_XXXXXX_trajectory.png          raw vs smoothed object motion, body vs object depth
#   review/episode_XXXXXX.mp4                     full video, every RENDER_EVERY-th episode only
#
# Safe to re-run (same or new RUN_ID): stages skip when their output is complete.
# Exit status: 0 only when every requested episode finished.
#
# Usage:
#   ./run_track1.sh
#   EPISODES="9 14" ./run_track1.sh
#   PARALLEL=0 ./run_track1.sh          # lanes one after the other
#
# Knobs (env):
#   EPISODES       "A-B" and/or "N N N"                              (default 0-29)
#   PARALLEL       1 = object and body lanes side by side           (default 1)
#   OBJECT_WORKERS episodes the object lane runs at once             (default 2)
#   RENDER_EVERY   full video when idx % N == 0; 0 = never            (default 10)
#   SMOOTH_SIGMA   body temporal smoothing, frames                    (default 2.0)
#   PACK           1 = pack the parquet at the end                    (default 1)
#   COMMIT_URL     GitHub commit URL for the parquet; placeholder otherwise
#   GPU            CUDA device                                        (default 0)
#   RUN_ID         log folder name                       (default run_YYYYmmdd_HHMMSS)
# =============================================================================
set -uo pipefail

BASEPATH="${BASEPATH:-/local-data/ivan/v2d}"
EPISODES="${EPISODES:-0-29}"
PARALLEL="${PARALLEL:-1}"
OBJECT_WORKERS="${OBJECT_WORKERS:-2}"
RENDER_EVERY="${RENDER_EVERY:-10}"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-2.0}"
PACK="${PACK:-1}"
COMMIT_URL="${COMMIT_URL:-}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"

export V2D_BACKEND=venv
export CUDA_VISIBLE_DEVICES="$GPU"
WEIGHTS="$BASEPATH/weights"
OUT="$BASEPATH/out"
SUB="$BASEPATH/submission"
T1="$BASEPATH/track1"
LOGS="$BASEPATH/logs/$RUN_ID"
REVIEW="$LOGS/review"
STATE="$LOGS/state"
VIDEO_DIR="$BASEPATH/challenge/track_1/videos/chunk-000/observation.images.exo_camera"
META="$BASEPATH/challenge/track_1/meta/episodes_metadata.jsonl"
SAMPLE="$BASEPATH/v2d_submission_kit/data/track_1_sample_submission.parquet"
MHR_MODEL="$WEIGHTS/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt"
SPANS="$SUB/scored_spans.json"
SUMMARY="$LOGS/summary.tsv"

PY_PIPE=python                                  # /opt/conda: v2d pipeline driver
PY_BODY=/opt/venvs/sam3d_body/bin/python        # torch + MHR model + matplotlib
PY_RENDER=/opt/venvs/wilor/bin/python           # pyrender
PY_SUB="$BASEPATH/.venv-submit/bin/python"      # pandas + pyarrow + kit deps

unset CONFIG_FILE CHECKPOINT_FILE
mkdir -p "$OUT" "$SUB/episodes" "$SUB/reports" "$REVIEW" "$STATE"
ln -sfn "$RUN_ID" "$BASEPATH/logs/latest"
exec > >(tee -a "$LOGS/batch.log") 2>&1

ts() { date +%T; }
echo "===== run $RUN_ID  $(date -Is)  episodes: $EPISODES  parallel: $PARALLEL  object workers: $OBJECT_WORKERS  gpu: $GPU ====="

# sam3d dino-weights fix (see setup_me.sh): some venvs look under /data/weights_dir.
if [[ -d "$WEIGHTS/sam3d/torch_home/hub/checkpoints" ]]; then
    mkdir -p /data/weights_dir/sam3d/torch_home/hub/checkpoints 2>/dev/null || true
    cp -n "$WEIGHTS"/sam3d/torch_home/hub/checkpoints/dinov2_*.pth \
          /data/weights_dir/sam3d/torch_home/hub/checkpoints/ 2>/dev/null || true
fi

if [[ ! -x "$PY_SUB" ]]; then
    echo "missing $PY_SUB — create it with:"
    echo "  python3 -m venv .venv-submit && .venv-submit/bin/pip install -r v2d_submission_kit/requirements.txt"
    exit 1
fi
[[ -f "$SPANS" ]] || "$PY_SUB" "$T1/scored_spans.py" --sample "$SAMPLE" --out "$SPANS" || exit 1

expand_episodes() {
    local tok lo hi
    for tok in $1; do
        if [[ "$tok" == *-* ]]; then
            lo="${tok%-*}"; hi="${tok#*-}"
            for ((n = lo; n <= hi; n++)); do echo "$n"; done
        else
            echo "$tok"
        fi
    done
}
EP_LIST="$(expand_episodes "$EPISODES")"

meta_field() {  # $1 = episode index, $2 = field
    python3 -c "
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get('episode_index') == int(sys.argv[2]):
        print(r.get(sys.argv[3], '')); break
" "$META" "$1" "$2"
}
span_field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" "$SPANS" "$1" "$2"; }
json_field() {
    python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print(','.join(map(str,v)) if isinstance(v,list) else ('' if v is None else v))" "$1" "$2"
}
epname() { printf 'episode_%06d' "$1"; }

# ---------------------------------------------------------------- object lane
object_episode() {  # $1 = N; writes STATE/<ep>.{ready,object_ok,object_fail}
    local N="$1" EP WORK LOG
    EP="$(epname "$N")"; WORK="$OUT/$EP"; LOG="$LOGS/$EP.object.log"
    mkdir -p "$WORK"
    echo "[$(ts)] $EP object start" >> "$LOG"
    if "$PY_PIPE" "$T1/run_object.py" --video "$VIDEO_DIR/$EP.mp4" --out_dir "$WORK" \
            --prompt "$(meta_field "$N" object_prompt)" --object_name "$(meta_field "$N" object)" \
            --first "$(span_field "$N" first)" --last "$(span_field "$N" last)" \
            --weights "$WEIGHTS" --review_dir "$REVIEW" --episode "$N" \
            --ready_marker "$STATE/$EP.ready" >> "$LOG" 2>&1; then
        touch "$STATE/$EP.object_ok"; rm -f "$STATE/$EP.object_fail"
        echo "[$(ts)] $EP object OK  $(grep -E '^  selection:' "$LOG" | tail -1)"
        return 0
    fi
    grep -E 'Error|error:' "$LOG" | tail -1 | cut -c1-200 > "$STATE/$EP.object_fail"
    echo "[$(ts)] $EP object FAILED: $(cat "$STATE/$EP.object_fail")"
    return 1
}

object_lane() {
    # OBJECT_WORKERS episodes at a time, round-robin; any failure is retried alone in the finish pass.
    local N i=0 pids=()
    local -a lanes
    for N in $EP_LIST; do
        lanes[$((i % OBJECT_WORKERS))]+="$N "
        i=$((i + 1))
    done
    for i in "${!lanes[@]}"; do
        ( for N in ${lanes[$i]}; do object_episode "$N"; done ) &
        pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i"; done
    touch "$STATE/object_lane.done"
}

# ------------------------------------------------------------------ body lane
body_episode() {  # $1 = N; needs frames + intrinsics; writes STATE/<ep>.{body_ok,body_fail}
    local N="$1" EP WORK BODY LOG OFFSET
    EP="$(epname "$N")"; WORK="$OUT/$EP"; BODY="$WORK/body"; LOG="$LOGS/$EP.body.log"
    OFFSET="$(json_field "$WORK/track1_meta.json" frame_offset)"; OFFSET="${OFFSET:-0}"
    mkdir -p "$BODY"
    if [[ ! -f "$BODY/bbox_track.pt" ]]; then
        echo "[$(ts)] person track" >> "$LOG"
        "$PY_PIPE" -m v2d.detectron2.docker.run_track_bboxes --rgb_path "$WORK/frames" \
            --weights_dir "$WEIGHTS/detectron2" --output_path "$BODY/bbox_track.pt" \
            --batch_size 8 >> "$LOG" 2>&1 || { echo "person-track" > "$STATE/$EP.body_fail"; return 1; }
    fi
    if [[ ! -f "$BODY/mhr_mesh.pt" ]]; then
        echo "[$(ts)] SAM3D-Body" >> "$LOG"
        "$PY_PIPE" -m v2d.sam3d_body.docker.run_estimate_mhr_params --rgb_path "$WORK/frames" \
            --cam_intrinsics_path "$WORK/intrinsics_stable.json" \
            --weights_dir "$WEIGHTS/sam3d_body" --bbox_path "$BODY/bbox_track.pt" \
            --output_params_path "$BODY/mhr_params.pt" --output_mesh_path "$BODY/mhr_mesh.pt" \
            --batch_size 8 >> "$LOG" 2>&1 || { echo "sam3d-body" > "$STATE/$EP.body_fail"; return 1; }
    fi
    if [[ ! -f "$BODY/mhr_fit.npz" ]]; then
        echo "[$(ts)] MHR fit (one identity, sigma $SMOOTH_SIGMA)" >> "$LOG"
        "$PY_BODY" "$T1/fit_mhr.py" --mesh "$BODY/mhr_mesh.pt" --model "$MHR_MODEL" \
            --first "$(span_field "$N" first)" --last "$(span_field "$N" last)" --frame_offset "$OFFSET" \
            --smooth_sigma "$SMOOTH_SIGMA" --out "$BODY/mhr_fit.npz" >> "$LOG" 2>&1 \
            || { echo "mhr-fit" > "$STATE/$EP.body_fail"; return 1; }
    fi
    touch "$STATE/$EP.body_ok"; rm -f "$STATE/$EP.body_fail"
    return 0
}

body_lane() {
    local N EP
    for N in $EP_LIST; do
        EP="$(epname "$N")"
        # Wait for this episode's frames + intrinsics, or for the object lane to give up on it.
        while [[ ! -f "$STATE/$EP.ready" ]]; do
            if [[ -f "$STATE/$EP.object_fail" || -f "$STATE/object_lane.done" ]]; then break; fi
            sleep 10
        done
        if [[ ! -f "$STATE/$EP.ready" ]]; then
            echo "[$(ts)] $EP body skipped (no frames: object lane failed before depth)"
            continue
        fi
        if body_episode "$N"; then
            echo "[$(ts)] $EP body OK"
        else
            echo "[$(ts)] $EP body FAILED at $(cat "$STATE/$EP.body_fail") (finish pass will retry)"
        fi
    done
}

# ---------------------------------------------------------------- finish pass
N_FAIL=0
record() {  # $1=N $2=status $3=note
    printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$SUMMARY"
    printf '[%s] [episode %06d] %s  %s\n' "$(ts)" "$1" "$2" "${3:-}"
    [[ "$2" == FAIL* ]] && N_FAIL=$((N_FAIL + 1))
}

finish_episode() {  # $1 = N
    local N="$1" EP WORK BODY LOG OFFSET FIRST LAST REPORT FLAGS L0 L1 FR
    EP="$(epname "$N")"; WORK="$OUT/$EP"; BODY="$WORK/body"; LOG="$LOGS/$EP.finish.log"
    FIRST="$(span_field "$N" first)"; LAST="$(span_field "$N" last)"

    if [[ ! -f "$STATE/$EP.object_ok" ]]; then
        echo "[$(ts)] $EP retrying object stages (sequential)"
        object_episode "$N" || { record "$N" "FAIL(object)" "$(cat "$STATE/$EP.object_fail" 2>/dev/null)"; return; }
    fi
    if [[ ! -f "$STATE/$EP.body_ok" ]]; then
        echo "[$(ts)] $EP running/retrying body stages (sequential)"
        body_episode "$N" || { record "$N" "FAIL(body:$(cat "$STATE/$EP.body_fail"))" "see $EP.body.log"; return; }
    fi
    OFFSET="$(json_field "$WORK/track1_meta.json" frame_offset)"; OFFSET="${OFFSET:-0}"

    "$PY_SUB" "$T1/build_episode.py" --episode "$N" --ep_dir "$WORK" --out_dir "$SUB/episodes" \
        --frame_offset "$OFFSET" >> "$LOG" 2>&1 || { record "$N" "FAIL(build)" "see $EP.finish.log"; return; }

    REPORT="$SUB/reports/$EP.json"
    "$PY_BODY" "$T1/qa_episode.py" --episode "$N" --ep_dir "$WORK" --sub_dir "$SUB/episodes" \
        --spans "$SPANS" --model "$MHR_MODEL" --frame_offset "$OFFSET" --out "$REPORT" \
        --export_body "$BODY/submitted_body.pt" >> "$LOG" 2>&1
    grep '^\[qa' "$LOG" | tail -1
    cp -f "$REPORT" "$LOGS/" 2>/dev/null
    FLAGS="$( [[ -f "$REPORT" ]] && json_field "$REPORT" flags )"

    L0=$((FIRST - OFFSET)); L1=$((LAST - OFFSET))
    FR="$(python3 -c "import sys; a,b=int(sys.argv[1]),int(sys.argv[2]); print(','.join(str(round(a+(b-a)*i/5)) for i in range(6)))" "$L0" "$L1")"
    "$PY_RENDER" "$BASEPATH/render_results.py" \
        --frames_dir "$WORK/frames" --intrinsics "$WORK/intrinsics_stable.json" \
        --body_mesh "$BODY/submitted_body.pt" \
        --object_mesh "$SUB/episodes/${EP}_object.obj" --object_poses_dir "$WORK/poses_smoothed" \
        --frames "$FR" --title "$EP submitted body (blue) + object (orange) | flags: ${FLAGS:-none}" \
        --output "$REVIEW/${EP}_result.png" >> "$LOG" 2>&1 || echo "  ($EP result sheet failed, see $EP.finish.log)"
    "$PY_BODY" "$T1/viz_trajectory.py" --episode "$N" --work_dir "$WORK" --sub_dir "$SUB/episodes" \
        --spans "$SPANS" --model "$MHR_MODEL" --frame_offset "$OFFSET" \
        --out "$REVIEW/${EP}_trajectory.png" >> "$LOG" 2>&1 || echo "  ($EP trajectory plot failed, see $EP.finish.log)"

    if [[ "$RENDER_EVERY" -gt 0 && $((N % RENDER_EVERY)) -eq 0 ]]; then
        echo "[$(ts)] $EP full video" >> "$LOG"
        "$PY_RENDER" "$BASEPATH/render_results.py" \
            --frames_dir "$WORK/frames" --intrinsics "$WORK/intrinsics_stable.json" \
            --body_mesh "$BODY/submitted_body.pt" \
            --object_mesh "$SUB/episodes/${EP}_object.obj" --object_poses_dir "$WORK/poses_smoothed" \
            --frame_start "$L0" --frame_end "$L1" --output "$REVIEW/$EP.mp4" --fps 30 >> "$LOG" 2>&1 \
            || echo "  ($EP video failed, see $EP.finish.log)"
    fi
    record "$N" "DONE${FLAGS:+($FLAGS)}" ""
}

# ---------------------------------------------------------------- run
printf 'episode\tstatus\tnote\n' > "$SUMMARY"
T0=$SECONDS
if [[ "$PARALLEL" == "1" ]]; then
    object_lane > >(tee -a "$LOGS/object_lane.log") 2>&1 &
    PID_OBJ=$!
    body_lane > >(tee -a "$LOGS/body_lane.log") 2>&1 &
    PID_BODY=$!
    wait "$PID_OBJ"; wait "$PID_BODY"
else
    object_lane > >(tee -a "$LOGS/object_lane.log") 2>&1
    body_lane > >(tee -a "$LOGS/body_lane.log") 2>&1
fi
echo "[$(ts)] lanes finished after $(( (SECONDS - T0) / 60 )) min; finish pass"
for N in $EP_LIST; do finish_episode "$N"; done

# ---- QA roll-up + review index ------------------------------------------------
"$PY_SUB" - "$SUB/reports" "$LOGS/qa_report.tsv" "$REVIEW" "$SUMMARY" <<'PY'
import html, json, sys
from pathlib import Path
reports, tsv, review, summary = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
cols = ["episode", "acc_h_self_cm", "acc_o_self_cm", "pen_cm", "contact_gap_median_cm",
        "body_minus_object_dz_cm", "obj_iou_mean", "body_in_bbox", "object_extent_m", "flags"]
fmt = lambda c, v: (",".join(v or []) if c == "flags" else "" if v is None
                    else f"{v:.3f}" if isinstance(v, float) else str(v))
status = {}
for line in summary.read_text().splitlines()[1:]:
    parts = line.split("\t")
    status[int(parts[0])] = (parts[1], parts[2] if len(parts) > 2 else "")
rows = {}
for p in sorted(reports.glob("episode_*.json")):
    r = json.loads(p.read_text())
    if int(r["episode"]) in status:
        rows[int(r["episode"])] = [fmt(c, r.get(c)) for c in cols]
tsv.write_text("\t".join(cols) + "\n" + "".join("\t".join(v) + "\n" for v in rows.values()))
print(f"wrote {tsv} ({len(rows)} episodes)")

out = ["<html><head><meta charset='utf-8'><title>Track 1 review</title><style>"
       "body{font-family:sans-serif;background:#111;color:#eee} img{max-width:100%;margin:4px 0}"
       "h2{border-top:2px solid #444;padding-top:10px} .fail{color:#f66} .flag{color:#fb3} "
       "table{border-collapse:collapse} td,th{border:1px solid #444;padding:2px 6px}</style></head><body>",
       "<h1>Track 1 review</h1><p>Per episode check: (1) the GREEN box is the object the person handles; "
       "(2) the mask follows it; (3) blue body and orange object sit on the person / object; "
       "(4) trajectory: no spikes, object depth near the body while held.</p>",
       "<table><tr><th>ep</th><th>status</th><th>note</th>" + "".join(f"<th>{c}</th>" for c in cols[1:]) + "</tr>"]
for ep in sorted(status):
    st, note = status[ep]
    cls = "fail" if st.startswith("FAIL") else ("flag" if "(" in st else "")
    vals = rows.get(ep, [""] * len(cols))[1:]
    out.append(f"<tr><td><a href='#ep{ep}'>{ep}</a></td><td class='{cls}'>{html.escape(st)}</td>"
               f"<td>{html.escape(note)}</td>" + "".join(f"<td>{html.escape(v)}</td>" for v in vals) + "</tr>")
out.append("</table>")
for ep in sorted(status):
    name = f"episode_{ep:06d}"
    out.append(f"<h2 id='ep{ep}'>{name} — {html.escape(status[ep][0])}</h2><p>{html.escape(status[ep][1])}</p>")
    for suffix in ("_object_candidates.png", "_masks.png", "_result.png", "_trajectory.png"):
        if (review / f"{name}{suffix}").exists():
            out.append(f"<div>{suffix[1:-4]}</div><a href='{name}{suffix}'><img src='{name}{suffix}'></a>")
    if (review / f"{name}.mp4").exists():
        out.append(f"<div>video</div><video src='{name}.mp4' controls width='960'></video>")
out.append("</body></html>")
(review / "index.html").write_text("\n".join(out))
print(f"wrote {review / 'index.html'}")
PY

echo
echo "===== summary ($SUMMARY) ====="
show_table() { if command -v column >/dev/null; then column -t -s $'\t' "$1"; else cat "$1"; fi; }
show_table "$SUMMARY"
show_table "$LOGS/qa_report.tsv"

# ---- pack -------------------------------------------------------------------
if [[ "$PACK" == "1" ]]; then
    n_have=$(find "$SUB/episodes" -maxdepth 1 -name 'episode_*.npz' | wc -l)
    n_need=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$SPANS")
    commit_arg=()
    [[ -n "$COMMIT_URL" ]] && commit_arg=(--commit "$COMMIT_URL")
    if [[ "$n_have" -ge "$n_need" ]]; then
        "$PY_SUB" "$T1/pack_submission.py" "${commit_arg[@]}" || N_FAIL=$((N_FAIL + 1))
    else
        echo "packing partial ($n_have/$n_need episodes; not uploadable)"
        "$PY_SUB" "$T1/pack_submission.py" --partial || N_FAIL=$((N_FAIL + 1))
    fi
fi

echo "===== run $RUN_ID finished $(date -Is) after $(( (SECONDS - T0) / 60 )) min: $N_FAIL failure(s). Review: $REVIEW/index.html ====="
[[ "$N_FAIL" -eq 0 ]]
