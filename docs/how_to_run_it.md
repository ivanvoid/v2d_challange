##
## Setup models + data
##

export BASEPATH=/local-data/ivan/v2d_test
cd $BASEPATH
export HF_TOKEN=...
export ACCEPT_NVIDIA_EULA=1

cp /local-data/ivan/v2d/fetch_repos.sh .
cp /local-data/ivan/v2d/fetch_weights.sh .

mkdir weights
export WEIGHTS="$BASEPATH/weights"
# if load fails just restart it 
download_weights.sh $WEIGHTS

# Downloading challange data
git clone https://huggingface.co/datasets/nvidia/video_to_data_challenge challenge


##
## Inference
##

export V2D_BACKEND=venv
export OUT="$BASEPATH/out"
mkdir $OUT

export VIDEO_NAME="episode_000014"
export VIDEO="$BASEPATH/challenge/track_1/videos/chunk-000/observation.images.exo_camera/$VIDEO_NAME.mp4"


PROMPT=$(python3 -c "
import json
VIDEO_ID=14 # your number
for line in open('challenge/track_1/meta/episodes_metadata.jsonl'):
    r=json.loads(line)
    if r['episode_index']==VIDEO_ID: print(r['object_prompt']); break
")
echo "prompt: $PROMPT"

unset CONFIG_FILE CHECKPOINT_FILE

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" \
  --output_dir "$OUT/$VIDEO_NAME" \
  --object_prompt "$PROMPT" \
  --anycalib_weights        "$WEIGHTS/anycalib" \
  --moge_weights            "$WEIGHTS/moge" \
  --wilor_weights           "$WEIGHTS/wilor" \
  --sam2_weights            "$WEIGHTS/sam2" \
  --grounding_dino_weights  "$WEIGHTS/grounding_dino" \
  --sam3d_weights           "$WEIGHTS/sam3d" \
  --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort



python3 - "$OUT/$VIDEO_NAME/wilor_raw" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
hits = [(int(os.path.basename(f)[:6]), len(json.load(open(f))))
        for f in sorted(glob.glob(d + "/*.json"))]
hits = [(i, n) for i, n in hits if n > 0]
print("frames with hands:", len(hits), "/ total")
print("first 10:", hits[:10])
if hits:
    mid = hits[len(hits)//2]
    print(">>> suggested --reference_frame", mid[0], "(", mid[1], "hands )")
PY

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" \
  --output_dir "$OUT/$VIDEO_NAME" \
  --object_prompt "$PROMPT" \
  --anycalib_weights        "$WEIGHTS/anycalib" \
  --moge_weights            "$WEIGHTS/moge" \
  --wilor_weights           "$WEIGHTS/wilor" \
  --sam2_weights            "$WEIGHTS/sam2" \
  --grounding_dino_weights  "$WEIGHTS/grounding_dino" \
  --sam3d_weights           "$WEIGHTS/sam3d" \
  --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort \
  --reference_frame 280

# IF crash run this snippet and re-run previos command 
/opt/venvs/sam2/bin/python -m v2d.sam2.lib.video_to_masks \
  --video_path   $OUT/$VIDEO_NAME/frames \
  --prompts_path $OUT/$VIDEO_NAME/sam2_prompts.json \
  --weights_dir  $WEIGHTS/sam2 \
  --masks_dir    $OUT/$VIDEO_NAME/masks


##
## Visualize prediction
##
/opt/venvs/wilor/bin/python render_results.py \
    --frames_dir       $OUT/$VIDEO_NAME/frames \
    --intrinsics       $OUT/$VIDEO_NAME/intrinsics_stable.json \
    --body_mesh        $OUT/$VIDEO_NAME/mhr_mesh.pt \
    --object_mesh      $OUT/$VIDEO_NAME/mesh_scaled.obj \
    --object_poses_dir $OUT/$VIDEO_NAME/poses_smoothed \
    --output           $OUT/$VIDEO_NAME/body_object_overlay.mp4 \
    --fps 30