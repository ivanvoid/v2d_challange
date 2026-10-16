# setup on brain161
# run on brain302


export WEIGHTS=/local-data/ivan/v2d/weights

mkdir -p "$WEIGHTS"

download_weights.sh "$WEIGHTS" --only moge,anycalib
ls -R "$WEIGHTS"

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment (do not hardcode it)}"
export ACCEPT_NVIDIA_EULA=1
PATH="/opt/venvs/sam3d/bin:$PATH" download_weights.sh "$WEIGHTS" --only sam3d
PATH="/opt/venvs/foundation_pose/bin:$PATH" download_weights.sh "$WEIGHTS" --only foundation_pose
download_weights.sh "$WEIGHTS" --only grounding_dino,sam2,wilor,gsplat_refinement,wilor

rm -rf "$WEIGHTS/sam2/.cache"
/opt/venvs/moge/bin/hf download facebook/sam2.1-hiera-large --local-dir "$WEIGHTS/sam2"
ls "$WEIGHTS/sam2"

download_weights.sh "$WEIGHTS" --only detectron2

cd /local-data/ivan/v2d
git clone https://huggingface.co/datasets/nvidia/video_to_data_challenge challenge


pip install -e /workspace/v2d_depth \
            -e /workspace/v2d_droid_slam/docker \
            -e /workspace/v2d_geocalib/docker \
            -e /workspace/v2d_hawor/docker \
            -e /workspace/v2d_hamer/docker

pip install -e /workspace/v2d_mediapipe/docker
python -m v2d.pipelines.run_ego_wilor --help

export HF_TOKEN=hf_xxxxxxxx       # sam3d is gated
export ACCEPT_NVIDIA_EULA=1       # foundation_pose EULA (review it first)

download_weights.sh "$WEIGHTS" --only sam3d_body
mkdir -p "$WEIGHTS/sam3d_body/torch_home/hub"
git clone https://github.com/facebookresearch/dinov3.git \
  "$WEIGHTS/sam3d_body/torch_home/hub/facebookresearch_dinov3_main"

export VIDEO=/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4
export OUT=/local-data/ivan/v2d/out/body_pose && mkdir -p "$OUT"

TORCH_HOME="$WEIGHTS/sam3d_body/torch_home"
HF_HOME="$WEIGHTS/sam3d_body/hf_home"
/opt/venvs/sam3d_body/bin/python -m v2d.sam3d_body.lib.estimate_mhr_params \
  --rgb_path "$VIDEO" \
  --weights_dir "$WEIGHTS/sam3d_body" \
  --bbox_path "$OUT/bbox_track.pt" \
  --output_params_path "$OUT/mhr_params.pt" \
  --debug 1

  

export OUT=/local-data/ivan/v2d/out/body_pose
mkdir -p "$OUT"

python -m v2d.detectron2.docker.run_track_bboxes \
  --rgb_path "/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4" \
  --weights_dir "$WEIGHTS/detectron2" \
  --output_path "$OUT/bbox_track.pt"

python -m v2d.pipelines.run_ego_wilor \
    --video_path  /local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4 \
    --output_dir  /local-data/ivan/v2d/out/episode_000000 \
    --moge_weights     "$WEIGHTS/moge" \
    --sam2_weights     "$WEIGHTS/sam2" \
    --wilor_weights    "$WEIGHTS/wilor" \
    --anycalib_weights "$WEIGHTS/anycalib" \
    --undistort


export TORCH_HOME="local-data/ivan/v2d/weights/sam3d/torch_home"

export VIDEO=/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4
export OUT=/local-data/ivan/v2d/out
grep '"episode_index": 0' /local-data/ivan/v2d/challenge/track_1/meta/episodes_metadata.jsonl

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" \
  --output_dir "$OUT/episode_000000" \
  --object_prompt "<object_prompt from metadata>" \
  --anycalib_weights        "$WEIGHTS/anycalib" \
  --moge_weights            "$WEIGHTS/moge" \
  --wilor_weights           "$WEIGHTS/wilor" \
  --sam2_weights            "$WEIGHTS/sam2" \
  --grounding_dino_weights  "$WEIGHTS/grounding_dino" \
  --sam3d_weights           "$WEIGHTS/sam3d" \
  --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort \
  --gravity_align





# # moge -> loader reads $WEIGHTS/moge/model.pt
# mkdir -p "$WEIGHTS/moge" && cd "$WEIGHTS/moge" \
#   && git clone https://huggingface.co/Ruicheng/moge-2-vitl-normal _t \
#   && mv _t/model.pt model.pt && rm -rf _t \
#   && du -h model.pt  

# # sam2 -> $WEIGHTS/sam2/sam2.1_hiera_large.pt (+ its yaml config)
# mkdir -p "$WEIGHTS/sam2" && cd "$WEIGHTS/sam2" \
#   && git clone https://huggingface.co/facebook/sam2.1-hiera-large _t \
#   && mv _t/* . && rm -rf _t \
#   && du -h sam2.1_hiera_large.pt     # expect ~900 MB


python - <<'PY'
import json, glob, os
d = os.path.expandvars("$OUT/episode_000000/wilor_raw")
hits = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        if json.load(open(f)):
            hits.append(int(os.path.splitext(os.path.basename(f))[0]))
    except Exception:
        pass
print("frames with hands:", hits[:20], "..." if len(hits) > 20 else "")
print("total frames with hands:", len(hits))
PY

frames with hands: [240, 243, 245, 246, 247, 248, 606] 
total frames with hands: 7

export HF_HOME=/opt/huggingface
python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" --output_dir "$OUT/episode_000000" \
  --object_prompt "a shiny skinny hula hoop." --reference_frame 240 \
  --anycalib_weights "$WEIGHTS/anycalib" --moge_weights "$WEIGHTS/moge" \
  --wilor_weights "$WEIGHTS/wilor" --sam2_weights "$WEIGHTS/sam2" \
  --grounding_dino_weights "$WEIGHTS/grounding_dino" \
  --sam3d_weights "$WEIGHTS/sam3d" --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort


export WEIGHTS=/local-data/ivan/v2d/weights
export VIDEO=/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4
export OUT=/local-data/ivan/v2d/out

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" --output_dir "$OUT/episode_000000" \
  --object_prompt "a shiny skinny hula hoop." --reference_frame 240 \
  --anycalib_weights "$WEIGHTS/anycalib" --moge_weights "$WEIGHTS/moge" \
  --wilor_weights "$WEIGHTS/wilor" --sam2_weights "$WEIGHTS/sam2" \
  --grounding_dino_weights "$WEIGHTS/grounding_dino" \
  --sam3d_weights "$WEIGHTS/sam3d" --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort


unset CONFIG_FILE CHECKPOINT_FILE
export WEIGHTS=/local-data/ivan/v2d/weights
export OUT=/local-data/ivan/v2d/out
export VIDEO=/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4
ls -la "$WEIGHTS/wilor/pretrained_models/models/MANO_RIGHT.pkl"

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" --output_dir "$OUT/episode_000000" \
  --object_prompt "a shiny skinny hula hoop." --reference_frame 240 \
  --anycalib_weights "$WEIGHTS/anycalib" --moge_weights "$WEIGHTS/moge" \
  --wilor_weights "$WEIGHTS/wilor" --sam2_weights "$WEIGHTS/sam2" \
  --grounding_dino_weights "$WEIGHTS/grounding_dino" \
  --sam3d_weights "$WEIGHTS/sam3d" --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort


### run on a video 

cd /local-data/ivan/v2d
unset CONFIG_FILE CHECKPOINT_FILE
export V2D_BACKEND=venv
export WEIGHTS=/local-data/ivan/v2d/weights
export OUT=/local-data/ivan/v2d/out
export VIDEO=/local-data/ivan/v2d/challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000003.mp4


PROMPT=$(python3 -c "
import json
for line in open('challenge/track_1/meta/episodes_metadata.jsonl'):
    r=json.loads(line)
    if r['episode_index']==3: print(r['object_prompt']); break
")
echo "prompt: $PROMPT"

# fix for dino weights (can be also semi-linked)
mkdir -p /data/weights_dir/sam3d/torch_home/hub/checkpoints
cp /local-data/ivan/v2d/weights/sam3d/torch_home/hub/checkpoints/dinov2_*.pth \
   /data/weights_dir/sam3d/torch_home/hub/checkpoints/
ls -la /data/weights_dir/sam3d/torch_home/hub/checkpoints/
# rm -rf /data/weights_dir/sam3d/torch_home
# mkdir -p /data/weights_dir/sam3d
# ln -s /local-data/ivan/v2d/weights/sam3d/torch_home \
#       /data/weights_dir/sam3d/torch_home

python -m v2d.pipelines.run_ego_wilor \
  --video_path "$VIDEO" \
  --output_dir "$OUT/episode_000003" \
  --object_prompt "$PROMPT" \
  --anycalib_weights        "$WEIGHTS/anycalib" \
  --moge_weights            "$WEIGHTS/moge" \
  --wilor_weights           "$WEIGHTS/wilor" \
  --sam2_weights            "$WEIGHTS/sam2" \
  --grounding_dino_weights  "$WEIGHTS/grounding_dino" \
  --sam3d_weights           "$WEIGHTS/sam3d" \
  --foundation_pose_weights "$WEIGHTS/foundation_pose" \
  --undistort


### If crash
unset CONFIG_FILE CHECKPOINT_FILE
export WEIGHTS=/local-data/ivan/v2d/weights
export OUT=/local-data/ivan/v2d/out
/opt/venvs/sam2/bin/python -m v2d.sam2.lib.video_to_masks \
  --video_path   $OUT/episode_000003/frames \
  --prompts_path $OUT/episode_000003/sam2_prompts.json \
  --weights_dir  $WEIGHTS/sam2 \
  --masks_dir    $OUT/episode_000003/masks


# Stage 2 — bbox (detectron2 venv)
/opt/venvs/detectron2/bin/python -m v2d.detectron2.lib.track_bboxes \
    --rgb_path    $OUT/episode_000003/frames \
    --output_path $OUT/episode_000003/bbox_track.pt \
    --weights_dir weights/detectron2 \
    --model_size b --det_cat_id 0 --batch_size 8

# Stage 3 — body mesh (MUST include --output_mesh_path)
TORCH_HOME=$PWD/weights/sam3d_body/torch_home \
HF_HOME=$PWD/weights/sam3d_body/hf_home \
/opt/venvs/sam3d_body/bin/python -m v2d.sam3d_body.lib.estimate_mhr_params \
    --rgb_path            $OUT/episode_000003/frames \AC
    --cam_intrinsics_path $OUT/episode_000003/intrinsics_stable.json \
    --weights_dir         weights/sam3d_body \
    --bbox_path           $OUT/episode_000003/bbox_track.pt \
    --output_params_path  $OUT/episode_000003/mhr_params.pt \
    --output_mesh_path    $OUT/episode_000003/mhr_mesh.pt \
    --batch_size 8 --debug 0

# Stage 4 — fused overlay
/opt/venvs/wilor/bin/python render_results.py \
    --frames_dir       $OUT/episode_000003/frames \
    --intrinsics       $OUT/episode_000003/intrinsics_stable.json \
    --body_mesh        $OUT/episode_000003/mhr_mesh.pt \
    --object_mesh      $OUT/episode_000003/mesh_scaled.obj \
    --object_poses_dir $OUT/episode_000003/poses_smoothed \
    --output           $OUT/episode_000003/body_object_overlay.mp4 \
    --fps 30



    git clone https://github.com/nvidia-isaac/video_to_data.git
cd video_to_data/reconstruction