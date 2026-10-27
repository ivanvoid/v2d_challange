#!/usr/bin/env bash
# =============================================================================
# setup_beta_envs.sh — the venvs the beta models of run_beta.sh run in (once per host).
#
#   ./setup_beta_envs.sh                      # all: vda trellis2 hunyuan cotracker
#   ./setup_beta_envs.sh trellis2 cotracker   # some
#   V2D_VENV_ROOT=$HOME/venvs ./setup_beta_envs.sh   # if /opt/venvs is not writable (run_beta.sh needs
#                                                    # the same V2D_VENV_ROOT, and the production venvs there)
#
#   vda        torch 2.1.1 cu121 + xformers 0.0.23 (Video-Depth-Anything/requirements.txt)
#   trellis2   torch 2.6.0 cu124, flash-attn 2.7.3, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel
#              (TRELLIS.2/setup.sh without its sudo apt / pillow-simd); builds CUDA code: needs nvcc 12.x
#   hunyuan    torch 2.6.0 cu124, Hunyuan3D-2 requirements, custom_rasterizer + differentiable_renderer (nvcc)
#   cotracker  torch 2.6.0 cu124
# The model code itself is imported from models/<repo> (sys.path), not installed. Uses uv when available
# (python 3.10 fetched if needed), else python3.10 -m venv. Log: logs/beta/setup_<venv>.log.
# =============================================================================
set -euo pipefail
BASEPATH="${V2D_DATA:-${BASEPATH:-/local-data/ivan/v2d}}"
M="$BASEPATH/models"
VENVS="${V2D_VENV_ROOT:-/opt/venvs}"
TORCH124=(torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124)
COMMON=(numpy opencv-python-headless trimesh scipy pillow tqdm)
WANT=("${@:-vda trellis2 hunyuan cotracker}")
WANT=(${WANT[*]})
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"     # A40
mkdir -p "$BASEPATH/logs/beta"
[[ -w "$VENVS" || -w "$(dirname "$VENVS")" ]] || { echo "$VENVS not writable: set V2D_VENV_ROOT"; exit 2; }
command -v "$CUDA_HOME/bin/nvcc" > /dev/null || echo "warning: no nvcc in $CUDA_HOME (trellis2/hunyuan build CUDA code; set CUDA_HOME)"
export PATH="$CUDA_HOME/bin:$PATH"

mkvenv() {  # $1 = path
    if [[ -x "$1/bin/python" ]]; then return 0; fi
    if command -v uv > /dev/null; then uv venv --python 3.10 "$1"
    else python3.10 -m venv "$1" && "$1/bin/pip" install -U pip wheel setuptools; fi
}
pipi() {  # $1 = venv, rest = pip install args
    local V="$1"; shift
    if command -v uv > /dev/null; then uv pip install --python "$V/bin/python" "$@"
    else "$V/bin/pip" install "$@"; fi
}

setup_vda() {
    local V="$VENVS/vda"
    mkvenv "$V"
    pipi "$V" torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
    pipi "$V" xformers==0.0.23 numpy==1.24.0 einops==0.4.1 easydict imageio==2.37.0 imageio-ffmpeg==0.4.7 \
        matplotlib opencv-python-headless pillow tqdm
    "$V/bin/python" -c "import sys; sys.path.insert(0, '$M/Video-Depth-Anything'); \
from video_depth_anything.video_depth import VideoDepthAnything; print('vda ok')"
}

setup_trellis2() {
    local V="$VENVS/trellis2" X
    X="$(mktemp -d "${TMPDIR:-/tmp}/trellis2_ext.XXXX")"
    mkvenv "$V"
    pipi "$V" "${TORCH124[@]}"
    pipi "$V" "${COMMON[@]}" setuptools wheel ninja packaging psutil imageio imageio-ffmpeg easydict "transformers==4.56.2" tensorboard pandas \
        lpips zstandard kornia timm "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
    pipi "$V" flash-attn==2.7.3 --no-build-isolation
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$X/nvdiffrast"
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$X/nvdiffrec"
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$X/CuMesh"
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$X/FlexGEMM"
    cp -r "$M/TRELLIS.2/o-voxel" "$X/o-voxel"
    for e in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do pipi "$V" "$X/$e" --no-build-isolation; done
    rm -rf "$X"
    "$V/bin/python" -c "import sys; sys.path.insert(0, '$M/TRELLIS.2'); import o_voxel, cumesh, flex_gemm, nvdiffrast.torch; \
from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('trellis2 ok')"
}

setup_hunyuan() {
    local V="$VENVS/hunyuan" H="$M/Hunyuan3D-2"
    mkvenv "$V"
    pipi "$V" "${TORCH124[@]}"
    pipi "$V" "${COMMON[@]}" setuptools wheel ninja pybind11 -r "$H/requirements.txt" "transformers==4.56.2" "diffusers==0.35.2"
    (cd "$H/hy3dgen/texgen/custom_rasterizer" && "$V/bin/python" setup.py install)
    (cd "$H/hy3dgen/texgen/differentiable_renderer" && "$V/bin/python" setup.py install)
    "$V/bin/python" -c "import torch, sys; sys.path.insert(0, '$H'); import custom_rasterizer; \
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; from hy3dgen.texgen import Hunyuan3DPaintPipeline; print('hunyuan ok')"
}

setup_cotracker() {
    local V="$VENVS/cotracker"
    mkvenv "$V"
    pipi "$V" "${TORCH124[@]}"
    pipi "$V" "${COMMON[@]}" einops
    "$V/bin/python" -c "import sys; sys.path.insert(0, '$M/co-tracker'); from cotracker.predictor import CoTrackerPredictor; print('cotracker ok')"
}

FAIL=()
for v in "${WANT[@]}"; do
    case "$v" in vda|trellis2|hunyuan|cotracker) ;; *) echo "unknown venv $v"; exit 2 ;; esac
    echo "[$(date +%T)] $v -> $VENVS/$v (log logs/beta/setup_$v.log)"
    set +e                                   # (errexit is ignored inside an if condition: run, then test)
    ( set -euo pipefail; "setup_$v" ) > "$BASEPATH/logs/beta/setup_$v.log" 2>&1; rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%T)] $v: $(tail -1 "$BASEPATH/logs/beta/setup_$v.log")"
    else
        FAIL+=("$v"); echo "[$(date +%T)] $v FAILED — tail of the log:"; tail -15 "$BASEPATH/logs/beta/setup_$v.log"
    fi
done
[[ ${#FAIL[@]} -eq 0 ]] || { echo "failed: ${FAIL[*]}"; exit 1; }
