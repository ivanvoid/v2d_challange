#!/usr/bin/env bash
# =============================================================================
# fetch_repos.sh — point the track1_v2d venvs at a live reconstruction repo.
#
# The image bakes DEPENDENCIES (third-party pip packages + compiled CUDA/C++
# extensions) and a snapshot of the v2d.* module libs at /workspace. This script
# re-installs those module libs *editable* from a repo you provide, so you can
# iterate on the v2d source without rebuilding the image. All third-party deps
# are already present, so the re-install is fast and needs no network.
#
# Usage:
#   fetch_repos.sh [REPO_DIR | GIT_URL] [--ref <git-ref>] [--include-fp-lib]
#
#   REPO_DIR   path to a reconstruction repo checkout (contains modules/).
#              Bind-mount your repo into the container and pass it here.
#   GIT_URL    an https/ssh git URL to clone into $V2D_REPO_DIR (default
#              /workspace/_repo) at optional --ref.
#   (no arg)   defaults to /workspace (the baked snapshot) — a no-op refresh.
#
# Notes:
#   * foundation_pose has native code (mycpp C++ + bundlesdf/mycuda CUDA) that is
#     compiled into the image from the BAKED snapshot. Its Python lib is NOT
#     re-pointed by default because the compiled .so lives under /workspace.
#     Pass --include-fp-lib to re-point it anyway (only safe if the native
#     sources are unchanged; native changes require an image rebuild).
#
#   V2D_VENV_ROOT   venv root (default: /opt/venvs)
# =============================================================================
set -euo pipefail

VENV_ROOT="${V2D_VENV_ROOT:-/opt/venvs}"
REF=""
INCLUDE_FP_LIB=0
REPO_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --ref=*) REF="${1#*=}"; shift ;;
        --include-fp-lib) INCLUDE_FP_LIB=1; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) REPO_ARG="$1"; shift ;;
    esac
done

REPO_ARG="${REPO_ARG:-/workspace}"

# Resolve a usable modules/ directory from REPO_ARG (local path or git URL).
if [[ "$REPO_ARG" == *"://"* || "$REPO_ARG" == git@* ]]; then
    dest="${V2D_REPO_DIR:-/workspace/_repo}"
    echo "==> Cloning $REPO_ARG -> $dest"
    if [[ -d "$dest/.git" ]]; then
        git -C "$dest" fetch --all --tags
    else
        rm -rf "$dest"; git clone "$REPO_ARG" "$dest"
    fi
    [[ -n "$REF" ]] && git -C "$dest" checkout "$REF"
    REPO_DIR="$dest"
else
    REPO_DIR="$(cd "$REPO_ARG" && pwd)"
fi

# Accept either a repo root (contains modules/) or the modules/ dir itself.
if [[ -d "$REPO_DIR/modules" ]]; then
    MODULES="$REPO_DIR/modules"
elif [[ -d "$REPO_DIR/v2d_common" ]]; then
    MODULES="$REPO_DIR"
else
    echo "ERROR: could not find modules/ (or v2d_common/) under $REPO_DIR" >&2
    exit 3
fi
echo "==> Using module source: $MODULES"

_pip() { echo "${VENV_ROOT}/$1/bin/pip"; }

# Editable, no-deps (deps already baked), no network.
_install() {
    local pip_bin="$1"; shift
    "$pip_bin" install --no-deps --no-build-isolation "$@"
}

echo "== anycalib =="
_install "$(_pip anycalib)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_anycalib/lib"

echo "== moge =="
_install "$(_pip moge)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_moge/lib"

echo "== grounding_dino =="
_install "$(_pip grounding_dino)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_mv" \
    -e "$MODULES/v2d_grounding_dino/lib"

echo "== sam2 =="
_install "$(_pip sam2)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_mv" \
    -e "$MODULES/v2d_sam2/lib"

echo "== wilor =="
_install "$(_pip wilor)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_wilor/lib"

echo "== sam3d =="
_install "$(_pip sam3d)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_sam3d/lib"

echo "== gsplat_refinement =="
_install "$(_pip gsplat_refinement)" -e "$MODULES/v2d_common" \
    -e "$MODULES/v2d_gsplat_refinement/lib"

echo "== foundation_pose =="
_install "$(_pip foundation_pose)" -e "$MODULES/v2d_common" -e "$MODULES/v2d_mv" \
    -e "$MODULES/v2d_mesh/lib"
if [[ "$INCLUDE_FP_LIB" == "1" ]]; then
    echo "   (re-pointing v2d_foundation_pose/lib; native .so must be unchanged)"
    _install "$(_pip foundation_pose)" -e "$MODULES/v2d_foundation_pose/lib"
else
    echo "   (keeping BAKED v2d_foundation_pose/lib — native code. Use --include-fp-lib to override.)"
fi

# Orchestration layer on the base interpreter (drivers, wrappers, pipelines).
echo "== base (orchestration) =="
pip install --no-deps --no-build-isolation \
    -e "$MODULES/v2d_common" -e "$MODULES/v2d_docker" \
    -e "$MODULES/v2d_mv" -e "$MODULES/v2d_mesh/lib"
for w in anycalib moge grounding_dino sam2 wilor sam3d foundation_pose gsplat_refinement; do
    pip install --no-deps --no-build-isolation -e "$MODULES/v2d_${w}/docker"
done
pip install --no-deps --no-build-isolation -e "$MODULES/v2d_pipelines"

echo
echo "Repos re-pointed at: $MODULES"
