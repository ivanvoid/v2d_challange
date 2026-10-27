# CONTAINER_UPDATE.md — what to bake into the container for the beta models

The beta pipeline (`run_beta.sh`, see its header) needs four venvs that the container image does not have yet.
So far they were created by hand inside the running container with `setup_beta_envs.sh`. They live under
`/opt/venvs`, which is part of the container filesystem, so **they are lost when the image is rebuilt or the
container is replaced**. This file lists everything to add to the image so that nobody has to redo it.

Anything marked "(not verified)" was not tested when this was written. The setup logs are in
`logs/beta/setup_<venv>.log`.

**Submission packer dependency discovered in the stage-8 ablations:**
`$V2D_DATA/.venv-submit` also needs `Pillow==11.3.0` to load a textured
`mesh_scaled.obj` through Trimesh in `track1/build_episode.py`. Without it,
stage 8 inference succeeds but packing fails with `No module named 'PIL'`.
Install Pillow when rebuilding `.venv-submit`; the current container's venv has it.

## 1. What the container looks like today (found by inspection)

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 |
| CUDA toolkit | `/usr/local/cuda`, nvcc 12.4.131 (`CUDA_HOME=/usr/local/cuda`) |
| Compiler / build tools | gcc/g++ 11.4, git, git-lfs, ninja (from `/opt/conda/bin`) |
| Conda | `/opt/conda` (Python 3.11) |
| System Python | `/usr/bin/python3.10` **without** the `venv`/`ensurepip` module (`python3.10-venv` missing) |
| Production venvs | `/opt/venvs/<name>`, created with `/opt/conda/bin/python -m venv --system-site-packages`, torch 2.5.1+cu124 pinned in `/opt/pip-constraints.txt` |
| Project packages | `/workspace/v2d_*` (installed with `pip install -e`) |
| GPUs | 3x A40 (46 GB, sm_86). GPU 1 is reserved/shared: the pipeline never uses it |
| Shared store | `/mnt/tepig/models` (holds gated DINOv3; must be mounted) |

## 2. What the beta models add

| Venv | Python | torch | Extra | Used by |
|---|---|---|---|---|
| `/opt/venvs/vda` | 3.10 | 2.1.1+cu121 | xformers 0.0.23, numpy 1.24.0, einops 0.4.1 | `BETA_DEPTH=vda` (stage 1) |
| `/opt/venvs/trellis2` | 3.10 | 2.6.0+cu124 | flash-attn 2.7.3, nvdiffrast v0.4.0, nvdiffrec, CuMesh, FlexGEMM, o-voxel (all compiled), utils3d@9a4eb15 | `BETA_MESH=trellis2` (stage 5) |
| `/opt/venvs/hunyuan` | 3.10 | 2.6.0+cu124 | Hunyuan3D-2 requirements, `custom_rasterizer` + `differentiable_renderer` (compiled) | `BETA_MESH=hunyuan` (stage 5) |
| `/opt/venvs/cotracker` | 3.10 | 2.6.0+cu124 | einops, scipy, trimesh | `BETA_TRACK=cotracker` (stage 7) |

Differences from the production venvs, which matter when baking them:
- These are **self-contained Python 3.10 venvs made by `uv`** (no `--system-site-packages`), because the
  torch versions differ from the pinned 2.5.1. Their `bin/python` is a symlink to `/usr/bin/python3.10`, so the
  image must keep that interpreter.
- `vda` uses torch 2.1.1 on cu121 (what Video-Depth-Anything's requirements ask for). The cu121 wheels bring their own CUDA libs, so no extra toolkit is needed for it.
- `trellis2` and `hunyuan` **compile CUDA code at install time**. They need nvcc 12.x, gcc, ninja,
  `TORCH_CUDA_ARCH_LIST=8.6` (A40) and plenty of RAM/CPU (`MAX_JOBS=8`). The build is slow, and `flash-attn` is the
  step most likely to fail or take long when no prebuilt wheel matches.
- The model code itself is **not installed**: it is imported from `models/<repo>` via `sys.path`.

## 3. Changes to make to the image

1. **Install `uv`** system-wide so `setup_beta_envs.sh` works without tricks. Today it is only in
   `~/.local/bin` (installed with `/opt/conda/bin/pip install --user uv`; the astral.sh installer was blocked
   by a connection reset). In the image:
   ```dockerfile
   RUN /opt/conda/bin/pip install uv     # gives /opt/conda/bin/uv
   ```
   Alternative: `apt-get install -y python3.10-venv` (setup then uses `python3.10 -m venv`).
2. **Make sure these exist** (already true now, do not drop them): `/usr/bin/python3.10`, nvcc 12.4 in `/usr/local/cuda`,
   gcc/g++, git, ninja, network access to github.com and download.pytorch.org at build time.
3. **Create the venvs at build time** with the repo script (needs `models/` present, see section 4):
   ```dockerfile
   ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 V2D_DATA=/local-data/ivan/v2d
   RUN ./setup_beta_envs.sh vda cotracker trellis2 hunyuan
   ```
   Hunyuan/TRELLIS.2 builds do not need a GPU (only nvcc), so they can run in `docker build`.
   (Dockerfile lines above are a sketch, not verified.)
4. **Freeze the result** so a rebuild is reproducible even when upstream versions move (see section 6).

## 4. What is NOT in the image (shared storage, restore with the scripts)

| What | Where | How to restore |
|---|---|---|
| Model repos | `models/` | `./fetch_models.sh repos` (clones with submodules) |
| Weights | `weights/` | `./fetch_models.sh weights` (idempotent; `ONLY="vda cotracker trellis2 hunyuan3d" ...` for just the beta ones) |
| Gated HF files | `HF_TOKEN` env | export before `fetch_models.sh` (never hardcode) |
| DINOv3 ViT-L | `weights/trellis2/dinov3-vitl16-pretrain-lvd1689m` | symlink to `/mnt/tepig/models/facebook/dinov3-vitl16-pretrain-lvd1689m`; otherwise accept the gated licence on HF |
| Challenge data | `challenge/` | `git clone https://huggingface.co/datasets/nvidia/video_to_data_challenge challenge` |

`setup_beta_envs.sh` reads `models/Video-Depth-Anything`, `models/TRELLIS.2` (incl. `o-voxel`) and `models/Hunyuan3D-2`,
so run `fetch_models.sh repos` before the venv step.

Known wart: `models/Video-Depth-Anything/checkpoints` is a symlink to `/backup/ivan/v2d/weights/...`, which does not
exist in this container. It is harmless (the beta code reads `weights/video_depth_anything/checkpoints`), and
`fetch_models.sh` re-links it.

## 5. Verify after a rebuild

```bash
for v in vda trellis2 hunyuan cotracker; do /opt/venvs/$v/bin/python -c "import torch;print('$v', torch.__version__, torch.cuda.is_available())"; done
# the import checks that setup_beta_envs.sh ends with ('<venv> ok' = last line of each setup log):
tail -n1 logs/beta/setup_{vda,trellis2,hunyuan,cotracker}.log
# cheap end-to-end test of the tracking-only path (stages 7-8, one GPU, never GPU 1):
BETA_DEPTH=moge BETA_MESH=sam3d BETA_TRACK=cotracker EPISODES="6" GPU=0 ./run_beta.sh
```

Status of the venv creation in the current container (fill in; it was still running when this was written):
- [x] cotracker: `cotracker ok`
- [x] vda: `vda ok`
- [ ] trellis2
- [x] hunyuan: built; the script's own import check failed only because it imported `custom_rasterizer` before
  `torch` (`libc10.so` not found). With `import torch` first it passes (`hunyuan ok 2.6.0+cu124 True`). The
  script's check is fixed (`import torch` first).
  **Pin `transformers==4.56.2` and `diffusers==0.35.2` in this venv.** Unpinned installs gave transformers
  5.20.2, whose DINOv2 parameter names (`q_proj`) do not match the Hunyuan checkpoint (`attention.attention.query`):
  `RuntimeError: Error(s) in loading state_dict for SingleImageEncoder`. Pinning transformers down also pulls
  `huggingface_hub` below 1.0, which breaks the unpinned diffusers 0.41 (`cannot import name 'resolve_revision'`),
  so diffusers must be pinned too. Both are in `setup_beta_envs.sh`.
- [x] trellis2: `trellis2 ok` (about 25 min, mostly the flash-attn and CUDA extension builds).
  **Pin `transformers==4.56.2` in this venv.** The unpinned install pulled transformers 5.20.2, whose
  `DINOv3ViTModel` has no `.layer` attribute, so TRELLIS.2's feature extractor crashed on the first frame
  (`AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'`) after a ~400 s model load. The script
  now pins it; in the current container it was fixed with `uv pip install transformers==4.56.2`.
  Lesson: the "ok" import check does not exercise the model; run `mesh_gen.py` on one frame after any rebuild.

## 6. Freeze the exact package versions (do this once everything works)

```bash
mkdir -p v2d_challange/container
for v in vda trellis2 hunyuan cotracker; do
  uv pip freeze --python /opt/venvs/$v/bin/python > v2d_challange/container/requirements_$v.txt
done
```
Compiled packages (nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel, custom_rasterizer,
differentiable_renderer, flash-attn) show up as local/git entries. Keep the git commits that
`setup_beta_envs.sh` clones: `nvdiffrast` is pinned to tag v0.4.0, but `nvdiffrec` (branch `renderutils`),
`CuMesh` and `FlexGEMM` are **unpinned**. Record their commit SHAs if exact reproduction matters.

## 7. Models still to add later (checkpoints/venvs not set up yet)

| Model | Repo in `models/` | Blocker |
|---|---|---|
| TAPNext++ | `tapnet` | weights are downloaded (`weights/tapnet`); the pipeline is not wired to use them; needs its own venv |
| CARI4D | `CARI4D` | weights downloaded (`weights/cari4d`); not wired; needs SMPL-H pkl and `kid_template.npy` (registration) |
| DROID-SLAM | `DROID-SLAM` | checkpoint missing (`tools/download_model.sh`, TRAM ships one) |
| FoundationStereo | `FoundationStereo` | checkpoint missing (Google Drive, see its README) |
| MegaPose / GigaPose / SAM-6D | `megapose6d`, `gigapose`, `SAM-6D` | checkpoints missing (see each README) |
| InterTrack | `InterTrack` | checkpoints missing |
| TRAM / WHAM / 4D-Humans / PromptHMR | `tram`, `WHAM`, `4D-Humans`, `PromptHMR` | need SMPL/SMPL-X registration; WHAM/TRAM fetch scripts prompt for credentials |

When one of these gets wired in: add a `setup_<name>()` function and its name to the `case` list in
`setup_beta_envs.sh`, add a row to the table in section 2, and add the venv to the freeze loop in section 6.
Do not rely on manual `pip install`s inside the running container, because they disappear with it.
