# TripoSR (inference code)

The `tsr/` package of [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (Stability AI and VAST, 2024;
Tochilkin et al., "TripoSR: Fast 3D Object Reconstruction from a Single Image", arXiv:2403.02151), MIT licence
(`LICENSE`), commit 107cefd (2026-06-04), unchanged. The weights (stabilityai/TripoSR,
~1.7 GB) download from Hugging Face on first use.

`python_research/reconstruct_3d.py` uses it to turn the Met objects photographed only once: it renders the
reconstruction with an alpha channel, renames the checkpoint's ViT keys for the installed transformers, and
does not need `torchmcubes`, `rembg` or `trimesh` (no mesh is extracted).
