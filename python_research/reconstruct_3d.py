"""The Met objects photographed only once, turning in the hand all the same: TripoSR (third_party/TripoSR,
single image -> 3D, Tochilkin et al. 2024) reconstructs each from its main cut-out and renders it all
around; the object then turns in the hand like those photographed from several sides (met_views.py).

The model stands an object photographed from a little above tilted: turned about its own axis, it would
lean one way, then the other. So the orbit of the cameras is tilted too, by the angle (about the axis
across the view) at which the object's silhouette leans least over a turn (best_tilt). An object the
catalog turns to lie lengthwise (a sword) is reconstructed upright, as photographed, so that it turns about
its own length. N_VIEWS views (512 px, 15 degrees apart) are rendered all around with their alpha and kept
in data/objects/ai_renders/<object>/ (the reconstruction is not redone unless --force); for the hand, each
is cleared of the model's faint haze, cropped, enlarged by Real-ESRGAN (4x) and brought to the main
cut-out's height (at most AI_VIEW_PX; lying objects: its length).

In the hand the object starts from its real photograph (the main cut-out), fades into the reconstruction,
makes one turn (one view per VIEW_S step: 6 s) and fades back to the photograph - every step a view of
data/objects/views/<object>/, listed in data/objects/ai_views.json, which load_catalog() reads. The back of
the object is the model's guess: the photograph shows only its front. REVIEW lists the objects whose
reconstruction looked right on the review sheet (--sheet: data/objects/ai_views_preview.jpg).

Usage: python reconstruct_3d.py [--force] [--sheet] [--only name,name]"""
import argparse
import json
import math
import re
import sys
import types

import cv2
import numpy as np

from common.handheld import CATALOG_DIR, _bleed, catalog_entries, sprite_path
from common.paths import ROOT
from common.imaging import write_image
from met_views import DWELL, TRANSITION, between, on_canvas, straight

sys.path.insert(0, str(ROOT / "third_party" / "TripoSR"))
for unused in ("torchmcubes", "rembg", "trimesh"):  # mesh extraction, background removal, mesh export: not used
    sys.modules.setdefault(unused, types.SimpleNamespace(marching_cubes=None))

RENDERS = CATALOG_DIR / "ai_renders"
VIEWS_DIR = CATALOG_DIR / "views"
OUT = CATALOG_DIR / "ai_views.json"
N_VIEWS, SIZE = 24, 512  # views of one turn (15 degrees apart), render size
AI_VIEW_PX = 800  # the tallest view: a 512 px render holds no more detail (the photograph itself stays whole)
OBJECTS = ["spearhead_bronze", "amphora_small_cypriot", "shaffron_japanese", "shaffron_dutch", "shield_indian",
           "shield_french", "breastplate_italian", "javelin_japanese", "dagger_mughal", "sword_ottoman",
           "sword_two_handed", "saber_presentation"]
# kept after the review sheet; left out: the Japanese javelin (a loose flap floats beside its tip as it turns) and
# the two-handed sword (blurred, in a grey haze)
REVIEW = set(OBJECTS) - {"javelin_japanese", "sword_two_handed"}


def load_model():
    """TSR.from_pretrained, with the checkpoint's ViT keys renamed as the installed transformers names them."""
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from tsr.models import nerf_renderer
    from tsr.system import TSR

    nerf_renderer.TriplaneNeRFRenderer._forward = _forward_rgba
    cfg = OmegaConf.load(hf_hub_download("stabilityai/TripoSR", "config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(hf_hub_download("stabilityai/TripoSR", "model.ckpt"), map_location="cpu")
    names = {"attention.attention.query": "attention.q_proj", "attention.attention.key": "attention.k_proj",
             "attention.attention.value": "attention.v_proj", "attention.output.dense": "attention.o_proj",
             "intermediate.dense": "mlp.fc1", "output.dense": "mlp.fc2"}
    renamed = {}
    for k, v in ckpt.items():
        m = re.match(r"(image_tokenizer\.model\.)encoder\.layer\.(\d+)\.(.+)", k)
        if m:
            rest = m.group(3)
            for a, b in names.items():
                if rest.startswith(a):
                    rest = b + rest[len(a):]
                    break
            k = f"{m.group(1)}layers.{m.group(2)}.{rest}"
        renamed[k] = v
    model.load_state_dict(renamed)
    model.renderer.set_chunk_size(8192)
    return model.to("cuda")


def _forward_rgba(self, decoder, triplane, rays_o, rays_d, **kwargs):
    """TriplaneNeRFRenderer._forward, returning the colour premultiplied by the opacity and the opacity
    (the original composites onto white)."""
    import torch
    from tsr.utils import rays_intersect_bbox

    shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    t_near, t_far, valid = rays_intersect_bbox(rays_o, rays_d, self.cfg.radius)
    t_near, t_far = t_near[valid], t_far[valid]
    t_vals = torch.linspace(0, 1, self.cfg.num_samples_per_ray + 1, device=triplane.device)
    t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
    z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]
    xyz = rays_o[valid][:, None, :] + z_vals[..., None] * rays_d[valid][..., None, :]
    out = self.query_triplane(decoder=decoder, positions=xyz, triplane=triplane)
    alpha = 1 - torch.exp(-(t_vals[1:] - t_vals[:-1]) * out["density_act"][..., 0])
    acc = torch.cat([torch.ones_like(alpha[:, :1]), torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)], dim=-1)
    w = alpha * acc
    rgba = torch.zeros(rays_o.shape[0], 4, device=triplane.device)
    rgba[valid, :3] = (w[..., None] * out["color"]).sum(dim=-2)
    rgba[valid, 3] = w.sum(dim=-1)
    return rgba.view(*shape, 4)


def cameras(n: int, size: int, tilt_deg: float, distance: float = 1.9, fovy_deg: float = 40.0):
    """Rays of n views around the scene (TripoSR's cameras: z up, elevation 0), the whole orbit turned by
    tilt_deg about the y axis (the axis across the input view)."""
    import torch
    import torch.nn.functional as F
    from tsr.utils import get_ray_directions, get_rays

    az = torch.linspace(0, 2 * math.pi, n + 1)[:n]
    pos = torch.stack([distance * torch.cos(az), distance * torch.sin(az), torch.zeros_like(az)], -1)
    look = F.normalize(-pos, dim=-1)
    right = F.normalize(torch.linalg.cross(look, torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)), dim=-1)
    up = F.normalize(torch.linalg.cross(right, look), dim=-1)
    c2w = torch.zeros(n, 4, 4)
    c2w[:, :3, :3] = torch.stack([right, up, -look], -1)
    c2w[:, :3, 3] = pos
    c2w[:, 3, 3] = 1
    t = math.radians(tilt_deg)
    turn = torch.eye(4)
    turn[0, 0], turn[0, 2], turn[2, 0], turn[2, 2] = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
    focal = 0.5 * size / math.tan(0.5 * math.radians(fovy_deg))
    d = get_ray_directions(H=size, W=size, focal=1.0)[None].repeat(n, 1, 1, 1)
    d[..., :2] = d[..., :2] / focal
    return get_rays(d, turn @ c2w, keepdim=True, normalize=True)


def render(model, code, n: int, size: int, tilt: float) -> list[np.ndarray]:
    """n premultiplied RGBA float views (0-1) of the scene code."""
    import torch

    rays_o, rays_d = cameras(n, size, tilt)
    with torch.no_grad():
        return [model.renderer(model.decoder, code, rays_o[i].cuda(), rays_d[i].cuda()).cpu().numpy() for i in range(n)]


def lean(alpha: np.ndarray) -> float:
    """How far a silhouette leans: its top fifth's centre minus its bottom fifth's, in heights."""
    ys = np.nonzero(alpha.max(1) > 0.5)[0]
    if len(ys) < 10:
        return 0.0
    y1, y2 = ys.min(), ys.max()
    h, cols = y2 - y1 + 1, np.arange(alpha.shape[1])

    def centre(lo: int, hi: int) -> float:
        w = alpha[lo:hi].sum(0)
        return float((w * cols).sum() / max(w.sum(), 1e-6))
    return (centre(y1, y1 + h // 5 + 1) - centre(y2 - h // 5, y2 + 1)) / h


def best_tilt(model, code) -> float:
    """The orbit's tilt (-30..30 degrees) at which the silhouette leans least over a turn of 12 views."""
    scores = {}
    for tilt in np.arange(-30, 31, 2.0):
        leans = [lean(v[..., 3]) for v in render(model, code, 12, 96, float(tilt))]
        scores[float(tilt)] = float(np.sqrt(np.mean(np.square(leans))))
    return min(scores, key=scores.get)


def reconstruct(model, name: str, entry: dict) -> None:
    """Renders the object's turn into RENDERS/<name>/ (straight BGRA, the catalog's orientation)."""
    import torch
    from PIL import Image
    from tsr.utils import resize_foreground

    sprite = cv2.imread(str(sprite_path(entry)), cv2.IMREAD_UNCHANGED)
    rot = entry.get("rotate", 0)
    if rot:  # back to the photo's upright pose: the turn is then about the object's own length
        sprite = np.ascontiguousarray(np.rot90(sprite, -rot // 90))
    image = np.asarray(resize_foreground(Image.fromarray(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)), 0.85)).astype(np.float32) / 255
    rgb = image[..., :3] * image[..., 3:] + 0.5 * (1 - image[..., 3:])  # on grey, as TripoSR was trained
    with torch.no_grad():
        code = model([Image.fromarray((rgb * 255).astype(np.uint8))], device="cuda")[0]
    tilt = best_tilt(model, code)
    folder = RENDERS / name
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in enumerate(render(model, code, N_VIEWS, SIZE, tilt)):
        v = np.clip(v, 0, 1)
        colour = v[..., :3] / np.maximum(v[..., 3:], 1e-3)
        out = np.clip(np.dstack([colour[..., ::-1], v[..., 3:]]) * 255, 0, 255).astype(np.uint8)
        if rot:
            out = np.ascontiguousarray(np.rot90(out, rot // 90))
        cv2.imwrite(str(folder / f"{k:02d}.png"), out)
    (folder / "tilt.txt").write_text(f"{tilt}\n", encoding="utf-8")
    print(f"{name}: tilt {tilt:+.0f} degrees", flush=True)


def premultiplied_clean(render: np.ndarray) -> np.ndarray:
    """A render (straight uint8 BGRA) as premultiplied float BGRA without the reconstruction's haze: the
    faint density the model leaves around the object (alpha under 10%, and anything not within a few pixels
    of its solid body: the parts over half opaque that are at least 5% of the largest) is cleared, so that
    the object can be cropped tight."""
    alpha = render[..., 3].astype(np.float32)
    solid = (alpha > 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.isin(labels, 1 + np.nonzero(areas >= 0.05 * areas.max())[0])
        near = cv2.dilate(keep.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # its soft edge stays
        alpha = np.where(near & (alpha > 25), alpha, 0)
    return np.dstack([render[..., :3].astype(np.float32) * alpha[..., None] / 255, alpha])


def views(name: str, entry: dict) -> list[str]:
    """The object's turning steps (data/objects/views/<name>/): its photograph, the reconstruction's turn,
    the photograph again - super-resolved, at one height (lying: one length)."""
    from common.composition import super_resolve

    main = cv2.imread(str(sprite_path(entry)), cv2.IMREAD_UNCHANGED)
    renders = []
    for k in range(N_VIEWS):
        r = cv2.imread(str(RENDERS / name / f"{k:02d}.png"), cv2.IMREAD_UNCHANGED)
        renders.append(super_resolve(straight(premultiplied_clean(r))))
    lengthwise = "width_m" in entry
    canvases = [straight(c) for c in on_canvas([main] + renders, lengthwise, AI_VIEW_PX)[0]]
    photo, turn = canvases[0], canvases[1:] + [canvases[1]]  # one whole turn, back to where it started
    t = np.arange(1, TRANSITION + 1) / (TRANSITION + 1)
    steps = [photo] * DWELL + [between(photo, turn[0], s, lengthwise) for s in t] + turn
    steps += [between(turn[-1], photo, s, lengthwise) for s in t]
    folder = VIEWS_DIR / name
    for old in folder.glob("*.png"):
        old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    order, seen = [], {}
    for s in steps:
        if id(s) not in seen:
            seen[id(s)] = len(seen)
            if s is not photo:
                cv2.imwrite(str(folder / f"{seen[id(s)]:02d}.png"), _bleed(s), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        order.append(seen[id(s)])
    return [name if k == 0 else f"views/{name}/{k:02d}" for k in order]  # step 0: the main cut-out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct the Met objects photographed once (TripoSR)")
    parser.add_argument("--force", action="store_true", help="Reconstruct again")
    parser.add_argument("--sheet", action="store_true", help="Also save ai_views_preview.jpg")
    parser.add_argument("--only", help="Comma-separated objects")
    args = parser.parse_args()
    entries = {sprite_path(e).stem: e for e in catalog_entries()}
    names = args.only.split(",") if args.only else OBJECTS
    todo = [n for n in names if args.force or not (RENDERS / n / f"{N_VIEWS - 1:02d}.png").exists()]
    if todo:
        model = load_model()
        for name in todo:
            reconstruct(model, name, entries[name])
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        if name in REVIEW:
            result[name] = views(name, entries[name])
            print(f"{name}: {len(set(result[name]))} views")
        else:
            result.pop(name, None)
            for old in (VIEWS_DIR / name).glob("*.png"):
                old.unlink()
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved: {OUT} ({len(result)} objects)")
    if args.sheet:
        rows = []
        for name in names:
            main = cv2.imread(str(sprite_path(entries[name])), cv2.IMREAD_UNCHANGED)
            tiles = [main] + [cv2.imread(str(RENDERS / name / f"{k:02d}.png"), cv2.IMREAD_UNCHANGED) for k in range(0, N_VIEWS, 3)]
            row = []
            for v in tiles:
                a = v[..., 3:] / 255.0
                t = (a * v[..., :3] + (1 - a) * 200).astype(np.uint8)
                row.append(cv2.resize(t, (max(1, round(t.shape[1] * 180 / t.shape[0])), 180), interpolation=cv2.INTER_AREA))
            row = np.hstack(row)
            rows.append(row)
        width = max(r.shape[1] for r in rows)
        sheet = np.vstack([np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
        print("Saved:", write_image(CATALOG_DIR / "ai_views_preview.jpg", sheet))


if __name__ == "__main__":
    main()
