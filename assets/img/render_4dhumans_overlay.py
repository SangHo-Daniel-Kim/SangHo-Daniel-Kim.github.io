#!/usr/bin/env python3
"""Single image → 4D-Humans mesh overlay (semi-transparent) → JPG."""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# 4D-Humans
FOURD = "/data-local/sajakim/4D-Humans"
sys.path.insert(0, FOURD)
os.chdir(FOURD)

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
MESH_ALPHA = 0.42  # mesh opacity (lower = more transparent)


def render_overlay(img_path: str, out_path: str, mesh_alpha: float = MESH_ALPHA) -> None:
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    from detectron2.config import LazyConfig
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    import hmr2

    cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    det_cfg = LazyConfig.load(str(cfg_path))
    det_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(det_cfg)
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    det_out = detector(img_cv2)
    inst = det_out["instances"]
    valid = (inst.pred_classes == 0) & (inst.scores > 0.5)
    boxes = inst.pred_boxes.tensor[valid].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError("No person detected")

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts, all_cam_t = [], []
    img_size = None
    scaled_focal_length = None

    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_cam = out["pred_cam"]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        ).detach().cpu().numpy()

        for n in range(batch["img"].shape[0]):
            all_verts.append(out["pred_vertices"][n].detach().cpu().numpy())
            all_cam_t.append(pred_cam_t_full[n])

    cam_view = renderer.render_rgba_multiple(
        all_verts,
        cam_t=all_cam_t,
        render_res=img_size[0],
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )

    bg = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    mesh_rgb = cam_view[:, :, :3]
    mesh_a = cam_view[:, :, 3:4] * mesh_alpha
    overlay = bg * (1.0 - mesh_a) + mesh_rgb * mesh_a
    overlay_bgr = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved: {out_path} ({len(all_verts)} person(s), alpha={mesh_alpha})")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "/data-local/sajakim/SangHo-Daniel-Kim.github.io/assets/img/IMG_7244.JPG"
    stem = Path(inp).stem
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(inp).with_name(f"{stem}_4dhumans.jpg"))
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else MESH_ALPHA
    render_overlay(inp, out, alpha)
