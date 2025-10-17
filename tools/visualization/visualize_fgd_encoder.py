"""
FGD Encoder Visualization Script

This script loads a distillation configuration (DEIMFGDDistiller), forwards a single image,
recomputes FGD spatial/channel attentions and foreground/background masks on selected pairs,
and saves overlay visualizations using the provided base image.

Usage example:
  python tools/visualization/visualize_fgd_encoder.py \
      -c configs/distill_deimv2_different/deimv2_hgnetv2_n_from_dinov3_s_encoder_distill.yml \
      -i 可视化1.jpg \
      -o outputs/visualization/fgd_encoder \
      -d cpu
"""

import os
import sys
import math
import json
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_tensor_image(im_pil: Image.Image, size_hw: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    t = T.Compose([
        T.Resize(size_hw),
        T.ToTensor(),
    ])
    return t(im_pil).unsqueeze(0)


def compute_attention(features: torch.Tensor, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (s_attention, c_attention) following engine/distill/fgd.py::_get_attention.
    s_attention shape: (N, H, W)
    c_attention shape: (N, C)
    """
    n, c, h, w = features.shape
    value = features.abs()
    spatial_map = value.mean(dim=1)  # (N, H, W)
    s_attn = (h * w) * F.softmax((spatial_map / temp).view(n, -1), dim=1)
    s_attn = s_attn.view(n, h, w)

    channel_map = value.mean(dim=(2, 3))  # (N, C)
    c_attn = c * F.softmax(channel_map / temp, dim=1)
    return s_attn, c_attn


def compute_masks_from_boxes(
    boxes_xyxy: torch.Tensor,
    img_wh: Tuple[int, int],
    feat_hw: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate FGD mask creation using (pseudo) boxes.

    boxes_xyxy: (M, 4) on original image pixels (float or int)
    img_wh: (W, H)
    feat_hw: (H, W)

    Returns: (mask_fg, mask_bg) with shapes (1, H, W)
    """
    w_img, h_img = img_wh
    h_feat, w_feat = feat_hw
    device = boxes_xyxy.device if isinstance(boxes_xyxy, torch.Tensor) else torch.device('cpu')
    dtype = torch.float32

    mask_fg = torch.zeros((1, h_feat, w_feat), device=device, dtype=dtype)
    mask_bg = torch.ones_like(mask_fg)

    if boxes_xyxy is None or (isinstance(boxes_xyxy, torch.Tensor) and boxes_xyxy.numel() == 0):
        # No boxes -> keep fg all zeros, bg normalized
        bg_sum = mask_bg.sum()
        if bg_sum > 0:
            mask_bg = mask_bg / bg_sum
        return mask_fg, mask_bg

    if not isinstance(boxes_xyxy, torch.Tensor):
        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
    else:
        boxes = boxes_xyxy.clone().to(dtype)

    # scale boxes to feature resolution
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(w_img) * float(w_feat)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(h_img) * float(h_feat)

    xmin = boxes[:, 0].floor().clamp(0, w_feat - 1).long()
    xmax = boxes[:, 2].ceil().clamp(0, w_feat - 1).long()
    ymin = boxes[:, 1].floor().clamp(0, h_feat - 1).long()
    ymax = boxes[:, 3].ceil().clamp(0, h_feat - 1).long()

    area = 1.0 / ((ymax - ymin + 1).clamp(min=1).float() * (xmax - xmin + 1).clamp(min=1).float())
    for box_id in range(boxes.size(0)):
        y0, y1 = ymin[box_id], ymax[box_id]
        x0, x1 = xmin[box_id], xmax[box_id]
        current = mask_fg[0, y0:y1 + 1, x0:x1 + 1]
        mask_fg[0, y0:y1 + 1, x0:x1 + 1] = torch.maximum(current, area[box_id])

    mask_bg[0] = torch.where(mask_fg[0] > 0, torch.zeros_like(mask_bg[0]), mask_bg[0])
    bg_sum = mask_bg[0].sum()
    if bg_sum > 0:
        mask_bg[0] = mask_bg[0] / bg_sum
    return mask_fg, mask_bg


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def overlay_heatmap(
    base_img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    clip_quantile: Optional[float] = None,
    colormap: str = 'jet',
) -> np.ndarray:
    """Apply a chosen colormap to heatmap and overlay onto base image (BGR).

    - colormap: 'jet' (默认), 'turbo', or 'hot'. 推荐 'turbo'（Google 提供的更现代的热力图，红色为高值，蓝色为低值），
      也可使用 'hot'（黑-红-黄-白，焦点区域为红黄）。
    - clip_quantile: 如果提供 (0,1)，则将低于该分位值的像素抑制为 0，增强热点对比。
    """
    hm = heatmap.astype(np.float32)
    heat_u8: np.ndarray
    if clip_quantile is not None and 0.0 < clip_quantile < 1.0 and hm.size > 0:
        vals = hm[np.isfinite(hm)]
        if vals.size > 0:
            qv = float(np.quantile(vals, clip_quantile))
            hm = np.where(hm < qv, 0.0, hm)
        heat_u8 = normalize_to_uint8(hm)
    else:
        heat_u8 = normalize_to_uint8(hm)

    # 选择色图
    cm_map = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        # OpenCV 不原生支持 Turbo；这里提供近似替代：使用 'JET' 或 'HOT'。
        # 若需更严格的 Turbo，可自定义 LUT。
    }
    cm_key = cm_map.get(colormap.lower(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(heat_u8, cm_key)
    overlay = cv2.addWeighted(colored, alpha, base_img_bgr, 1.0 - alpha, 0)
    return overlay


def save_channel_bar(attn_t: torch.Tensor, attn_s: torch.Tensor, out_path: str, top_k: int = 32):
    """Save a channel-attention bar comparison (teacher vs student).

    Handles different channel dimensions gracefully by selecting Top-K independently
    for teacher and student, and then comparing distributions by rank rather than
    exact channel indices.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Move to CPU numpy
    t = attn_t.detach().cpu().numpy()[0]
    s = attn_s.detach().cpu().numpy()[0]

    # Select Top-K independently to avoid index mismatch when C_t != C_s
    K_t = min(top_k, len(t))
    K_s = min(top_k, len(s))
    K = min(K_t, K_s)

    t_top = np.sort(t)[-K:][::-1]
    s_top = np.sort(s)[-K:][::-1]

    plt.figure(figsize=(10, 4))
    width = 0.4
    x = np.arange(K)
    plt.bar(x - width/2, t_top, width=width, label='Teacher')
    plt.bar(x + width/2, s_top, width=width, label='Student')
    plt.title('Channel Attention (Top-K by rank)')
    plt.xlabel('Top-K rank (descending)')
    plt.ylabel('Attention weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_channel_heatmap(attn: torch.Tensor, out_path: str, height: int = 64):
    """将通道注意力向量渲染为热力图PNG。

    - 输入 attn: 形状为 (N, C)，取 N=1 的通道注意力。
    - 渲染方式：将 1D 的通道权重归一化到 0-255，复制到指定高度，形成 (H, C) 的灰度图，
      再使用 Jet 颜色映射生成彩色热图并保存。
    """
    # 移到 CPU，取 N=1
    vec = attn.detach().cpu().numpy()
    if vec.ndim == 2:
        vec = vec[0]
    vec = vec.astype(np.float32)
    # 归一化到 [0,255]
    min_v = float(np.min(vec))
    max_v = float(np.max(vec))
    if max_v > min_v:
        vec_n = ((vec - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        vec_n = np.zeros_like(vec, dtype=np.uint8)
    # 复制到高度，得到灰度图 (H, C)
    gray = np.tile(vec_n[np.newaxis, :], (height, 1))
    # Jet 颜色映射
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # 保存 BGR 图像
    cv2.imwrite(out_path, colored)


def save_channel_heatmap_diverging(
    attn: torch.Tensor,
    out_path: str,
    height: int = 64,
    center: float = 1.0,
    clip_quantile: Optional[float] = None,
):
    """以发散色图（Blue-White-Red）渲染通道注意力，并可选分位数裁剪。

    - 输入 attn: 形状为 (N, C)，取 N=1。
    - 发散色图：以 center（默认=1.0）为白色，中性；低于 center 为蓝色，高于 center 为红色；
      颜色强度与偏离 center 的幅度成正比。
    - 分位数裁剪：如果设置 clip_quantile（如 0.7），根据 |attn - center| 的分位数抑制小偏差，突出显著通道。
    """
    vec = attn.detach().cpu().numpy()
    if vec.ndim == 2:
        vec = vec[0]
    vec = vec.astype(np.float32)
    dev = vec - float(center)
    # 可选分位数裁剪：抑制小偏差
    if clip_quantile is not None and 0.0 < clip_quantile < 1.0:
        vals = np.abs(dev[np.isfinite(dev)])
        if vals.size > 0:
            thr = float(np.quantile(vals, clip_quantile))
            dev = np.where(np.abs(dev) < thr, 0.0, dev)
    # 归一化到 [-1, 1]
    max_abs = float(np.max(np.abs(dev))) if dev.size > 0 else 0.0
    if max_abs > 0:
        norm = (dev / max_abs).clip(-1.0, 1.0)
    else:
        norm = np.zeros_like(dev, dtype=np.float32)
    C = norm.shape[0]
    # 构造发散颜色（BGR）：负（蓝），零（白），正（红）
    B = np.zeros((C,), dtype=np.float32)
    G = np.zeros((C,), dtype=np.float32)
    R = np.zeros((C,), dtype=np.float32)
    neg_idx = np.where(norm < 0)[0]
    pos_idx = np.where(norm >= 0)[0]
    t_neg = -norm[neg_idx]  # [0,1]
    t_pos = norm[pos_idx]   # [0,1]
    # 负侧：白 -> 蓝（随 |neg| 增强）
    B[neg_idx] = 255.0
    G[neg_idx] = 255.0 * (1.0 - t_neg)
    R[neg_idx] = 255.0 * (1.0 - t_neg)
    # 正侧：白 -> 红（随 pos 增强）
    B[pos_idx] = 255.0 * (1.0 - t_pos)
    G[pos_idx] = 255.0 * (1.0 - t_pos)
    R[pos_idx] = 255.0
    B = B.clip(0, 255).astype(np.uint8)
    G = G.clip(0, 255).astype(np.uint8)
    R = R.clip(0, 255).astype(np.uint8)
    # 拼接为一行 (W=C, H=height)
    row = np.stack([B, G, R], axis=1)  # (C, 3)
    colored = np.tile(row[np.newaxis, :, :], (height, 1, 1))  # (H, C, 3)
    cv2.imwrite(out_path, colored)


def visualize_pair(
    pair: Dict[str, any],
    temp: float,
    pred_module,
    pred_encoded_feats: Tuple[torch.Tensor, ...],
    postprocessor,
    base_img_bgr: np.ndarray,
    orig_wh: Tuple[int, int],
    out_dir: str,
    score_thr: float = 0.4,
    max_boxes: int = 20,
    score_mode: str = 'fixed',
    score_quantile: float = 0.7,
    topk_boxes: Optional[int] = None,
    use_boxes_manual: bool = False,
    boxes_manual: Optional[torch.Tensor] = None,
    overlay_clip_quantile: Optional[float] = None,
):
    """Visualize one feature pair: attentions and masks.

    The `pred_module` and `pred_encoded_feats` are used to obtain pseudo boxes for
    foreground mask generation. They can come from either the teacher or the student
    (fallback) depending on checkpoint availability.
    """
    name = pair.get('name', 'pair')
    ensure_dir(out_dir)

    # Original image size for upsampling overlays
    w_img, h_img = orig_wh

    # Extract features
    s_feat: torch.Tensor = pair['student_feat']
    t_feat: torch.Tensor = pair['teacher_feat']
    n, c_s, h_s, w_s = s_feat.shape
    n_t, c_t, h_t, w_t = t_feat.shape
    assert n == 1, 'Only single-image visualization supported.'

    # Compute attentions
    s_attn_t, c_attn_t = compute_attention(t_feat, temp)
    s_attn_s, c_attn_s = compute_attention(s_feat, temp)

    # Boxes source: manual (e.g., ground-truth/file) or predicted via decoder+postprocessor
    if use_boxes_manual and boxes_manual is not None:
        boxes_sel = boxes_manual.to(t_feat.device)
    else:
        # Predictions (use decoder and postprocessor to get pseudo boxes)
        with torch.no_grad():
            teacher_outputs = pred_module.decoder(pred_encoded_feats, targets=None)
            postprocessor.deploy()
            w_img, h_img = orig_wh
            orig_size = torch.tensor([[w_img, h_img]], device=t_feat.device)
            labels, boxes, scores = postprocessor(teacher_outputs, orig_size)

        # Select boxes by score thresholding strategy and cap by max_boxes
        scores_np = scores[0].detach().cpu().numpy()
        if score_mode == 'quantile':
            thr = float(np.quantile(scores_np, score_quantile))
            keep = (scores[0] >= thr)
            boxes_sel = boxes[0][keep]
        elif score_mode == 'topk':
            k = int(topk_boxes or max_boxes)
            k = min(k, boxes[0].shape[0])
            idx = torch.argsort(scores[0], descending=True)[:k]
            boxes_sel = boxes[0][idx]
        else:
            keep = (scores[0] > score_thr)
            boxes_sel = boxes[0][keep]
        if boxes_sel.shape[0] > max_boxes:
            boxes_sel = boxes_sel[:max_boxes]

    # Compute masks at student feature resolution (use s_feat spatial size)
    mask_fg, mask_bg = compute_masks_from_boxes(boxes_sel, orig_wh, (h_s, w_s))

    # Upsample attention and mask to image size for overlay
    s_attn_t_up = F.interpolate(s_attn_t.unsqueeze(1), size=(h_img, w_img), mode='bilinear', align_corners=False)[0, 0]
    s_attn_s_up = F.interpolate(s_attn_s.unsqueeze(1), size=(h_img, w_img), mode='bilinear', align_corners=False)[0, 0]
    mask_fg_up = F.interpolate(mask_fg.unsqueeze(1), size=(h_img, w_img), mode='nearest')[0, 0]
    mask_bg_up = F.interpolate(mask_bg.unsqueeze(1), size=(h_img, w_img), mode='nearest')[0, 0]

    # Compute focal attention overlays: s_attn * mask_fg (resize mask to each feature map size first)
    # Resize mask to teacher feature resolution if available
    mask_fg_t = F.interpolate(mask_fg.unsqueeze(1), size=(h_t, w_t), mode='nearest')[0, 0] if (h_t > 0 and w_t > 0) else mask_fg[0]
    mask_fg_s = mask_fg[0]  # already at student feature resolution
    focal_t = (s_attn_t[0] * mask_fg_t)
    focal_s = (s_attn_s[0] * mask_fg_s)
    focal_t_up = F.interpolate(focal_t.unsqueeze(0).unsqueeze(0), size=(h_img, w_img), mode='bilinear', align_corners=False)[0, 0]
    focal_s_up = F.interpolate(focal_s.unsqueeze(0).unsqueeze(0), size=(h_img, w_img), mode='bilinear', align_corners=False)[0, 0]

    # Convert tensors to numpy
    s_attn_t_np = s_attn_t_up.detach().cpu().numpy()
    s_attn_s_np = s_attn_s_up.detach().cpu().numpy()
    mask_fg_np = mask_fg_up.detach().cpu().numpy()
    focal_t_np = focal_t_up.detach().cpu().numpy()
    focal_s_np = focal_s_up.detach().cpu().numpy()

    # Save overlays
    cm_name = getattr(args, 'overlay_colormap', 'jet')
    ov_t = overlay_heatmap(base_img_bgr, s_attn_t_np, alpha=0.6, clip_quantile=overlay_clip_quantile, colormap=cm_name)
    ov_s = overlay_heatmap(base_img_bgr, s_attn_s_np, alpha=0.6, clip_quantile=overlay_clip_quantile, colormap=cm_name)
    ov_mask = overlay_heatmap(base_img_bgr, mask_fg_np, alpha=0.6, colormap=cm_name)
    ov_mask_bg = overlay_heatmap(base_img_bgr, mask_bg_up.detach().cpu().numpy(), alpha=0.6, colormap=cm_name)
    ov_focal_t = overlay_heatmap(base_img_bgr, focal_t_np, alpha=0.6, clip_quantile=overlay_clip_quantile, colormap=cm_name)
    ov_focal_s = overlay_heatmap(base_img_bgr, focal_s_np, alpha=0.6, clip_quantile=overlay_clip_quantile, colormap=cm_name)

    cv2.imwrite(os.path.join(out_dir, f"{name}_s_attn_teacher_overlay.png"), ov_t)
    cv2.imwrite(os.path.join(out_dir, f"{name}_s_attn_student_overlay.png"), ov_s)
    cv2.imwrite(os.path.join(out_dir, f"{name}_mask_fg_overlay.png"), ov_mask)
    cv2.imwrite(os.path.join(out_dir, f"{name}_mask_bg_overlay.png"), ov_mask_bg)
    cv2.imwrite(os.path.join(out_dir, f"{name}_s_attn_teacher_focal_overlay.png"), ov_focal_t)
    cv2.imwrite(os.path.join(out_dir, f"{name}_s_attn_student_focal_overlay.png"), ov_focal_s)

    # Save channel attention bar chart
    save_channel_bar(c_attn_t, c_attn_s, os.path.join(out_dir, f"{name}_c_attn_topk.png"))
    # Save channel attention heatmaps (Jet colormap)
    save_channel_heatmap(c_attn_t, os.path.join(out_dir, f"{name}_c_attn_teacher_heatmap.png"))
    save_channel_heatmap(c_attn_s, os.path.join(out_dir, f"{name}_c_attn_student_heatmap.png"))
    # Save diverging channel heatmaps with optional quantile clipping around center=1.0
    save_channel_heatmap_diverging(
        c_attn_t,
        os.path.join(out_dir, f"{name}_c_attn_teacher_diverging_q70.png"),
        height=64,
        center=1.0,
        clip_quantile=0.7,
    )
    save_channel_heatmap_diverging(
        c_attn_s,
        os.path.join(out_dir, f"{name}_c_attn_student_diverging_q70.png"),
        height=64,
        center=1.0,
        clip_quantile=0.7,
    )
    # Save diverging channel heatmaps WITHOUT clipping (center=1.0, no quantile suppression)
    save_channel_heatmap_diverging(
        c_attn_t,
        os.path.join(out_dir, f"{name}_c_attn_teacher_diverging_noclip.png"),
        height=64,
        center=1.0,
        clip_quantile=None,
    )
    save_channel_heatmap_diverging(
        c_attn_s,
        os.path.join(out_dir, f"{name}_c_attn_student_diverging_noclip.png"),
        height=64,
        center=1.0,
        clip_quantile=None,
    )

    # Save raw numpy arrays for reproducibility
    s_attn_t_raw = s_attn_t.detach().cpu().numpy()
    s_attn_s_raw = s_attn_s.detach().cpu().numpy()
    mask_fg_raw = mask_fg.detach().cpu().numpy()
    mask_bg_raw = mask_bg.detach().cpu().numpy()
    focal_t_raw = focal_t.detach().cpu().numpy()
    focal_s_raw = focal_s.detach().cpu().numpy()
    c_attn_t_raw = c_attn_t.detach().cpu().numpy()
    c_attn_s_raw = c_attn_s.detach().cpu().numpy()

    # Normalize masks to [0,1] for saving, matching FGD visualization convention
    def _minmax01(x):
        x = x.astype(np.float32)
        min_v = float(x.min())
        max_v = float(x.max())
        if max_v > min_v:
            x = (x - min_v) / (max_v - min_v)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        return x

    mask_fg_save = _minmax01(mask_fg_raw)
    mask_bg_save = _minmax01(mask_bg_raw)

    np.save(os.path.join(out_dir, f"{name}_s_attn_teacher.npy"), s_attn_t_raw)
    np.save(os.path.join(out_dir, f"{name}_s_attn_student.npy"), s_attn_s_raw)
    np.save(os.path.join(out_dir, f"{name}_mask_fg.npy"), mask_fg_save)
    np.save(os.path.join(out_dir, f"{name}_mask_bg.npy"), mask_bg_save)
    np.save(os.path.join(out_dir, f"{name}_s_attn_teacher_focal.npy"), focal_t_raw)
    np.save(os.path.join(out_dir, f"{name}_s_attn_student_focal.npy"), focal_s_raw)
    np.save(os.path.join(out_dir, f"{name}_c_attn_teacher.npy"), c_attn_t_raw)
    np.save(os.path.join(out_dir, f"{name}_c_attn_student.npy"), c_attn_s_raw)


def main(args):
    # Build config
    cfg = YAMLConfig(args.config)

    # Avoid backbone trying to download pretrained weights
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # Instantiate model (DEIMFGDDistiller)
    distiller = cfg.model
    device = torch.device(args.device)
    distiller = distiller.to(device)

    # Allow overriding teacher/student ckpt by CLI
    # Override ckpt paths if provided (mutate distiller directly)
    if args.teacher_ckpt:
        distiller.teacher_ckpt = args.teacher_ckpt
    if args.student_ckpt:
        distiller.student_ckpt = args.student_ckpt

    # Load checkpoints (teacher may be missing)
    use_student_for_boxes = False
    if getattr(distiller, 'teacher_ckpt', None):
        if os.path.exists(distiller.teacher_ckpt):
            distiller.load_teacher_checkpoint(distiller.teacher_ckpt)
        else:
            print(f"[visualize_fgd_encoder] teacher_ckpt not found: {distiller.teacher_ckpt}. Will use student predictions for masks.")
            use_student_for_boxes = True
    if getattr(distiller, 'student_ckpt', None):
        # Use non-strict for student to tolerate head mismatches
        distiller.load_student_checkpoint(distiller.student_ckpt, strict=False)
    distiller.set_teacher_eval()
    distiller.eval()

    # Force dynamic anchor generation to match current feature resolution
    # Avoid mismatches when eval_spatial_size in YAML differs from the actual input size
    try:
        if hasattr(distiller, 'student') and hasattr(distiller.student, 'decoder'):
            distiller.student.decoder.eval_spatial_size = None
        if hasattr(distiller, 'teacher') and hasattr(distiller.teacher, 'decoder'):
            distiller.teacher.decoder.eval_spatial_size = None
    except Exception as _:
        pass

    # Ensure feature pairs active regardless of start_epoch in YAML
    distiller.set_epoch(10**6)

    # Load base image
    im_pil = Image.open(args.input).convert('RGB')
    w_img, h_img = im_pil.size
    base_bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    # Allow custom input resolution for resizing (square)
    input_size = int(getattr(args, 'input_size', 640) or 640)
    img_tensor = to_tensor_image(im_pil, size_hw=(input_size, input_size)).to(device)

    # Forward to get pairs and teacher encoded feats
    with torch.no_grad():
        outputs = distiller(img_tensor, targets=None)

    distill_pairs: List[Dict[str, any]] = outputs.get('distill_pairs', [])
    teacher_outs = outputs.get('teacher_outputs', {})
    teacher_encoded_feats = teacher_outs.get('encoded_feats', tuple())
    student_encoded_feats = outputs.get('encoded_feats', tuple())
    if len(distill_pairs) == 0:
        raise RuntimeError('No active distillation feature pairs; check the config and start_epoch settings.')
    if (len(teacher_encoded_feats) == 0 and not use_student_for_boxes) and all((p.get('meta') or {}).get('source','backbone')=='encoder' for p in distill_pairs):
        raise RuntimeError('Teacher encoded features not available; verify meta.source=="encoder" in feature_pairs.')

    # Temp from DEIMCriterion.distill_cfg per pair; fallback to 0.9
    # Build a name->temp dict by scanning YAML
    pair_temp = {}
    try:
        distill_cfg = cfg.yaml_cfg.get('DEIMCriterion', {}).get('distill_cfg', [])
        for entry in distill_cfg:
            nm = entry.get('name')
            loss = entry.get('loss', {})
            tval = loss.get('temp', 0.9)
            if nm:
                pair_temp[nm] = float(tval)
    except Exception:
        pass

    # PostProcessor for predictions
    postprocessor = cfg.postprocessor.to(device)

    # Output directory
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_root = os.path.join(args.output, args.exp_name if args.exp_name else base_name)
    ensure_dir(out_root)

    # Helper: read boxes from file or COCO GT
    def read_boxes_from_file(path: Optional[str]) -> Optional[torch.Tensor]:
        if not path:
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            boxes = data.get('boxes')
            if boxes is None:
                return None
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            return boxes_t
        except Exception as e:
            print(f"[visualize_fgd_encoder] Failed to read boxes_file: {path}. Error: {e}")
            return None

    def read_boxes_from_coco_gt(ann_path: Optional[str], image_id: Optional[int]) -> Optional[torch.Tensor]:
        if not ann_path or image_id is None:
            return None
        try:
            with open(ann_path, 'r') as f:
                coco = json.load(f)
            anns = coco.get('annotations', [])
            boxes_xyxy = []
            for a in anns:
                if int(a.get('image_id', -1)) == int(image_id):
                    # COCO bbox format: [x,y,w,h] in image pixels
                    x, y, w, h = a.get('bbox', [0, 0, 0, 0])
                    x1 = float(x)
                    y1 = float(y)
                    x2 = float(x) + float(w)
                    y2 = float(y) + float(h)
                    # skip invalid boxes
                    if w > 0 and h > 0:
                        boxes_xyxy.append([x1, y1, x2, y2])
            if len(boxes_xyxy) == 0:
                print(f"[visualize_fgd_encoder] No GT boxes found for image_id={image_id} in {ann_path}")
                return None
            return torch.tensor(boxes_xyxy, dtype=torch.float32)
        except Exception as e:
            print(f"[visualize_fgd_encoder] Failed to read COCO GT: {ann_path}. Error: {e}")
            return None

    # Visualize each pair
    for pair in distill_pairs:
        nm = pair.get('name', 'pair')
        temp = float(pair_temp.get(nm, 0.9))
        pair_dir = os.path.join(out_root, nm)
        # choose source for pseudo boxes
        boxes_from = getattr(args, 'boxes_from', 'auto')
        use_boxes_student = use_student_for_boxes
        if boxes_from == 'student':
            use_boxes_student = True
        elif boxes_from == 'teacher':
            use_boxes_student = False
        elif boxes_from == 'gt' or boxes_from == 'file':
            pass  # handled below

        # Always define prediction modules for fallback (even when using manual boxes)
        if use_boxes_student:
            pred_module = distiller.student
            pred_encoded = student_encoded_feats
        else:
            pred_module = distiller.teacher
            pred_encoded = teacher_encoded_feats

        # Try to read manual boxes if requested
        boxes_manual = None
        use_boxes_manual = False
        if boxes_from == 'gt':
            boxes_manual = read_boxes_from_coco_gt(getattr(args, 'coco_ann', None), getattr(args, 'coco_image_id', None))
            use_boxes_manual = boxes_manual is not None
        elif boxes_from == 'file':
            boxes_manual = read_boxes_from_file(getattr(args, 'boxes_file', None))
            use_boxes_manual = boxes_manual is not None

        visualize_pair(
            pair=pair,
            temp=temp,
            pred_module=pred_module,
            pred_encoded_feats=pred_encoded,
            postprocessor=postprocessor,
            base_img_bgr=base_bgr,
            orig_wh=(w_img, h_img),
            out_dir=pair_dir,
            score_thr=args.score_thr,
            max_boxes=args.max_boxes,
            score_mode=args.score_mode,
            score_quantile=args.score_quantile,
            topk_boxes=args.topk_boxes,
            use_boxes_manual=use_boxes_manual,
            boxes_manual=boxes_manual,
            overlay_clip_quantile=getattr(args, 'overlay_clip_quantile', None),
        )

    print(f"Saved FGD encoder visualizations to: {out_root}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to distillation YAML config (encoder version).')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to base image for visualization overlays.')
    parser.add_argument('-o', '--output', type=str, default='outputs/visualization/fgd_encoder',
                        help='Output directory root for saved visualizations.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda).')
    parser.add_argument('--score_thr', type=float, default=0.4,
                        help='Score threshold for pseudo boxes from teacher (used when score_mode=fixed).')
    parser.add_argument('--max_boxes', type=int, default=20,
                        help='Maximum number of boxes to use for mask generation.')
    parser.add_argument('--score_mode', type=str, default='fixed', choices=['fixed', 'quantile', 'topk'],
                        help='Bounding box selection strategy: fixed threshold, quantile threshold, or top-k by score.')
    parser.add_argument('--score_quantile', type=float, default=0.7,
                        help='Quantile for dynamic thresholding when score_mode=quantile. Example: 0.7 keeps top 30% scores.')
    parser.add_argument('--topk_boxes', type=int, default=None,
                        help='Top-K boxes to keep by score when score_mode=topk. Defaults to max_boxes if not set.')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name to create a dedicated subfolder under output.')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='Override teacher checkpoint path at runtime (optional).')
    parser.add_argument('--student_ckpt', type=str, default=None,
                        help='Override student checkpoint path at runtime (optional).')
    parser.add_argument('--input_size', type=int, default=640,
                        help='Square input resize dimension (e.g., 640 or 320).')
    parser.add_argument('--overlay_colormap', type=str, default='jet', choices=['jet','hot'],
                        help='Colormap for spatial overlays: jet (default) or hot. Hot gives red-yellow focus with cooler background.')
    parser.add_argument('--boxes_from', type=str, default='auto', choices=['auto', 'teacher', 'student'],
                        help='(Deprecated) Select source for pseudo boxes used to build masks: teacher, student, or auto.')
    # Extended boxes source: allow GT or file-specified boxes for mask generation
    parser.add_argument('--boxes_source', type=str, default=None, choices=['gt', 'file'],
                        help='Use ground-truth boxes (COCO) or boxes from a JSON file for mask generation.')
    parser.add_argument('--boxes_file', type=str, default=None,
                        help='Path to a JSON file containing {"boxes": [[x1,y1,x2,y2], ...]} .')
    parser.add_argument('--coco_ann', type=str, default=None,
                        help='Path to COCO annotation JSON (instances_val2017.json etc.) when using boxes_source=gt.')
    parser.add_argument('--coco_image_id', type=int, default=None,
                        help='COCO image id to fetch GT boxes when using boxes_source=gt.')
    parser.add_argument('--overlay_clip_quantile', type=float, default=None,
                        help='分位数裁剪以增强高响应区域（如 0.7 表示抑制低于 70% 分位的响应，用于改善教师/学生热力图阈值显示）。')
    args = parser.parse_args()
    # Backward compatibility: if boxes_source provided, override boxes_from for internal routing
    if getattr(args, 'boxes_source', None) is not None:
        if args.boxes_source in ['gt', 'file']:
            setattr(args, 'boxes_from', args.boxes_source)
    main(args)