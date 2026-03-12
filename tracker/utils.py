from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm.auto import tqdm

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

from .track import TrackState
from .tracker import Tracker


CLS_ID, SEP_ID, DOT_ID = 101, 102, 1012


def load_gdino_model(config_path: str, ckpt_path: str, device: str):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval().to(device)
    return model


def preprocess_frame_bgr(frame_bgr: np.ndarray):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_t, _ = transform(frame_pil, None)
    return frame_rgb, frame_t


def build_caption_from_labels(labels: List[str]) -> str:
    text = ". ".join([l.strip().lower() for l in labels if l.strip()])
    return text + ("" if text.endswith(".") else ".")


def phrase_spans_from_token_ids(input_ids: List[int]):
    cut = [i for i, t in enumerate(input_ids) if t in (CLS_ID, SEP_ID, DOT_ID)]
    spans = []
    for a, b in zip(cut, cut[1:]):
        s, e = a + 1, b
        if s < e:
            spans.append((s, e))
    return spans


def per_phrase_scores_from_token_logits(token_logits: torch.Tensor, tokenized, agg: str = "max") -> torch.Tensor:
    ids = tokenized["input_ids"]
    if not isinstance(ids, list):
        ids = ids[0].tolist()

    L = min(len(ids), token_logits.shape[1])
    tok = token_logits[:, :L]
    spans = phrase_spans_from_token_ids(ids)

    cols = []
    for s, e in spans:
        seg = tok[:, s:e]
        if seg.numel() == 0:
            cols.append(torch.zeros(tok.shape[0], device=tok.device))
        elif agg == "mean":
            cols.append(seg.mean(dim=1))
        else:
            cols.append(seg.max(dim=1).values)

    if len(cols) == 0:
        return torch.empty((tok.shape[0], 0), device=tok.device)

    return torch.stack(cols, dim=1)


def infer_with_phrase_scores(
    model,
    image_t: torch.Tensor,
    labels: List[str],
    box_threshold: float = 0.35,
    device: str = "cuda",
):
    caption = build_caption_from_labels(labels)
    model = model.to(device)
    image_t = image_t.to(device)

    with torch.no_grad():
        outputs = model(image_t[None], captions=[caption])

    pred_logits = outputs["pred_logits"].sigmoid()[0].detach().cpu()
    pred_boxes = outputs["pred_boxes"][0].detach().cpu()

    keep = pred_logits.max(dim=1).values > box_threshold
    logits_kept = pred_logits[keep]
    boxes_kept = pred_boxes[keep]

    tokenized = model.tokenizer(caption)
    phrase_scores = per_phrase_scores_from_token_logits(logits_kept, tokenized, agg="max")

    if phrase_scores.numel() > 0:
        best_scores, best_idx = phrase_scores.max(dim=1)
    else:
        best_scores = torch.empty(0)
        best_idx = torch.empty(0, dtype=torch.long)

    return boxes_kept, phrase_scores, best_idx, best_scores


def boxes_cxcywh_to_xyxy(boxes: torch.Tensor | np.ndarray, w: int, h: int) -> np.ndarray:
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype)
    b = boxes * scale
    return box_convert(b, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()


def boxes_xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float32)
    out = np.empty_like(xyxy)
    out[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    out[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    out[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    out[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return out


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    return x1, y1, x2, y2


def _stable_color(track_id: int) -> Tuple[int, int, int]:
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]
    return palette[track_id % len(palette)]


def _scale_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    factor = float(np.clip(factor, 0.15, 1.0))
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def draw_tracks(
    image_rgb: np.ndarray,
    tracks: Iterable,
    label_names: List[str],
    history: Dict[int, deque],
    max_history: int = 30,
    draw_trails: bool = True,
):
    img = image_rgb.copy()
    h, w = img.shape[:2]

    for track in tracks:
        if track.state == TrackState.TERMINATED:
            continue

        color = _stable_color(track.track_id)
        x1, y1, x2, y2 = _clip_box(track.tlbr, w, h)
        cx, cy = int(track.x), int(track.y)

        history[track.track_id].append((cx, cy))
        while len(history[track.track_id]) > max_history:
            history[track.track_id].popleft()

        if draw_trails:
            pts = list(history[track.track_id])
            n = len(pts)
            for i, pt in enumerate(pts):
                age = (i + 1) / max(n, 1)
                radius = max(1, int(round(1 + 5 * age)))  # older small, newer large
                dot_color = _scale_color(color, 0.25 + 0.75 * age)
                cv2.circle(img, pt, radius, dot_color, -1, cv2.LINE_AA)

        thickness = 2 if track.state == TrackState.ACTIVE else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label_idx = int(track.label)
        label_name = label_names[label_idx] if 0 <= label_idx < len(label_names) else f"label_{label_idx}"
        score = float(track.score_vector[label_idx]) if label_idx < len(track.score_vector) else 0.0
        text = f"id={track.track_id} {label_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(0, y1 - th - baseline - 4)
        box_right = min(w - 1, x1 + tw + 4)
        cv2.rectangle(img, (x1, ty), (box_right, y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return img


def run_tracker_on_video(
    model,
    video_path: str,
    output_path: str,
    labels: List[str],
    device: str,
    box_threshold: float = 0.35,
    max_frames: int | None = None,
    draw_trails: bool = True,
    show_progress: bool = True,
):
    video_path = str(video_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames is not None and total_frames > 0:
        total_to_process = min(total_frames, max_frames)
    elif max_frames is not None:
        total_to_process = max_frames
    else:
        total_to_process = total_frames if total_frames > 0 else None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Tracker()
    history = defaultdict(deque)

    pbar = tqdm(total=total_to_process, desc="Tracking video", unit="frame", disable=not show_progress)
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            frame_rgb, frame_t = preprocess_frame_bgr(frame_bgr)
            boxes_kept, phrase_scores, _, _ = infer_with_phrase_scores(
                model=model,
                image_t=frame_t,
                labels=labels,
                box_threshold=box_threshold,
                device=device,
            )

            if len(boxes_kept) > 0:
                xyxy = boxes_cxcywh_to_xyxy(boxes_kept, width, height)
                boxes_xywh = boxes_xyxy_to_xywh(xyxy)
                tracker.update(boxes_xywh.astype(np.float32), phrase_scores.numpy().astype(np.float32))
            else:
                tracker.update(
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0, len(labels)), dtype=np.float32),
                )

            vis = draw_tracks(
                frame_rgb,
                tracker.curr_tracks,
                labels,
                history,
                draw_trails=draw_trails,
            )
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    return output_path
