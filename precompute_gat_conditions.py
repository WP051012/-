"""
预计算 GAT 目标嵌入 (64-dim)，用作 FOMAML 域感知 condition。

从 labels/ 目录的 YOLO 检测标注文件中加载场景数据，
通过 track_id 精确匹配目标行人，运行冻结 GAT 提取 target_emb。

运行方式:
    python precompute_gat_conditions.py \
        --perception-ckpt checkpoints/stage2_best.pt \
        --label-dir labels \
        --output data/gat_conditions.pt \
        --frame-idx 4 \
        --batch-size 128

Label 文件格式 (per-frame sections):
    ### Frame: blurred_{video}_{frame}.txt ###
    class_id xc yc w h track_id   (归一化坐标 0-1)

输出格式:
    {video_name: {"track_id__obs_start": tensor(64,), ...}, ...}
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.dataset import TrajectoryDataset
from src.perception_model import TrafficPerceptionModel

# COCO class_id → class_name
CLASS_ID_TO_NAME = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
}


# ---------------------------------------------------------------------------
# Label file parsing
# ---------------------------------------------------------------------------

def parse_label_file(
    path: str,
    img_width: float,
    img_height: float,
) -> Dict[int, List[dict]]:
    """
    Parse a YOLO-format label file into per-frame detection data.

    Returns: {frame_num: [{bbox, position, velocity, class_name, track_id}, ...]}
    """
    frames: Dict[int, List[dict]] = defaultdict(list)
    current_frame = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Frame header: ### Frame: ..._{frame_num}.txt ###
            if line.startswith("### Frame:"):
                m = re.search(r"_(\d+)\.txt", line)
                if m:
                    current_frame = int(m.group(1))
                continue

            if current_frame is None:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                track_id = int(parts[5])
            except (ValueError, IndexError):
                continue

            class_name = CLASS_ID_TO_NAME.get(class_id, "unknown")

            # Normalized → pixel coordinates
            # bbox: [x1, y1, x2, y2] in pixels
            half_w = w * img_width / 2
            half_h = h * img_height / 2
            cx = xc * img_width
            cy = yc * img_height

            frames[current_frame].append({
                "bbox": np.array([cx - half_w, cy - half_h,
                                  cx + half_w, cy + half_h], dtype=np.float32),
                "position": np.array([cx, cy], dtype=np.float32),
                "velocity": np.array([0.0, 0.0], dtype=np.float32),
                "class_name": class_name,
                "track_id": track_id,
            })

    return dict(frames)


def load_video_labels(
    video_name: str,
    label_dir: Path,
    img_width: float,
    img_height: float,
    cache: Dict[str, Optional[Dict[int, List[dict]]]],
) -> Optional[Dict[int, List[dict]]]:
    """Load label data for a video (handles split files {video}.txt, {video}2.txt, ...)."""
    if video_name in cache:
        return cache[video_name]

    # Try base file and numbered suffixes
    all_frames: Dict[int, List[dict]] = {}
    for suffix in ["", "2", "3", "4", "5"]:
        path = label_dir / f"{video_name}{suffix}.txt"
        if not path.exists():
            if suffix == "":
                continue  # base file might not exist, try suffixes
            break  # no more split files

        parsed = parse_label_file(str(path), img_width, img_height)
        all_frames.update(parsed)

    if not all_frames:
        cache[video_name] = None
        return None

    cache[video_name] = all_frames
    return all_frames


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: dict, ckpt_path: str, device: str) -> TrafficPerceptionModel:
    """Load perception model and checkpoint, freeze everything, eval mode."""
    model = TrafficPerceptionModel(config, stage=2).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state") or ckpt.get("model") or ckpt

    loaded = 0
    skipped = 0
    for k, v in sd.items():
        if k.startswith("flow_chain."):
            skipped += 1
            continue
        try:
            target = model
            parts = k.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            param = getattr(target, parts[-1])
            if isinstance(param, (nn.Parameter, torch.Tensor)):
                if isinstance(param, nn.Parameter):
                    param.data.copy_(v)
                else:
                    setattr(target, parts[-1], v)
                loaded += 1
        except (AttributeError, RuntimeError):
            skipped += 1

    print(f"Perception checkpoint: {loaded} params loaded, {skipped} skipped")

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def get_perception_graph(model: TrafficPerceptionModel):
    if hasattr(model, "perception_graph"):
        return model.perception_graph
    return None


# ---------------------------------------------------------------------------
# Batched GAT extraction (disjoint-union)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_extract_gat_embeddings(
    model: TrafficPerceptionModel,
    frames: list,
    device: str,
) -> list:
    """
    Batch-process single-frame scenes through disjoint-union GAT.

    Each entry in `frames`:
        bboxes:     (N, 4)  tensor [x1,y1,x2,y2] pixels
        class_names: [str] * N
        positions:  (N, 2)  tensor center pixels
        velocities: (N, 2)  tensor

    Returns list of target_emb (D_gat,) tensors, target is always at index 0.
    """
    pg = get_perception_graph(model)
    if pg is None:
        se = model.simple_encoder
        results = []
        for fd in frames:
            node_feats = se(
                bboxes=fd["bboxes"].to(device),
                class_names=fd["class_names"],
                positions=fd["positions"].to(device),
                velocities=fd["velocities"].to(device),
                device=device,
            )
            if node_feats.shape[0] == 0:
                results.append(torch.zeros(model.node_feat_dim))
            else:
                results.append(node_feats[0].detach().cpu())
        return results

    # Collect all nodes
    all_node_feats = []
    all_bboxes = []
    all_positions = []
    all_velocities = []
    frame_offsets = []
    valid_indices = []
    cum_nodes = 0

    for fi, fd in enumerate(frames):
        b_t = fd["bboxes"].to(device)
        p_t = fd["positions"].to(device)
        v_t = fd["velocities"].to(device)
        cn_t = fd["class_names"]
        N = b_t.shape[0]

        if N == 0:
            frame_offsets.append(-1)
            continue

        node_feats = pg.node_encoder(
            bboxes=b_t, class_names=cn_t,
            positions=p_t, velocities=v_t, device=device,
        )
        all_node_feats.append(node_feats)
        all_bboxes.append(b_t)
        all_positions.append(p_t)
        all_velocities.append(v_t)
        frame_offsets.append(cum_nodes)
        valid_indices.append(fi)
        cum_nodes += N

    if cum_nodes == 0:
        return [torch.zeros(model.node_feat_dim) for _ in frames]

    big_node_feats = torch.cat(all_node_feats, dim=0).to(device)
    big_bboxes = torch.cat(all_bboxes, dim=0).to(device)
    big_positions = torch.cat(all_positions, dim=0).to(device)
    big_velocities = torch.cat(all_velocities, dim=0).to(device)

    # Build per-frame graphs
    big_edge_parts = []
    big_edge_type_parts = []
    for vi, fi in enumerate(valid_indices):
        fd = frames[fi]
        pos_np = fd["positions"].detach().cpu().numpy()
        ei, _nt, et = pg.graph_builder.build(
            positions=pos_np, class_names=fd["class_names"], target_idx=0,
        )
        if ei.numel() == 0:
            continue
        ei_offset = ei + frame_offsets[fi]
        big_edge_parts.append(ei_offset)
        big_edge_type_parts.append(et)

    if not big_edge_parts:
        results = []
        for fi in range(len(frames)):
            off = frame_offsets[fi]
            if off < 0:
                results.append(torch.zeros(model.node_feat_dim))
            else:
                results.append(big_node_feats[off].detach().cpu())
        return results

    big_edge_index = torch.cat(big_edge_parts, dim=1).to(device)
    big_edge_types = torch.cat(big_edge_type_parts, dim=0).to(device)
    big_src = big_edge_index[0]
    big_dst = big_edge_index[1]

    rel_spatial = pg.rel_spatial_encoder(
        bbox_src=big_bboxes[big_src], bbox_dst=big_bboxes[big_dst],
    )
    big_edge_weight = pg.edge_weight_calc(
        h_src=big_node_feats[big_src], h_dst=big_node_feats[big_dst],
        spatial_src=rel_spatial, spatial_dst=rel_spatial,
    )
    big_edge_attr = pg.edge_feat_encoder(
        pos_src=big_positions[big_src], pos_dst=big_positions[big_dst],
        edge_types=big_edge_types,
        vel_src=big_velocities[big_src],
        vel_dst=big_velocities[big_dst],
    )

    big_node_emb = pg.gat(
        x=big_node_feats, edge_index=big_edge_index,
        edge_weight=big_edge_weight, edge_attr=big_edge_attr,
    )

    # Extract target (index 0 per frame) + ped_gru
    results = []
    for fi in range(len(frames)):
        off = frame_offsets[fi]
        if off < 0:
            results.append(torch.zeros(model.node_feat_dim))
            continue
        t_emb = big_node_emb[off:off + 1]
        if pg.use_ped_gru:
            t2 = t_emb.unsqueeze(0)
            t2, _ = pg.ped_gru(t2)
            t_emb = t2.squeeze(0)
        results.append(t_emb.squeeze(0).detach().cpu())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute GAT target embeddings from label files")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--perception-ckpt", required=True,
                        help="Path to stage2_best.pt")
    parser.add_argument("--label-dir", default="labels",
                        help="Directory with per-video YOLO label .txt files")
    parser.add_argument("--data-dir", default="data/processed/trajectories")
    parser.add_argument("--domain-map", default="data/domains/domain_labels_int.json")
    parser.add_argument("--output", default="data/gat_conditions.pt")
    parser.add_argument("--frame-idx", type=int, default=4,
                        help="Which observation frame to use (0-7, default=4 middle)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap samples (0=all)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    img_width = config.get("video", {}).get("width", 3840.0)
    img_height = config.get("video", {}).get("height", 2160.0)

    # Domain map
    with open(args.domain_map) as f:
        domain_map = json.load(f)

    # Dataset (trajectory_only, no precomputed)
    print("Building dataset...")
    ds = TrajectoryDataset(
        data_dir=args.data_dir,
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="trajectory_only",
        domain_label_map=domain_map,
        max_samples=args.max_samples,
    )
    print(f"  {len(ds)} samples")

    # Load model
    print("\nLoading perception model...")
    model = load_model(config, args.perception_ckpt, device)
    D_gat = model.node_feat_dim
    print(f"  GAT output dim: {D_gat}")

    # Label cache
    label_dir = Path(args.label_dir)
    label_cache: Dict[str, Optional[Dict[int, List[dict]]]] = {}

    # Collect samples
    print("\nCollecting samples with scene data (matching by track_id)...")
    frame_idx = args.frame_idx
    to_process = []
    missing_label = 0
    missing_frame = 0
    no_target = 0

    for idx in range(len(ds)):
        sample = ds.samples[idx]
        video = sample["video"]
        obs_frames = sample["obs_frames"]
        track_id = sample["track_id"]

        if frame_idx >= len(obs_frames):
            no_target += 1
            continue

        fi = int(obs_frames[frame_idx])

        # Load label data for this video
        frames_data = load_video_labels(video, label_dir, img_width, img_height, label_cache)
        if frames_data is None:
            missing_label += 1
            continue

        # Get detections for this frame
        fd = frames_data.get(fi)
        if fd is None:
            missing_frame += 1
            continue

        # Find target by track_id
        target_idx = None
        for di, det in enumerate(fd):
            if det["track_id"] == track_id:
                target_idx = di
                break

        if target_idx is None:
            no_target += 1
            continue

        # Reorder: target first
        if target_idx != 0:
            fd = fd.copy()
            fd.insert(0, fd.pop(target_idx))

        N = len(fd)
        to_process.append({
            "video": video,
            "track_id": track_id,
            "obs_start": sample["obs_start"],
            "domain_id": sample["domain_id"],
            "bboxes": torch.as_tensor(np.stack([d["bbox"] for d in fd]), dtype=torch.float32),
            "class_names": [d["class_name"] for d in fd],
            "positions": torch.as_tensor(np.stack([d["position"] for d in fd]), dtype=torch.float32),
            "velocities": torch.as_tensor(np.stack([d["velocity"] for d in fd]), dtype=torch.float32),
        })

    total_attempted = len(ds) - missing_label
    match_pct = len(to_process) / max(total_attempted, 1) * 100
    print(f"  Valid samples:  {len(to_process)}")
    print(f"  Missing label:  {missing_label}")
    print(f"  Missing frame:  {missing_frame}")
    print(f"  No target (track_id not in frame): {no_target}")
    print(f"  Match rate:     {match_pct:.1f}%")

    if len(to_process) == 0:
        print("\nERROR: No valid samples. Check --label-dir path.")
        sys.exit(1)

    # Batch process
    print(f"\nExtracting GAT embeddings (batch_size={args.batch_size})...")
    all_results: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    batch_size = args.batch_size
    n_batches = (len(to_process) + batch_size - 1) // batch_size
    start_time = time.time()

    for bi in tqdm(range(n_batches), desc="Batches"):
        batch = to_process[bi * batch_size:(bi + 1) * batch_size]
        frames = [
            {"bboxes": item["bboxes"], "class_names": item["class_names"],
             "positions": item["positions"], "velocities": item["velocities"]}
            for item in batch
        ]
        embs = batch_extract_gat_embeddings(model, frames, device)

        for item, emb in zip(batch, embs):
            key = f"{item['track_id']}__{item['obs_start']}"
            all_results[item["video"]][key] = emb

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s ({len(to_process) / elapsed:.0f} samples/s)")

    # Statistics
    all_embs = torch.stack([e for vd in all_results.values() for e in vd.values()])
    print(f"\nEmbedding statistics:")
    print(f"  Count: {all_embs.shape[0]}")
    print(f"  Dim:   {all_embs.shape[1]}")
    print(f"  Mean:  {all_embs.mean().item():.6f}")
    print(f"  Std:   {all_embs.std().item():.6f}")
    print(f"  Min:   {all_embs.min().item():.6f}")
    print(f"  Max:   {all_embs.max().item():.6f}")
    print(f"  NaN:   {torch.isnan(all_embs).any().item()}")
    print(f"  All zero: {(all_embs.abs().sum() == 0).item()}")

    # Per-domain stats
    print(f"\nPer-domain breakdown:")
    domain_embs = defaultdict(list)
    for item in to_process:
        key = f"{item['track_id']}__{item['obs_start']}"
        emb = all_results[item["video"]].get(key)
        if emb is not None:
            domain_embs[item["domain_id"]].append(emb)

    for did in sorted(domain_embs.keys()):
        embs = torch.stack(domain_embs[did])
        print(f"  Domain {did}: {embs.shape[0]:5d} samples  "
              f"mean_norm={embs.mean(dim=0).norm().item():.4f}  "
              f"std={embs.std().item():.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        video: {k: v.clone() for k, v in vdict.items()}
        for video, vdict in all_results.items()
    }
    torch.save(save_dict, output_path)
    print(f"\nSaved to {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
