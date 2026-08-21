"""
基于 ourmethod 认知图结构的域划分

使用与 TrafficPerceptionGraph 完全一致的表征:

节点特征 (GAT前原始特征):
  - spatial: [x1', y1', x2', y2', w', h', area', aspect] — 8维
  - motion:  [vx, vy, speed, angle, 0, 0] — 6维
  → 14维/节点 (pedestrian + vehicle), 4维(traffic_light)

边特征 (6种类型, 与 PerceptionGraphBuilder 一致):
  0: Core↔Vehicle     → rel_pos, rel_dir, speed_diff
  1: Core↔Person      → rel_pos, rel_dir
  2: Core↔Infra       → rel_pos, rel_dir
  3: Vehicle↔Vehicle  → rel_pos, rel_dir, speed_diff (dist<threshold)
  4: Infra↔Infra      → 全连接
  5: Person↔Person    → rel_pos, rel_dir (dist<threshold)

图级聚合 → 视频级 → KMeans (per site)
"""

import numpy as np
import json, sys, re
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
import argparse

PRECOMPUTED = Path("/root/red-light-prediction/data/precomputed")
OUT_DIR = Path("/root/red-light-prediction/data/domains")
IMG_W, IMG_H = 3840, 2160

CLASS_NAMES = ["pedestrian", "bicycle", "car", "motorcycle", "bus", "traffic_light"]

# Node type constants (matching PerceptionGraphBuilder)
CORE, VEHICLE, PERSON, INFRA = 0, 1, 2, 3


# ======================================================================
# Node features (matching NodeFeatureEncoder raw features)
# ======================================================================

def compute_node_features(bboxes, positions, velocities, class_ids):
    """
    计算 GAT 前的原始节点特征 (与 NodeFeatureEncoder._build_raw_features 一致).

    Returns: (N, 14) for agent nodes, 4-dim for traffic_light
             We pad all to 14-dim for uniformity, with unused dims = 0
    """
    N = bboxes.shape[0]
    feats = np.zeros((N, 14), dtype=np.float32)

    if N == 0:
        return feats

    # Spatial encoding (8-dim): [x1', y1', x2', y2', w', h', area', aspect]
    x1 = bboxes[:, 0] / IMG_W
    y1 = bboxes[:, 1] / IMG_H
    x2 = bboxes[:, 2] / IMG_W
    y2 = bboxes[:, 3] / IMG_H
    w = x2 - x1
    h = y2 - y1
    area = w * h
    aspect = w / (h + 1e-6)

    spatial = np.stack([x1, y1, x2, y2, w, h, area, aspect], axis=-1)  # (N, 8)
    feats[:, :8] = spatial

    # Motion encoding (6-dim): [vx, vy, speed, angle, 0, 0]
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    speed = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)
    feats[:, 8] = vx
    feats[:, 9] = vy
    feats[:, 10] = speed
    feats[:, 11] = angle
    feats[:, 12] = 0.0  # acc_x
    feats[:, 13] = 0.0  # acc_y

    return feats


# ======================================================================
# Graph construction (matching PerceptionGraphBuilder.build)
# ======================================================================

def classify_node(class_name):
    """Match PerceptionGraphBuilder._get_node_type"""
    if class_name == "pedestrian":
        return PERSON  # target is also PERSON here (we don't have target_idx)
    elif class_name in ("bicycle", "motorcycle", "car", "bus", "truck"):
        return VEHICLE
    elif class_name in ("traffic_light", "traffic_sign", "lane_line"):
        return INFRA
    return INFRA


def build_graph(positions, class_names, max_distance=30.0):
    """
    构建异构交通感知图 (匹配 PerceptionGraphBuilder.build, 无target).

    Edge types:
      3: Vehicle↔Vehicle (dist < threshold)
      4: Infra↔Infra (full)
      5: Person↔Person (dist < threshold)

    Note: 由于没有target_idx, 这里省略 Core↔* 边(type 0/1/2).
          这些边只在有target时才存在.
          但这不影响域划分 — 场景结构由所有agent之间的边决定.

    Returns:
      edge_index: (2, E) int
      edge_types: (E,) int
      node_types: [N] int
    """
    N = len(class_names)
    node_types = np.array([classify_node(cn) for cn in class_names], dtype=np.int32)

    src, dst, etype = [], [], []

    # --- Type 3: Vehicle↔Vehicle ---
    veh_idx = np.where(node_types == VEHICLE)[0]
    for i in range(len(veh_idx)):
        for j in range(i + 1, len(veh_idx)):
            u, v = veh_idx[i], veh_idx[j]
            dist = np.linalg.norm(positions[u] - positions[v])
            if dist < max_distance:
                src.extend([u, v]); dst.extend([v, u]); etype.extend([3, 3])

    # --- Type 4: Infra↔Infra ---
    infra_idx = np.where(node_types == INFRA)[0]
    for i in range(len(infra_idx)):
        for j in range(i + 1, len(infra_idx)):
            u, v = infra_idx[i], infra_idx[j]
            src.extend([u, v]); dst.extend([v, u]); etype.extend([4, 4])

    # --- Type 5: Person↔Person ---
    person_idx = np.where(node_types == PERSON)[0]
    for i in range(len(person_idx)):
        for j in range(i + 1, len(person_idx)):
            u, v = person_idx[i], person_idx[j]
            dist = np.linalg.norm(positions[u] - positions[v])
            if dist < max_distance:
                src.extend([u, v]); dst.extend([v, u]); etype.extend([5, 5])

    if not src:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty(0, dtype=np.int32),
            node_types,
        )

    return (
        np.array([src, dst], dtype=np.int64),
        np.array(etype, dtype=np.int32),
        node_types,
    )


# ======================================================================
# Edge features (matching EdgeFeatureEncoder)
# ======================================================================

def compute_edge_features(positions, velocities, edge_index, edge_types, node_types):
    """
    计算边特征 (匹配 EdgeFeatureEncoder.forward).

    每条边:
      [dx, dy, dist, rel_dir_x, rel_dir_y, speed_diff, edge_type_onehot(3)]
      → 9维
    """
    if edge_index.shape[1] == 0:
        return np.empty((0, 9), dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    E = len(src)

    # Relative position
    delta = positions[dst] - positions[src]  # (E, 2)
    dist = np.linalg.norm(delta, axis=1)     # (E,)

    # Relative direction
    rel_dir = delta / (dist[:, None] + 1e-6)  # (E, 2)

    # Speed diff
    speed_src = np.linalg.norm(velocities[src], axis=1)
    speed_dst = np.linalg.norm(velocities[dst], axis=1)
    speed_diff = (speed_dst - speed_src) / (speed_src + 1e-6)

    # Edge type encoding (3-dim: vehicle-vehicle, infra-infra, person-person)
    etype_enc = np.zeros((E, 3), dtype=np.float32)
    for i, et in enumerate([3, 4, 5]):
        etype_enc[edge_types == et, i] = 1.0

    edge_feats = np.column_stack([
        delta,                          # 2
        dist[:, None],                  # 1
        rel_dir,                        # 2
        speed_diff[:, None],            # 1
        etype_enc,                      # 3
    ])  # (E, 9)

    return edge_feats.astype(np.float32)


# ======================================================================
# Per-frame graph-level features
# ======================================================================

def compute_graph_level_features(node_feats, edge_feats, edge_index, edge_types, node_types, positions):
    """
    从节点和边特征计算图级统计量 (~50维).
    """
    N = node_feats.shape[0]
    E = edge_feats.shape[0] if edge_feats.size > 0 else 0
    features = []

    # ---- 1. 节点统计 ----
    # 1a. 数量
    features.append(float(N))

    # 1b. 类型分布
    type_counts = np.bincount(node_types, minlength=4)[:4].astype(np.float32)  # 4
    features.extend((type_counts / max(N, 1)).tolist())

    # 1c. 位置离散度 (spatial spread)
    if N >= 2:
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0)
        features.extend(pos_mean.tolist())  # 2
        features.extend(pos_std.tolist())   # 2

        try:
            hull = ConvexHull(positions)
            features.append(hull.volume)
        except:
            features.append(0.0)
    else:
        features.extend([0.0] * 5)

    # 1d. 速度分布
    speeds = np.linalg.norm(node_feats[:, 8:10], axis=1)  # vx, vy from motion encoding
    features.append(speeds.mean())
    features.append(speeds.std() if N > 1 else 0.0)
    features.append(np.percentile(speeds, 90))
    features.append(np.percentile(speeds, 50))

    # ---- 2. 边统计 ----
    # 2a. 总边数
    features.append(float(E))

    # 2b. 各类型边数
    for et in [3, 4, 5]:
        count = (edge_types == et).sum()
        features.append(float(count))

    # 2c. 图密度
    max_edges_possible = N * (N - 1) / 2.0
    features.append(E / max(max_edges_possible, 1))

    # 2d. 边距离分布 (per type)
    for et in [3, 4, 5]:
        mask = edge_types == et
        if mask.sum() > 0:
            et_dists = edge_feats[mask, 2]  # dist is column 2
            features.append(et_dists.mean())
            features.append(et_dists.std() if mask.sum() > 1 else 0.0)
        else:
            features.extend([0.0, 0.0])

    # 2e. 边方向分布 (per type)
    for et in [3, 4, 5]:
        mask = edge_types == et
        if mask.sum() > 0:
            et_dir = edge_feats[mask, 3:5]  # rel_dir columns
            features.append(et_dir[:, 0].std())  # dir_x spread
            features.append(et_dir[:, 1].std())  # dir_y spread
        else:
            features.extend([0.0, 0.0])

    # 2f. 速度差分布
    if E > 0:
        features.append(edge_feats[:, 5].mean())  # speed_diff mean
        features.append(edge_feats[:, 5].std() if E > 1 else 0.0)
    else:
        features.extend([0.0, 0.0])

    # ---- 3. 度分布 ----
    if E > 0 and N > 0:
        deg = np.bincount(np.concatenate([edge_index[0], edge_index[1]]), minlength=N)
        features.append(deg.mean())
        features.append(deg.std() if N > 1 else 0.0)
        features.append(float(deg.max()))
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


# ======================================================================
# Video-level feature extraction
# ======================================================================

def extract_video_features(data: dict) -> np.ndarray:
    """Aggregate per-frame graph features → video-level vector."""

    per_frame_feats = []

    for frame_id, frame_data in sorted(data.items(), key=lambda x: int(x[0])):
        positions = frame_data["positions"]
        velocities = frame_data["velocities"]
        bboxes = frame_data["bboxes"]
        class_names = list(frame_data["class_names"])
        class_ids = frame_data["class_ids"]

        N = positions.shape[0]

        # Node features
        node_feats = compute_node_features(bboxes, positions, velocities, class_ids)

        # Build graph
        ei, et, nt = build_graph(positions, class_names, max_distance=30.0)

        # Edge features
        ef = compute_edge_features(positions, velocities, ei, et, nt)

        # Graph-level summary
        gfeat = compute_graph_level_features(node_feats, ef, ei, et, nt, positions)
        per_frame_feats.append(gfeat)

    if not per_frame_feats:
        return np.zeros(120, dtype=np.float32)

    fv = np.array(per_frame_feats)  # (T, D)

    # Aggregate: mean + std
    feat_mean = fv.mean(axis=0)
    feat_std = fv.std(axis=0)

    # Key dimension percentiles
    key_dims = [0, 1, 9, 10, 11, 12, 13]  # N, type counts, E, density, etc.
    percentiles = []
    for idx in key_dims:
        if idx < fv.shape[1]:
            percentiles.extend(np.percentile(fv[:, idx], [10, 50, 90]).tolist())

    return np.concatenate([feat_mean, feat_std, np.array(percentiles)]).astype(np.float32)


# ======================================================================
# Site detection
# ======================================================================

def assign_site(video_name: str) -> str:
    if "timing" in video_name:
        return "site_A"
    m = re.search(r"_(\d{8})\d{6}_", video_name)
    if not m:
        m = re.search(r"_(\d{8})_", video_name)
    if m:
        d = m.group(1)
        if d in ("20260115", "20260121", "20260122"):
            return "site_A"
        elif d in ("20260123", "20260126", "20260127"):
            return "site_B"
    return "unknown"


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-dir", type=str, default=str(PRECOMPUTED))
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()

    precomputed_dir = Path(args.precomputed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    print("=" * 60)
    print("Step 1: 构建认知图并提取图表征")
    print("=" * 60)

    features_dict = {}
    npz_files = sorted(precomputed_dir.glob("*.npz"))
    for i, npz_path in enumerate(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)["data"].item()
            fv = extract_video_features(data)
            features_dict[npz_path.stem] = fv
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(npz_files)}...")

    names = sorted(features_dict.keys())
    features = np.stack([features_dict[n] for n in names])
    print(f"{len(names)} videos, {features.shape[1]} dims")

    # Normalize
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(features)

    # Per-site clustering
    print("\n" + "=" * 60)
    print("Step 2: 按机位分别聚类")
    print("=" * 60)

    mask_a = np.array([assign_site(n) == "site_A" for n in names])
    mask_b = np.array([assign_site(n) == "site_B" for n in names])
    print(f"Site A: {mask_a.sum()}, Site B: {mask_b.sum()}")

    all_labels = {}
    domain_stats = {}
    domain_id_counter = 0

    for site_name, mask in [("site_A", mask_a), ("site_B", mask_b)]:
        if mask.sum() == 0:
            continue

        feats = feats_scaled[mask]
        site_names = np.array(names)[mask]

        best_k, best_sil, best_labels = 2, -1, None
        max_k = min(args.max_k, len(feats) - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats)
            if len(set(labels)) >= 2:
                sil = silhouette_score(feats, labels)
                counts = np.bincount(labels)
                print(f"  {site_name} K={k}: sil={sil:.4f}, counts={counts.tolist()}")
                if sil > best_sil:
                    best_sil, best_k, best_labels = sil, k, labels

        print(f"  => {site_name} best K={best_k} (sil={best_sil:.4f})")

        for i, name in enumerate(site_names):
            local_label = int(best_labels[i])
            all_labels[str(name)] = {
                "site": site_name,
                "domain": f"{site_name}_d{local_label}",
                "domain_id": domain_id_counter + local_label,
            }

        for i in range(best_k):
            count = (best_labels == i).sum()
            cluster_feats = features[mask][best_labels == i]
            avg_objects = cluster_feats[:, 0].mean() if cluster_feats.size > 0 else 0
            domain_stats[f"{site_name}_d{i}"] = {
                "site": site_name,
                "count": int(count),
                "avg_objects_per_frame": float(avg_objects),
            }

        domain_id_counter += best_k

    # Save
    simple_labels = {k: v["domain_id"] for k, v in all_labels.items()}
    with open(output_dir / "domain_labels.json", "w") as f:
        json.dump(simple_labels, f, indent=2, ensure_ascii=False)
    with open(output_dir / "domain_labels_full.json", "w") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    with open(output_dir / "domain_stats.json", "w") as f:
        json.dump(domain_stats, f, indent=2, ensure_ascii=False)

    # Summary
    unique = sorted(set(v["domain"] for v in all_labels.values()))
    print(f"\n=== {len(unique)} domains ===")
    for d in unique:
        count = sum(1 for v in all_labels.values() if v["domain"] == d)
        info = domain_stats.get(d, {})
        avg_obj = info.get("avg_objects_per_frame", 0)
        print(f"  {d}: {count} videos, avg_objects/frame={avg_obj:.1f}")

    # Integer ID mapping
    domain_to_id = {d: i for i, d in enumerate(unique)}
    int_labels = {name: domain_to_id[all_labels[name]["domain"]] for name in names}
    with open(output_dir / "domain_labels_int.json", "w") as f:
        json.dump(int_labels, f, indent=2)

    print(f"\nDomain ID mapping: {domain_to_id}")


if __name__ == "__main__":
    main()
