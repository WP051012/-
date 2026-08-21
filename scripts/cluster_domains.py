"""
域聚类 — 基于预计算的场景节点特征自动划分场景域

特征提取: 从每帧对象数据提取丰富统计量，捕获场景密度、运动模式、时序变化
聚类: DBSCAN + KMeans(自动选K)，两种方法互相验证

输出:
  - data/domains/domain_labels.json
  - data/domains/domain_stats.json
  - data/domains/video_features.npz
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import argparse

CLASS_NAMES = ["pedestrian", "bicycle", "car", "motorcycle", "bus", "traffic_light"]


# ======================================================================
# 特征提取
# ======================================================================

def extract_rich_features(data: dict) -> np.ndarray:
    """
    从预计算数据提取丰富的视频级特征。

    维度设计 (~120维):
      A. 基础统计 (per-frame mean/std) — 30维
      B. 对象数量分布 (percentiles + histogram) — 15维
      C. 速度分布 (percentiles per class) — 24维
      D. 空间分布 (per-zone occupancy) — 18维
      E. 时序变化 (segment-based stats) — 30维
      F. 类别分布熵 — 3维
    """
    CLASS_NAMES_LOCAL = CLASS_NAMES
    frame_objects = []   # per-frame object count
    frame_cls_counts = []  # per-frame class counts (6-dim)
    frame_positions = []   # per-frame mean positions
    frame_velocities = []  # per-frame mean velocities
    frame_areas = []       # per-frame mean bbox areas
    per_class_vels = defaultdict(list)  # per-class velocities

    for frame_id, frame_data in sorted(data.items(), key=lambda x: int(x[0])):
        names = list(frame_data["class_names"])
        positions = frame_data["positions"]
        velocities = frame_data["velocities"]
        bboxes = frame_data["bboxes"]

        N = len(names)
        frame_objects.append(N)

        if N == 0:
            frame_cls_counts.append(np.zeros(6, dtype=np.float32))
            frame_positions.append(np.zeros(2, dtype=np.float32))
            frame_velocities.append(np.zeros(2, dtype=np.float32))
            frame_areas.append(0.0)
            continue

        # Per-frame class counts
        cls_counts = np.zeros(6, dtype=np.float32)
        for name in names:
            if name in CLASS_NAMES_LOCAL:
                cls_counts[CLASS_NAMES_LOCAL.index(name)] += 1
        frame_cls_counts.append(cls_counts)

        # Per-frame position/velocity/area
        frame_positions.append(positions.mean(axis=0))
        frame_velocities.append(np.abs(velocities).mean(axis=0))
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        frame_areas.append(areas.mean())

        # Per-class velocities
        for i, name in enumerate(names):
            if name in CLASS_NAMES_LOCAL:
                speed = np.linalg.norm(velocities[i])
                per_class_vels[name].append(speed)

    # ---- A. 基础统计 (per-frame mean/std) ----
    fv_arr = np.column_stack([
        np.array(frame_cls_counts),       # (T, 6)
        np.array(frame_positions),        # (T, 2)
        np.array(frame_velocities),       # (T, 2)
        np.array(frame_areas)[:, None],   # (T, 1)
    ])  # (T, 11)

    A_mean = fv_arr.mean(axis=0)   # (11,)
    A_std = fv_arr.std(axis=0)     # (11,)

    # ---- B. 对象数量分布 ----
    obj_arr = np.array(frame_objects, dtype=np.float32)
    B_percentiles = np.percentile(obj_arr, [10, 25, 50, 75, 90, 95])  # (6,)
    B_hist, _ = np.histogram(obj_arr, bins=5, range=(0, max(obj_arr.max(), 1)))  # (5,)
    B_hist = B_hist.astype(np.float32) / max(len(frame_objects), 1)
    B_max = np.array([obj_arr.max()])  # (1,)

    # ---- C. 速度分布 (per class) ----
    C_features = []
    for cls_name in CLASS_NAMES_LOCAL:
        vels = per_class_vels.get(cls_name, [0.0])
        v_arr = np.array(vels)
        C_features.extend([v_arr.mean(), v_arr.std(), np.percentile(v_arr, 50), np.percentile(v_arr, 90)])
    C_features = np.array(C_features)  # (24,)

    # ---- D. 空间分布 ----
    # Per-class position centroids (mean position per class over all frames)
    D_class_positions = []
    for cls_idx in range(6):
        # Find frames where this class appears and get positions
        cls_pos = []
        for fi, fcs in enumerate(frame_cls_counts):
            if fcs[cls_idx] > 0 and fi < len(frame_positions):
                cls_pos.append(frame_positions[fi])
        if cls_pos:
            D_class_positions.extend(np.array(cls_pos).mean(axis=0))
        else:
            D_class_positions.extend([0.0, 0.0])
    D_features = np.array(D_class_positions)  # (12,)

    # ---- E. 时序变化 (segment-based) ----
    T = len(frame_objects)
    n_segments = 3
    seg_len = max(1, T // n_segments)
    E_features = []
    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        end = min(start + seg_len, T)
        if start < end:
            seg_cls = np.array(frame_cls_counts[start:end])
            seg_obj = obj_arr[start:end]
            E_features.extend([
                seg_obj.mean(), seg_obj.std(),
                seg_cls.mean(axis=0).sum() if seg_cls.size > 0 else 0,  # total objects
            ])
    while len(E_features) < n_segments * 3:
        E_features.append(0.0)
    E_features = np.array(E_features[:n_segments * 3])  # (9,)
    # Add diff between segments (captures trend)
    seg_means = [obj_arr[i*seg_len:min((i+1)*seg_len, T)].mean() for i in range(n_segments)]
    for i in range(len(seg_means) - 1):
        E_features = np.append(E_features, seg_means[i+1] - seg_means[i])
    while len(E_features) < n_segments * 3 + (n_segments - 1):
        E_features = np.append(E_features, 0.0)

    # ---- F. 类别分布熵 ----
    total_cls = np.array(frame_cls_counts).sum(axis=0)  # (6,)
    total_sum = total_cls.sum()
    if total_sum > 0:
        cls_prob = total_cls / total_sum
        cls_prob = cls_prob[cls_prob > 0]
        F_entropy = -np.sum(cls_prob * np.log(cls_prob + 1e-8))  # scalar
        F_dominance = total_cls.max() / max(total_sum, 1)         # scalar
        F_vehicle_ratio = total_cls[2:6].sum() / max(total_sum, 1)  # vehicles / total
    else:
        F_entropy = 0.0; F_dominance = 0.0; F_vehicle_ratio = 0.0
    F_features = np.array([F_entropy, F_dominance, F_vehicle_ratio])

    # ---- 拼接 ----
    feature = np.concatenate([
        A_mean, A_std,           # 22
        B_percentiles, B_hist.ravel(), B_max,  # 6+5+1=12
        C_features,              # 24
        D_features,              # 12
        E_features,              # ~12
        F_features,              # 3
    ]).astype(np.float32)

    return feature


def extract_all_features(precomputed_dir: Path) -> dict:
    features = {}
    npz_files = sorted(precomputed_dir.glob("*.npz"))
    print(f"扫描 {len(npz_files)} 个预计算文件...")

    for i, npz_path in enumerate(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)["data"].item()
            fv = extract_rich_features(data)
            features[npz_path.stem] = fv
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(npz_files)}...")

    print(f"提取了 {len(features)} 个视频特征 (dim={next(iter(features.values())).shape[0] if features else 0})")
    return features


# ======================================================================
# 聚类
# ======================================================================

def assign_site(video_name: str) -> str:
    import re
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


def cluster_and_evaluate(feature_matrix: np.ndarray, method: str, eps: float, min_samples: int):
    """尝试多种参数，选最优聚类"""
    results = {}

    if method in ("dbscan", "both"):
        best_labels = None; best_n = 0; best_score = -1
        for eps_val in [eps, eps * 0.7, eps * 0.5, eps * 0.3]:
            db = DBSCAN(eps=eps_val, min_samples=min_samples, metric="cosine")
            labels = db.fit_predict(feature_matrix)
            n_domains = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_domains >= 2:
                mask = labels != -1
                if mask.sum() > n_domains * 2:
                    try:
                        sil = silhouette_score(feature_matrix[mask], labels[mask])
                    except:
                        sil = -1
                else:
                    sil = -1
                print(f"  DBSCAN eps={eps_val:.3f}: {n_domains} domains, {n_noise} noise, sil={sil:.3f}")
                if sil > best_score:
                    best_score = sil; best_labels = labels; best_n = n_domains

        if best_labels is not None:
            results["dbscan"] = {"labels": best_labels, "n_domains": best_n}
        else:
            # fallback: use original eps
            labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(feature_matrix)
            n_domains = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"  DBSCAN fallback eps={eps}: {n_domains} domains")
            results["dbscan"] = {"labels": labels, "n_domains": n_domains}

    if method in ("kmeans", "both"):
        n = feature_matrix.shape[0]
        max_k = min(12, n - 1)
        best_k, best_score, best_labels = 2, -1, None
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feature_matrix)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(feature_matrix, labels)
            db_score = davies_bouldin_score(feature_matrix, labels)
            print(f"  KMeans K={k}: sil={sil:.4f}, DB={db_score:.4f}")
            if sil > best_score:
                best_score = sil; best_k = k; best_labels = labels

        print(f"  最优 K={best_k} (silhouette={best_score:.4f})")
        results["kmeans"] = {"labels": best_labels, "n_domains": best_k}

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="场景域聚类")
    parser.add_argument("--precomputed-dir", type=str,
                        default="/root/red-light-prediction/data/precomputed")
    parser.add_argument("--output-dir", type=str,
                        default="/root/red-light-prediction/data/domains")
    parser.add_argument("--method", type=str, default="both",
                        choices=["dbscan", "kmeans", "both"])
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=50,
                        help="PCA降维 (0=不降维)")
    args = parser.parse_args()

    precomputed_dir = Path(args.precomputed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取特征
    print("=" * 60)
    print("Step 1: 提取视频级特征")
    print("=" * 60)
    features = extract_all_features(precomputed_dir)
    if not features:
        print("错误: 没有找到有效的预计算文件")
        sys.exit(1)

    video_names = sorted(features.keys())
    feature_matrix = np.stack([features[v] for v in video_names])
    print(f"原始特征维度: {feature_matrix.shape}")

    # 标准化
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # PCA
    if args.pca_dim > 0:
        pca_dim = min(args.pca_dim, feature_matrix_scaled.shape[0], feature_matrix_scaled.shape[1])
        pca = PCA(n_components=pca_dim)
        feature_matrix_use = pca.fit_transform(feature_matrix_scaled)
        print(f"PCA: {feature_matrix_scaled.shape[1]} → {pca_dim} (explained var: {pca.explained_variance_ratio_.sum():.3f})")
    else:
        feature_matrix_use = feature_matrix_scaled

    # 2. 聚类
    print("\n" + "=" * 60)
    print(f"Step 2: 聚类")
    print("=" * 60)
    results = cluster_and_evaluate(feature_matrix_use, args.method, args.eps, args.min_samples)

    # 3. 保存
    for method_name, result in results.items():
        labels = result["labels"]
        n_domains = result["n_domains"]

        domain_labels = {video_names[i]: int(labels[i]) for i in range(len(video_names))}
        with open(output_dir / f"domain_labels_{method_name}.json", "w") as f:
            json.dump(domain_labels, f, indent=2, ensure_ascii=False)

        # 统计
        domain_groups = defaultdict(list)
        for vname, label in domain_labels.items():
            domain_groups[label].append({"name": vname, "site": assign_site(vname)})

        stats = {}
        print(f"\n--- {method_name}: {n_domains} domains ---")
        for label, videos in sorted(domain_groups.items()):
            sites = defaultdict(int)
            for v in videos:
                sites[v["site"]] += 1
            stats[str(label)] = {"count": len(videos), "sites": dict(sites)}
            status = "噪声" if label == -1 else f"域{label}"
            site_info = " | ".join(f"{s}:{c}" for s, c in sites.items())
            print(f"  {status}: {len(videos)} 视频 [{site_info}]")

        with open(output_dir / f"domain_stats_{method_name}.json", "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # 4. 保存特征
    best_method = "kmeans" if "kmeans" in results else "dbscan"
    np.savez_compressed(
        output_dir / "video_features.npz",
        features=feature_matrix,
        video_names=np.array(video_names),
        domain_labels=np.array([results[best_method]["labels"][i] for i, _ in enumerate(video_names)]),
    )

    print(f"\n完成! 输出: {output_dir}")


if __name__ == "__main__":
    main()
