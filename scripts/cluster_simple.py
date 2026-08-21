"""
简单聚类 — 基于节点数量与空间分布
只从预计算数据里提取: 每帧物体数、类型分布、空间离散度
"""
import numpy as np, json, sys, re
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse

PRECOMPUTED = Path("/root/red-light-prediction/data/precomputed")
OUT_DIR = Path("/root/red-light-prediction/data/domains")
IMG_W, IMG_H = 3840, 2160
CLASS_NAMES = ["pedestrian", "bicycle", "car", "motorcycle", "bus", "traffic_light"]


def extract_simple_features(data: dict) -> np.ndarray:
    """每帧只提取: 物体数 + 类型占比 + 位置离散度 → video-level"""
    frame_feats = []
    for frame_id, frame_data in sorted(data.items(), key=lambda x: int(x[0])):
        positions = frame_data["positions"]
        names = list(frame_data["class_names"])
        N = len(names)

        if N == 0:
            frame_feats.append([0.0] * 10)
            continue

        # 类型占比
        cls_counts = np.zeros(6, dtype=np.float32)
        for name in names:
            if name in CLASS_NAMES:
                cls_counts[CLASS_NAMES.index(name)] += 1
        cls_ratio = cls_counts / N  # 6

        # 位置离散度
        pos_std = positions.std(axis=0) if N > 1 else np.zeros(2)  # 2

        # 物体数 (log scale to compress range)
        fv = np.concatenate([
            [np.log1p(N)],     # 1
            cls_ratio,          # 6
            pos_std,            # 2
            [N],                # 1 (raw count for averaging)
        ])
        frame_feats.append(fv)

    fv = np.array(frame_feats)  # (T, 10)

    # Video-level: mean + std of per-frame
    feat_mean = fv[:, :9].mean(axis=0)  # 9
    feat_std = fv[:, :9].std(axis=0)    # 9
    avg_count = fv[:, 9].mean()          # raw average count

    return np.concatenate([feat_mean, feat_std, [avg_count]]).astype(np.float32)  # 19


def assign_site(video_name: str) -> str:
    if "timing" in video_name:
        return "site_A"
    m = re.search(r"_(\d{8})\d{6}_", video_name)
    if not m:
        m = re.search(r"_(\d{8})_", video_name)
    if m:
        d = m.group(1)
        if d in ("20260115","20260121","20260122"): return "site_A"
        elif d in ("20260123","20260126","20260127"): return "site_B"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=0, help="0=auto, >0=manual K per site")
    args = parser.parse_args()

    # Extract
    print("Extracting features...")
    features_dict = {}
    for npz_path in sorted(PRECOMPUTED.glob("*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)["data"].item()
            features_dict[npz_path.stem] = extract_simple_features(data)
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")

    names = sorted(features_dict.keys())
    features = np.stack([features_dict[n] for n in names])
    print(f"{len(names)} videos, {features.shape[1]} dims")

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(features)

    # Per-site clustering
    mask_a = np.array([assign_site(n) == "site_A" for n in names])
    mask_b = np.array([assign_site(n) == "site_B" for n in names])
    print(f"Site A: {mask_a.sum()}, Site B: {mask_b.sum()}")

    all_labels = {}
    domain_id_offset = 0

    for site_name, mask in [("site_A", mask_a), ("site_B", mask_b)]:
        feats = feats_scaled[mask]
        site_names = np.array(names)[mask]

        if args.k > 0:
            best_k = args.k
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            best_labels = km.fit_predict(feats)
        else:
            best_k, best_sil, best_labels = 2, -1, None
            for k in range(2, min(6, len(feats))):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(feats)
                sil = silhouette_score(feats, labels)
                counts = np.bincount(labels)
                print(f"  {site_name} K={k}: sil={sil:.4f}, counts={counts.tolist()}")
                if sil > best_sil:
                    best_sil, best_k, best_labels = sil, k, labels

        print(f"  => {site_name} K={best_k} (sil={best_sil:.4f})")

        for i in range(best_k):
            count = (best_labels == i).sum()
            cluster_feats = features[mask][best_labels == i]
            avg_obj = cluster_feats[:, -1].mean()  # raw avg count
            # Type distribution
            avg_types = cluster_feats[:, 1:7].mean(axis=0)  # pedestrian..traffic_light
            type_str = ", ".join(f"{CLASS_NAMES[j]}={avg_types[j]:.1%}" for j in range(6) if avg_types[j] > 0.05)
            print(f"    D{i}: {count} videos, avg_objects={avg_obj:.1f}, types=[{type_str}]")

        # Map
        for i, name in enumerate(site_names):
            all_labels[str(name)] = int(best_labels[i]) + domain_id_offset
        domain_id_offset += best_k

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "domain_labels_int.json", "w") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    n_domains = len(set(all_labels.values()))
    print(f"\nTotal domains: {n_domains}")


if __name__ == "__main__":
    main()
