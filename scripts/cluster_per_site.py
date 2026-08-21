"""Per-site domain clustering — discover sub-domains within each camera position."""
import numpy as np, json, re, sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Re-extract features (same as cluster_domains.py)
sys.path.insert(0, "/root/red-light-prediction")
from scripts.cluster_domains import extract_rich_features, assign_site

PRECOMPUTED = Path("/root/red-light-prediction/data/precomputed")
OUT_DIR = Path("/root/red-light-prediction/data/domains")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load features
print("Extracting features...")
features_dict = {}
for npz_path in sorted(PRECOMPUTED.glob("*.npz")):
    try:
        data = np.load(npz_path, allow_pickle=True)["data"].item()
        fv = extract_rich_features(data)
        features_dict[npz_path.stem] = fv
    except Exception as e:
        print(f"  SKIP {npz_path.name}: {e}")

names = sorted(features_dict.keys())
features = np.stack([features_dict[n] for n in names])
print(f"{len(names)} videos, {features.shape[1]} dims")

# Split by site
mask_a = np.array([assign_site(n) == "site_A" for n in names])
mask_b = np.array([assign_site(n) == "site_B" for n in names])
print(f"Site A: {mask_a.sum()}, Site B: {mask_b.sum()}")

all_labels = {}
domain_id = 0
domain_stats = {}

for site_name, mask in [("site_A", mask_a), ("site_B", mask_b)]:
    if mask.sum() == 0:
        continue

    feats = features[mask]
    site_names = np.array(names)[mask]

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # Try K=2..10
    best_k, best_sil, best_labels = 2, -1, None
    max_k = min(10, len(feats) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feats_scaled)
        if len(set(labels)) >= 2:
            sil = silhouette_score(feats_scaled, labels)
            counts = np.bincount(labels)
            print(f"  {site_name} K={k}: sil={sil:.4f}, counts={counts.tolist()}")
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, labels

    print(f"  => {site_name} best K={best_k} (sil={best_sil:.4f})")

    # Map labels to global domain IDs
    for i, name in enumerate(site_names):
        local_label = int(best_labels[i])
        global_label = f"{site_name}_d{local_label}"
        all_labels[str(name)] = global_label

    # Stats
    for i in range(best_k):
        count = (best_labels == i).sum()
        domain_stats[f"{site_name}_d{i}"] = {"site": site_name, "count": int(count)}

# Save
with open(OUT_DIR / "domain_labels.json", "w") as f:
    json.dump(all_labels, f, indent=2, ensure_ascii=False)
with open(OUT_DIR / "domain_stats.json", "w") as f:
    json.dump(domain_stats, f, indent=2, ensure_ascii=False)

# Count domains
unique_domains = set(all_labels.values())
print(f"\nTotal domains: {len(unique_domains)}")
for d in sorted(unique_domains):
    count = sum(1 for v in all_labels.values() if v == d)
    print(f"  {d}: {count} videos")

# Save with integer IDs for training
domain_to_id = {d: i for i, d in enumerate(sorted(unique_domains))}
id_labels = {name: domain_to_id[d] for name, d in all_labels.items()}
with open(OUT_DIR / "domain_labels_int.json", "w") as f:
    json.dump(id_labels, f, indent=2)

print(f"Domain ID mapping: {domain_to_id}")
