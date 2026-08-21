"""
Minimal diagnostic: dump raw data for 3 samples to find the root cause.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, torch, numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from data.dataset import TrajectoryDataset, trajectory_collate_fn
from scripts.run_experiments import load_split_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM = torch.tensor([3840.0, 2160.0])

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

# Get junction/crossing ROI
junction_roi, crosswalk_roi = None, None
for key in ("intersection_A", "intersection_B"):
    c = config.get(key, {})
    jr = c.get("junction_roi")
    if jr and len(jr) >= 3:
        junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
        break

print(f"Junction ROI: {junction_roi}")

# Load val set
_, _, _, _, val_scene, _ = load_split_datasets(
    "data/processed/trajectories", label_dir="labels/", quick=False,
)

loader = DataLoader(val_scene, batch_size=1, shuffle=True,
                    collate_fn=trajectory_collate_fn)

for i, batch in enumerate(loader):
    if i >= 3:
        break

    print(f"\n{'='*60}")
    print(f"Sample {i}:")
    obs = batch["obs_trajectory"][0]  # (8, 2)
    target = batch["target_trajectory"][0]  # (12, 2)
    label = batch["is_violation"].item()
    video = batch["video"][0]
    tl_states = batch.get("traffic_light_states", "NOT IN BATCH")

    print(f"  Video: {video}, Label: {label}")
    print(f"  Obs first pos (px): ({obs[0,0].item():.0f}, {obs[0,1].item():.0f})")
    print(f"  Obs last  pos (px): ({obs[-1,0].item():.0f}, {obs[-1,1].item():.0f})")
    print(f"  Target last pos (px): ({target[-1,0].item():.0f}, {target[-1,1].item():.0f})")
    print(f"  Traffic light states IN BATCH: {tl_states}")

    scene_list = batch.get("scene_list", [])
    sc = scene_list[0] if scene_list else None

    if sc is None:
        print(f"  Scene: NONE!")
        continue

    print(f"  Scene keys: {list(sc.keys())}")
    print(f"  positions shape: {sc['positions'].shape}")
    print(f"  class_names type: {type(sc['class_names'])}")
    print(f"  class_names: {sc['class_names']}")
    print(f"  traffic_light_states (in scene): {sc.get('traffic_light_states', 'NOT FOUND')}")

    # Check positions
    pos = sc["positions"]  # (obs_len, max_N, 2)
    for t in range(pos.shape[0]):
        nz_count = (pos[t, :, 0] != 0).sum().item()
        if nz_count > 0:
            # Print first non-zero object
            for n in range(pos.shape[1]):
                if pos[t, n, 0] != 0:
                    cn = sc["class_names"][t][n] if t < len(sc["class_names"]) and n < len(sc["class_names"][t]) else "?"
                    print(f"    Frame {t}, obj {n}: ({pos[t,n,0]:.0f}, {pos[t,n,1]:.0f}) class='{cn}'")
                    break

    # Check P_cross manually with a simple check
    from src.classification.crossing_probability import point_in_polygon
    if junction_roi:
        last_x = obs[-1, 0].item()
        last_y = obs[-1, 1].item()
        in_junction = point_in_polygon(last_x, last_y, junction_roi)
        print(f"  Obs last pos in junction_roi: {in_junction}")

        tgt_x = target[-1, 0].item()
        tgt_y = target[-1, 1].item()
        tgt_in_junction = point_in_polygon(tgt_x, tgt_y, junction_roi)
        print(f"  Target last pos in junction_roi: {tgt_in_junction}")
