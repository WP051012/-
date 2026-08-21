"""Debug: trace why traffic_light_states is always empty."""
import sys; sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset

ds = TrajectoryDataset('data/processed/trajectories', mode='with_scene')

# Get a sample with has_scene
has_scene_indices = [i for i in range(min(1000, len(ds))) if ds.samples[i].get("has_scene")]
print(f"has_scene in first 1000: {len(has_scene_indices)}")
if has_scene_indices:
    idx = has_scene_indices[0]
    s = ds[idx]
    print(f"Sample {idx}: video={s['video']}, has_scene={ds.samples[idx].get('has_scene')}")
    scene = s.get("scene", {})
    tl = scene.get("traffic_light_states", [])
    print(f"scene keys: {list(scene.keys())}")
    print(f"traffic_light_states: {tl}")
    print(f"obs_frames from sample: {ds.samples[idx].get('obs_frames', 'N/A')[:5]}...")

    # Also test _load_traffic_lights directly
    vn = s["video"]
    tl_df = ds._load_traffic_lights(vn)
    print(f"Direct _load_traffic_lights: {'YES' if tl_df is not None else 'NONE'}")
    if tl_df is not None:
        print(f"  Shape: {tl_df.shape}, columns: {list(tl_df.columns)}")
        print(f"  Head:\n{tl_df.head(3)}")
else:
    print("No scene samples found in first 1000!")
    # Check if any sample has has_scene
    for i in range(0, min(10000, len(ds)), 100):
        if ds.samples[i].get("has_scene"):
            print(f"  Found has_scene at index {i}")
            break
    else:
        print("  No has_scene in any sampled index!")
