import sys; sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset
from src.classification.crossing_probability import compute_signal_factor
import random; random.seed(42)

ds = TrajectoryDataset('data/processed/trajectories', mode='trajectory_only')
for i in [0, 100, 500, 1000]:
    vn = ds.video_names[ds._idx_to_video[i]]
    tl = ds._load_traffic_lights(vn)
    print(f"sample {i}: video={vn}, tl_df={'YES' if tl is not None else 'NONE'}")

# Check signal factor distribution across first 100 scene samples
ds2 = TrajectoryDataset('data/processed/trajectories', mode='with_scene')
signals = []
for i in range(100):
    s = ds2[i]
    sig = compute_signal_factor(s.get("traffic_light_states", []))
    signals.append(sig)

from collections import Counter
print(f"\nSignal factor distribution (first 100 scene samples):")
for v, c in sorted(Counter(signals).items()):
    print(f"  {v:.1f}: {c}")
