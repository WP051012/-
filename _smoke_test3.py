"""Quick verify traffic_light_states INSIDE scene dict."""
import sys; sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset
from src.classification.crossing_probability import compute_signal_factor
from collections import Counter

ds = TrajectoryDataset('data/processed/trajectories', mode='with_scene')

signals = []
has_tl = 0
no_tl = 0
unique_states = set()

for i in range(500):
    s = ds[i]
    scene = s.get("scene", {})
    tl = scene.get("traffic_light_states", [])
    sig = compute_signal_factor(tl)
    signals.append(sig)
    for state in tl:
        unique_states.add(state)
    if tl and len(tl) > 0 and any(st != 'unknown' for st in tl):
        has_tl += 1
    else:
        no_tl += 1

print(f"Samples: {len(signals)}")
print(f"With real TL data: {has_tl}")
print(f"Without TL data: {no_tl}")
print(f"Unique states: {sorted(unique_states)}")
print(f"\nSignal factor distribution:")
for v, c in sorted(Counter(signals).items()):
    print(f"  {v:.1f}: {c} ({100*c/len(signals):.1f}%)")

# Check overall_state values across a few videos
from pathlib import Path
import pandas as pd
data_dir = Path('data/processed/trajectories')
video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
for vd in video_dirs[:3]:
    csv_path = vd / 'traffic_lights.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        states = df['overall_state'].value_counts().to_dict()
        print(f"{vd.name}: {states}")
