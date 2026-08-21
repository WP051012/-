"""Quick verify traffic_lights.csv loading + signal_factor distribution."""
import sys; sys.path.insert(0, '.')
from data.dataset import TrajectoryDataset
from src.classification.crossing_probability import compute_signal_factor
from collections import Counter

# Check via scene dataset
ds = TrajectoryDataset('data/processed/trajectories', mode='with_scene')

signals = []
tl_loaded = 0
tl_none = 0
for i in range(500):
    s = ds[i]
    tl = s.get("traffic_light_states", [])
    sig = compute_signal_factor(tl)
    signals.append(sig)
    if tl and len(tl) > 0 and tl[0] != 'unknown':
        tl_loaded += 1
    else:
        tl_none += 1

print(f"Sample size: {len(signals)}")
print(f"With real TL data: {tl_loaded}")
print(f"Without TL data (unknown): {tl_none}")
print(f"\nSignal factor distribution:")
for v, c in sorted(Counter(signals).items()):
    print(f"  {v:.1f}: {c} ({100*c/len(signals):.1f}%)")

# Check unique states
unique_states = set()
for i in range(500):
    tl = ds[i].get("traffic_light_states", [])
    for state in tl:
        unique_states.add(state)
print(f"\nUnique traffic light states: {sorted(unique_states)}")
