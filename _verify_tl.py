"""Quick verify traffic_light_states are loaded via load_split_datasets."""
import sys; sys.path.insert(0, '.')
from scripts.run_experiments import load_split_datasets
from src.classification.crossing_probability import compute_signal_factor
from collections import Counter

# Load full scene datasets (not quick)
_, _, _, train, val, test = load_split_datasets(
    'data/processed/trajectories', label_dir='labels/', quick=False)

for name, ds in [('test', test), ('val', val)]:
    signals = []
    for i in range(min(200, len(ds))):
        s = ds[i]
        scene = s.get("scene", {})
        tl = scene.get("traffic_light_states", [])
        signals.append(compute_signal_factor(tl))
    c = Counter(signals)
    print(f"{name}: n={len(signals)}, signal_factor={dict(c)}")

# Also check a few explicit samples
s0 = test[0]
scene = s0.get("scene", {})
tl = scene.get("traffic_light_states", [])
print(f"\nSample test[0]: traffic_light_states={tl}")
print(f"  video={s0['video']}")
print(f"  is_violation={s0['is_violation']}")
