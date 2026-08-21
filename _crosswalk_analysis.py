"""Analyze: how many FlowChain predicted trajectories cross the crosswalk?"""
import sys; sys.path.insert(0, '.')
import yaml, torch, numpy as np

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# Parse all ROIs
crosswalk_roi = None; junction_roi = None; stop_line = None
for key in ("intersection_A", "intersection_B"):
    c = config.get(key, {})
    cw = c.get("crosswalk_roi"); jr = c.get("junction_roi"); sl = c.get("stop_line")
    if cw and len(cw) >= 3:
        if isinstance(cw[0], (list, tuple)):
            crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
        else:
            crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
    if jr and len(jr) >= 3:
        junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
        if len(junction_roi) == 2:
            x1, y1 = junction_roi[0]; x2, y2 = junction_roi[1]
            junction_roi = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if sl and len(sl) >= 4: stop_line = [float(x) for x in sl]
    if crosswalk_roi and junction_roi: break

# Normalize
cw_norm = [(x/3840.0, y/2160.0) for (x, y) in crosswalk_roi]
jr_norm = [(x/3840.0, y/2160.0) for (x, y) in junction_roi]

print(f"Crosswalk ROI (norm): {cw_norm}")
print(f"Junction ROI (norm):  {jr_norm}")

# Point-in-polygon
def point_in_polygon(x, y, poly):
    n = len(poly)
    if n < 3: return False
    inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside

def trajectory_crosses_region(traj, region_poly):
    """Check if any point of trajectory falls inside region polygon."""
    for pt in traj:
        if point_in_polygon(float(pt[0]), float(pt[1]), region_poly):
            return True
    return False

# Load data
from scripts.run_experiments import load_split_datasets
from src.baselines.baseline_models import FlowChainBase
from src.classification.crossing_probability import compute_signal_factor

DEVICE = "cuda"
NORM = torch.tensor([3840.0, 2160.0])
NUM_MC = 100

_, _, _, train_scene, val_scene, test_scene = load_split_datasets(
    'data/processed/trajectories', label_dir='labels/', quick=False)
print(f"Test scene samples: {len(test_scene)}")

# Load FlowChain
flowchain = FlowChainBase(obs_len=8, pred_len=12, d_model=64, nvp_num_blocks=3).to(DEVICE)
ckpt = torch.load('checkpoints/flowchain_best.pt', map_location=DEVICE, weights_only=False)
flowchain.load_state_dict(ckpt); flowchain.eval()
norm_tensor = NORM.to(DEVICE)

results = []
for idx in range(len(test_scene)):
    s = test_scene[idx]
    is_viol = s["is_violation"]
    obs_raw = s["obs_trajectory"]
    scene = s.get("scene", {})
    tl_states = scene.get("traffic_light_states", [])
    signal = compute_signal_factor(tl_states)

    # FlowChain inference
    obs = obs_raw.unsqueeze(0).to(DEVICE) / norm_tensor
    with torch.no_grad():
        pred = flowchain(obs_trajectory=obs, num_samples=NUM_MC)
    samples = pred.get("samples")

    if samples is None:
        n_cross_cw = 0; n_cross_jr = 0
    else:
        s_px = samples[:, 0] * norm_tensor  # (100, 12, 2) in pixels
        n_cross_cw = sum(1 for k in range(NUM_MC) if trajectory_crosses_region(s_px[k], crosswalk_roi))
        n_cross_jr = sum(1 for k in range(NUM_MC) if trajectory_crosses_region(s_px[k], junction_roi))

    results.append({
        'idx': idx, 'is_viol': is_viol,
        'signal': signal, 'signal_label': {1.0:'red', 0.7:'yellow', 0.5:'unknown', 0.0:'green'}[signal],
        'n_cross_cw': n_cross_cw, 'n_cross_jr': n_cross_jr,
        'crosses_cw': n_cross_cw > 0, 'crosses_jr': n_cross_jr > 0,
    })

    if (idx + 1) % 200 == 0:
        print(f"  Progress: {idx+1}/{len(test_scene)}")

# Aggregate statistics
print(f"\n{'='*70}")
print(f"FLOWCHAIN CROSSWALK/JUNCTION CROSSING ANALYSIS (N={NUM_MC} MC samples)")
print(f"{'='*70}")

# Helper to compute stats
def stats(subset, label):
    n = len(subset)
    n_cross_cw = sum(1 for r in subset if r['crosses_cw'])
    n_cross_jr = sum(1 for r in subset if r['crosses_jr'])
    avg_cw = np.mean([r['n_cross_cw'] for r in subset])
    avg_jr = np.mean([r['n_cross_jr'] for r in subset])
    print(f"\n{label} (n={n}):")
    print(f"  ≥1 trajectory crosses crosswalk: {n_cross_cw}/{n} ({100*n_cross_cw/max(1,n):.1f}%)")
    print(f"  ≥1 trajectory crosses junction:  {n_cross_jr}/{n} ({100*n_cross_jr/max(1,n):.1f}%)")
    print(f"  Avg MC samples crossing crosswalk: {avg_cw:.2f}/{NUM_MC}")
    print(f"  Avg MC samples crossing junction:  {avg_jr:.2f}/{NUM_MC}")

# All test
stats(results, "ALL TEST SAMPLES")
# By violation
stats([r for r in results if r['is_viol']], "VIOLATIONS")
stats([r for r in results if not r['is_viol']], "NON-VIOLATIONS")

# By signal
for sig_val, sig_name in [(1.0, 'RED LIGHT'), (0.7, 'YELLOW LIGHT'), (0.5, 'UNKNOWN SIGNAL'), (0.0, 'GREEN LIGHT')]:
    subset = [r for r in results if r['signal'] == sig_val]
    if subset:
        stats(subset, sig_name)

# Violation breakdown by signal
print(f"\n{'='*70}")
print(f"VIOLATION DISTRIBUTION BY SIGNAL")
print(f"{'='*70}")
viols = [r for r in results if r['is_viol']]
for sig_val, sig_name in [(1.0, 'Red'), (0.7, 'Yellow'), (0.5, 'Unknown'), (0.0, 'Green')]:
    n = sum(1 for r in viols if r['signal'] == sig_val)
    print(f"  {sig_name}: {n}/{len(viols)} violations ({100*n/max(1,len(viols)):.0f}%)")
    if n > 0:
        crosses = sum(1 for r in viols if r['signal'] == sig_val and r['crosses_cw'])
        print(f"    of which FlowChain predicts crossing crosswalk: {crosses}/{n}")

# Detailed: violations that DO cross
print(f"\n{'='*70}")
print(f"VIOLATIONS WHERE FLOWCHAIN PREDICTS CROSSING (at least 1 MC sample)")
print(f"{'='*70}")
cross_viols = [r for r in viols if r['crosses_cw']]
print(f"Total: {len(cross_viols)}/{len(viols)}")
for r in cross_viols:
    print(f"  idx={r['idx']:4d}  signal={r['signal_label']:7s}  "
          f"MC_cross_cw={r['n_cross_cw']:3d}/{NUM_MC}  MC_cross_jr={r['n_cross_jr']:3d}/{NUM_MC}")

# Non-violations that DO cross (potential false positives)
print(f"\n{'='*70}")
print(f"NON-VIOLATIONS WHERE FLOWCHAIN PREDICTS CROSSING (potential FPs)")
print(f"{'='*70}")
fp_cross = [r for r in results if not r['is_viol'] and r['crosses_cw']]
print(f"Total: {len(fp_cross)}/{len(results)-len(viols)}")
# Show top 10 by most MC samples
fp_cross.sort(key=lambda r: r['n_cross_cw'], reverse=True)
for r in fp_cross[:10]:
    print(f"  idx={r['idx']:4d}  signal={r['signal_label']:7s}  "
          f"MC_cross_cw={r['n_cross_cw']:3d}/{NUM_MC}  MC_cross_jr={r['n_cross_jr']:3d}/{NUM_MC}")
