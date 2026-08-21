"""Debug: candidate filtering vs GT crossing vs FlowChain predictions."""
import sys, yaml, numpy as np, torch
sys.path.insert(0, '.')
from scripts.run_experiments import load_split_datasets
from data.dataset import is_crossing_candidate
from src.classification.crossing_probability import compute_signal_factor

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

crosswalk_roi = None; stop_line = None; junction_roi = None
for key in ('intersection_A', 'intersection_B'):
    c = config.get(key, {})
    cw = c.get('crosswalk_roi'); jr = c.get('junction_roi'); sl = c.get('stop_line')
    if cw and len(cw) >= 3:
        if isinstance(cw[0], (list, tuple)):
            crosswalk_roi = [(float(p[0]), float(p[1])) for p in cw]
        else:
            crosswalk_roi = [(float(cw[i]), float(cw[i+1])) for i in range(0, len(cw)//2*2, 2)]
    if jr and len(jr) >= 3:
        junction_roi = [(float(jr[i]), float(jr[i+1])) for i in range(0, len(jr)//2*2, 2)]
        if len(junction_roi) == 2:
            x1,y1=junction_roi[0]; x2,y2=junction_roi[1]
            junction_roi = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    if sl and len(sl) >= 4: stop_line = [float(x) for x in sl]
    if crosswalk_roi and junction_roi: break

_, _, _, train, val, test = load_split_datasets(
    'data/processed/trajectories', label_dir='labels/', quick=False)

def pip(x, y, poly):
    n = len(poly)
    if n < 3: return False
    inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
        j = i
    return inside

total = len(test)
viol_count = 0
pass_cand = 0; pass_cand_viol = 0
gt_crosses_cw = 0; gt_crosses_cw_viol = 0
fail_viols = []
green_viols = []

for i in range(total):
    s = test[i]
    obs = s['obs_trajectory'].numpy()
    tgt = s.get('target_trajectory')
    tgt_np = tgt.numpy() if tgt is not None else None
    is_viol = s['is_violation']
    if is_viol: viol_count += 1

    # Check if GT crosses crosswalk (entire obs+target)
    if tgt_np is not None:
        full_traj = np.concatenate([obs, tgt_np], axis=0)  # obs[-1] + target
        crosses_cw = any(pip(float(p[0]), float(p[1]), crosswalk_roi) for p in full_traj)
        if crosses_cw:
            gt_crosses_cw += 1
            if is_viol: gt_crosses_cw_viol += 1

    # Check candidate filter
    cand = is_crossing_candidate(obs, tgt_np, crosswalk_roi, stop_line)
    if cand:
        pass_cand += 1
        if is_viol: pass_cand_viol += 1
    elif is_viol:
        scene = s.get('scene', {})
        tl = scene.get('traffic_light_states', [])
        sig = compute_signal_factor(tl)
        last_pt = obs[-1]
        last_tgt = tgt_np[-1] if tgt_np is not None else last_pt
        fail_viols.append((i, last_pt, last_tgt, sig))

    # Green light violations
    if is_viol:
        scene = s.get('scene', {})
        tl = scene.get('traffic_light_states', [])
        sig = compute_signal_factor(tl)
        if sig == 0.0:
            green_viols.append((i, obs, tgt_np, tl))

print("=" * 60)
print("TEST SET — CANDIDATE FILTER ANALYSIS")
print("=" * 60)
print(f"Total: {total}, Violations: {viol_count}")
print(f"Pass candidate filter: {pass_cand} ({100*pass_cand/total:.1f}%)")
print(f"  Violations passing: {pass_cand_viol}/{viol_count}")
print(f"  Violations FAILING:  {viol_count-pass_cand_viol}/{viol_count}")
print(f"GT crosses crosswalk: {gt_crosses_cw} ({100*gt_crosses_cw/total:.1f}%)")
print(f"  Violations GT crosses: {gt_crosses_cw_viol}/{viol_count}")

print(f"\n--- Violations that FAIL candidate filter ({len(fail_viols)}) ---")
for idx, last_pt, last_tgt, sig in fail_viols:
    sig_name = {1.0:'red', 0.7:'yellow', 0.5:'unknown', 0.0:'GREEN'}[sig]
    print(f"  idx={idx:4d} last_obs=({last_pt[0]:7.1f},{last_pt[1]:7.1f}) "
          f"last_tgt=({last_tgt[0]:7.1f},{last_tgt[1]:7.1f}) signal={sig_name}")

print(f"\n--- GREEN LIGHT violations ({len(green_viols)}) ---")
for idx, obs, tgt_np, tl in green_viols:
    print(f"  idx={idx}: tl_states={tl[:4]}...")
    # Check: does GT cross crosswalk?
    if tgt_np is not None:
        crosses = any(pip(float(p[0]), float(p[1]), crosswalk_roi) for p in tgt_np)
        print(f"    GT crosses crosswalk: {crosses}")
    # Check: is pedestrian near crosswalk at obs[-1]?
    last = obs[-1]
    dist_cw = np.min([np.sqrt((last[0]-p[0])**2+(last[1]-p[1])**2) for p in crosswalk_roi])
    print(f"    dist to crosswalk: {dist_cw:.0f}px, last_pos=({last[0]:.0f},{last[1]:.0f})")

# Also check: do candidate filter conditions individually explain failures?
print(f"\n--- Breakdown of why violations fail candidate filter ---")
for idx, last_pt, last_tgt, sig in fail_viols:
    obs_np = test[idx]['obs_trajectory'].numpy()
    tgt_np = test[idx].get('target_trajectory')
    tgt_np = tgt_np.numpy() if tgt_np is not None else None

    # Condition A
    from data.dataset import _min_dist_to_polygon
    d = _min_dist_to_polygon(float(last_pt[0]), float(last_pt[1]), crosswalk_roi)
    cond_a = d < 80.0

    # Condition B
    cond_b = False
    if stop_line and len(obs_np) >= 4:
        vel = obs_np[-1] - obs_np[-4]
        v_angle = np.arctan2(float(vel[1]), float(vel[0]))
        sl_dx = float(stop_line[2]) - float(stop_line[0])
        sl_dy = float(stop_line[3]) - float(stop_line[1])
        cross_a1 = np.arctan2(-sl_dx, sl_dy)
        cross_a2 = np.arctan2(sl_dx, -sl_dy)
        def _adiff(a, b):
            d = abs(a - b)
            return min(d, 2 * np.pi - d)
        th = np.deg2rad(45.0)
        cond_b = _adiff(v_angle, cross_a1) < th or _adiff(v_angle, cross_a2) < th

    # Condition C
    cond_c = False
    if tgt_np is not None:
        for i in range(tgt_np.shape[0]):
            if pip(float(tgt_np[i, 0]), float(tgt_np[i, 1]), crosswalk_roi):
                cond_c = True
                break

    sig_name = {1.0:'red', 0.7:'yellow', 0.5:'unknown', 0.0:'GREEN'}[sig]
    print(f"  idx={idx:4d} sig={sig_name:7s} dist={d:6.0f}px condA={cond_a} condB={cond_b} condC={cond_c}")
