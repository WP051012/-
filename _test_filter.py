"""Test the new is_crossing_candidate with junction_roi + 80-90 degree rule."""
import sys, yaml, numpy as np
sys.path.insert(0, '.')
from scripts.run_experiments import load_split_datasets
from data.dataset import is_crossing_candidate as icc

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

for name, ds in [('train', train), ('val', val), ('test', test)]:
    kept = 0; kept_viol = 0; total_viol = 0
    use_gt = (name == 'train')
    for i in range(len(ds)):
        s = ds[i]
        is_viol = s['is_violation']
        if is_viol: total_viol += 1
        obs = s['obs_trajectory'].numpy()
        tgt = s.get('target_trajectory')
        tgt_np = tgt.numpy() if tgt is not None else None
        if not use_gt: tgt_np = None

        if icc(obs, tgt_np, crosswalk_roi, stop_line, junction_roi):
            kept += 1
            if is_viol: kept_viol += 1

    print(f"{name}: {kept}/{len(ds)} kept ({100*kept/len(ds):.1f}%), "
          f"violations: {kept_viol}/{total_viol} kept "
          f"({100*kept_viol/max(1,total_viol):.1f}%)")

# Also check: how many train samples pass by condition A (GT junction) vs B (heading)?
print("\n--- Train set: breakdown by condition ---")
a_only = b_only = both = 0; a_viol = b_viol = both_viol = 0
for i in range(len(train)):
    s = train[i]
    is_viol = s['is_violation']
    obs = s['obs_trajectory'].numpy()
    tgt = s.get('target_trajectory').numpy()

    # Condition A: GT enters junction
    cond_a = False
    for k in range(tgt.shape[0]):
        from data.dataset import _point_in_polygon_px
        if _point_in_polygon_px(float(tgt[k,0]), float(tgt[k,1]), junction_roi):
            cond_a = True; break

    # Condition B: heading 80-90 degrees
    cond_b = False
    if len(obs) >= 4:
        vel = obs[-1] - obs[-4]
        vn = np.sqrt(float(vel[0])**2 + float(vel[1])**2)
        if vn > 1e-6:
            sldx = float(stop_line[2])-float(stop_line[0])
            sldy = float(stop_line[3])-float(stop_line[1])
            sln = np.sqrt(sldx**2 + sldy**2)
            if sln > 1e-6:
                ca = abs(float(vel[0])*sldx+float(vel[1])*sldy)/(vn*sln)
                cond_b = ca <= np.cos(np.deg2rad(80.0))

    if cond_a and cond_b: both += 1
    elif cond_a: a_only += 1
    elif cond_b: b_only += 1

    if is_viol:
        if cond_a and cond_b: both_viol += 1
        elif cond_a: a_viol += 1
        elif cond_b: b_viol += 1

print(f"A only (GT junction):   {a_only} ({a_viol} viol)")
print(f"B only (heading 80-90): {b_only} ({b_viol} viol)")
print(f"Both A+B:               {both} ({both_viol} viol)")
print(f"Total kept:             {a_only+b_only+both}")
print(f"Total viol kept:        {a_viol+b_viol+both_viol}")
