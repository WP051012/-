import paramiko

DIAG = r'''
import glob, os, csv, json
from collections import Counter
import numpy as np
import pandas as pd

base = "/root/red-light-prediction/data/processed/trajectories"
with open("/root/red-light-prediction/data/domains/domain_labels_int.json") as f:
    dmap = json.load(f)

# violation map: (video, track_key) -> bool
viol_map = {}
for vp in glob.glob(base + "/*/violation_labels.csv"):
    video = os.path.basename(os.path.dirname(vp))
    with open(vp) as f:
        for row in csv.DictReader(f):
            key = row.get("track_key", "")
            val = str(row.get("is_violation", "0")).strip().lower()
            viol_map[(video, key)] = val in ("true", "1")

state_viol = Counter(); state_non = Counter()
has_csv = Counter()
n_viol = 0; n_non = 0

npz_files = sorted(glob.glob(base + "/*/trajectories.npz"))
for npz_path in npz_files:
    video = os.path.basename(os.path.dirname(npz_path))
    did = dmap.get(video, -1)
    if did != 5:  # test domain only
        continue
    # load traffic lights
    tl_csv = os.path.join(os.path.dirname(npz_path), "traffic_lights.csv")
    tl = pd.read_csv(tl_csv) if os.path.exists(tl_csv) else None
    has_csv["with_csv" if tl is not None else "no_csv"] += 1

    data = np.load(npz_path, allow_pickle=True)
    for tid in data.files:
        try:
            td = data[tid].item()
        except Exception:
            continue
        positions = td.get("positions")
        frames = td.get("frames")
        if positions is None or frames is None:
            continue
        T = positions.shape[0]
        if T < 20:
            continue
        is_viol = viol_map.get((video, tid), False)
        for start in range(0, T - 20, 8):
            last_frame = int(frames[start + 7])
            if tl is None:
                state = "unknown"
            else:
                idx = (tl["frame_id"] - last_frame).abs().idxmin()
                state = str(tl.iloc[idx].get("overall_state", "unknown"))
            if is_viol:
                state_viol[state] += 1; n_viol += 1
            else:
                state_non[state] += 1; n_non += 1

print("test-domain videos:", dict(has_csv))
print("total violation samples:", n_viol, " non-violation:", n_non)
print("\n=== last-obs-frame state (violation) ===")
for k, v in state_viol.most_common():
    print(f"  {k:10s} {v:6d}  ({100*v/max(1,n_viol):.1f}%)")
print("=== last-obs-frame state (non-violation) ===")
for k, v in state_non.most_common():
    print(f"  {k:10s} {v:6d}  ({100*v/max(1,n_non):.1f}%)")
'''

HOST = 'connect.cqa1.seetacloud.com'; PORT = 44037; USER = 'root'; PW = 'qaaPg/iZDISX'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
with sftp.open('/root/red-light-prediction/_state.py', 'w') as f:
    f.write(DIAG)
sftp.close()
_, o, e = c.exec_command("cd /root/red-light-prediction && /root/miniconda3/bin/python3.12 _state.py 2>&1")
print(o.read().decode('utf-8', 'replace'))
print(e.read().decode('utf-8', 'replace'))
c.close()
