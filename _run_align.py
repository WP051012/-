import paramiko

DIAG = r'''
import glob, os
import numpy as np
import pandas as pd

base = "/root/red-light-prediction/data/processed/trajectories"
npz_files = sorted(glob.glob(base + "/*/trajectories.npz"))
print("num npz:", len(npz_files))

for npz_path in npz_files[:5]:
    video = os.path.basename(os.path.dirname(npz_path))
    data = np.load(npz_path, allow_pickle=True)
    # collect frames ranges
    fr_min, fr_max = None, None
    n_traj = 0
    for tid in data.files:
        try:
            td = data[tid].item()
        except Exception:
            continue
        fr = td.get("frames")
        if fr is None:
            continue
        n_traj += 1
        fmin, fmax = int(fr.min()), int(fr.max())
        fr_min = fmin if fr_min is None else min(fr_min, fmin)
        fr_max = fmax if fr_max is None else max(fr_max, fmax)
    # traffic lights csv
    tl_csv = os.path.join(os.path.dirname(npz_path), "traffic_lights.csv")
    if os.path.exists(tl_csv):
        tl = pd.read_csv(tl_csv)
        tl_min, tl_max = int(tl["frame_id"].min()), int(tl["frame_id"].max())
        tl_step = int(np.min(np.diff(np.unique(tl["frame_id"])))) if len(tl["frame_id"]) > 1 else 0
    else:
        tl_min = tl_max = tl_step = None
    print(f"\n{video}: {n_traj} trajectories, frames range [{fr_min}, {fr_max}]")
    print(f"  traffic_lights.csv frame_id range [{tl_min}, {tl_max}] step={tl_step}")
    if fr_min is not None and tl_min is not None:
        overlap = fr_max >= tl_min and fr_min <= tl_max
        print(f"  overlap: {overlap}")
'''

HOST = 'connect.cqa1.seetacloud.com'; PORT = 44037; USER = 'root'; PW = 'qaaPg/iZDISX'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
with sftp.open('/root/red-light-prediction/_align.py', 'w') as f:
    f.write(DIAG)
sftp.close()
_, o, e = c.exec_command("cd /root/red-light-prediction && /root/miniconda3/bin/python3.12 _align.py 2>&1")
print(o.read().decode('utf-8', 'replace'))
print(e.read().decode('utf-8', 'replace'))
c.close()
