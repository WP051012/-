import paramiko

DIAG = r'''
import sys, json
from collections import Counter
sys.path.insert(0, "/root/red-light-prediction")
from data.dataset import TrajectoryDataset
from src.classification.crossing_probability import compute_signal_factor

with open("/root/red-light-prediction/data/domains/domain_labels_int.json") as f:
    dmap = json.load(f)

ds = TrajectoryDataset(
    data_dir="data/processed/trajectories", label_dir="labels",
    obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
    target_classes=["pedestrian"], mode="with_scene",
    max_scene_samples=3000, max_samples=5000, domain_label_map=dmap,
)
print("total samples:", len(ds))

last_state_viol = Counter(); last_state_non = Counter()
sf_viol = Counter(); sf_non = Counter()
examples_viol = []; examples_non = []

for idx in range(len(ds)):
    s = ds[idx]
    scene = s.get("scene", {})
    tl = scene.get("traffic_light_states", [])
    is_viol = bool(s.get("is_violation", False))
    last = tl[-1] if tl else "EMPTY"
    sf = compute_signal_factor(tl)
    (last_state_viol if is_viol else last_state_non)[last] += 1
    (sf_viol if is_viol else sf_non)[sf] += 1
    if is_viol and len(examples_viol) < 3:
        examples_viol.append((idx, tl, list(s.get("obs_frames", []))[:3], sf))
    if not is_viol and len(examples_non) < 3:
        examples_non.append((idx, tl, list(s.get("obs_frames", []))[:3], sf))

print("\n=== last-frame state (violation) ===", dict(last_state_viol))
print("=== last-frame state (non-violation) ===", dict(last_state_non))
print("\n=== signal_factor (violation) ===", dict(sf_viol))
print("=== signal_factor (non-violation) ===", dict(sf_non))
print("\n=== violation examples (idx, tl_states, obs_frames[:3], sf) ===")
for e in examples_viol: print("  ", e)
print("=== non-violation examples ===")
for e in examples_non: print("  ", e)
'''

HOST = 'connect.cqa1.seetacloud.com'; PORT = 44037; USER = 'root'; PW = 'qaaPg/iZDISX'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
with sftp.open('/root/red-light-prediction/_diag_tl.py', 'w') as f:
    f.write(DIAG)
sftp.close()
_, o, e = c.exec_command("cd /root/red-light-prediction && /root/miniconda3/bin/python3.12 _diag_tl.py 2>&1")
print(o.read().decode('utf-8', 'replace'))
print(e.read().decode('utf-8', 'replace'))
c.close()
