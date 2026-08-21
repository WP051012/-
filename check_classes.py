"""Check class distribution in precomputed data."""
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'
PY = '/root/miniconda3/bin/python'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)

def run(cmd):
    _, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        print(f"  [stderr]: {err[:300]}")
    return out

print("=== Class distribution in precomputed data (sample 100 files) ===")
result = run(f"""{PY} -c "
import numpy as np, os, glob
from collections import Counter

dir = '/root/red-light-prediction/data/precomputed'
npz_files = sorted(glob.glob(os.path.join(dir, '*.npz')))[:100]

class_counts = Counter()
det_counts_given_ped = Counter()  # detection count when pedestrian IS present
has_ped = 0
no_ped = 0

for npz_path in npz_files:
    try:
        raw = np.load(npz_path, allow_pickle=True)
        data = raw['data'].item()
    except:
        continue
    for frame_key, fd in data.items():
        cns = list(fd.get('class_names', []))
        n = len(cns)
        for cn in cns:
            class_counts[cn] += 1
        if 'pedestrian' in cns:
            has_ped += 1
            det_counts_given_ped[n] += 1
        else:
            no_ped += 1

print(f'Frames with pedestrian: {{has_ped}}')
print(f'Frames without pedestrian: {{no_ped}}')
print()
print('All class types found:')
for cn, cnt in class_counts.most_common():
    print(f'  {{cn:20s}}: {{cnt:8d}}')
print()
print('Detection count when pedestrian IS present:')
for n in sorted(det_counts_given_ped.keys()):
    cnt = det_counts_given_ped[n]
    print(f'  {{n:3d}} detections: {{cnt:6d}} frames')
" """)
print(result)

c.close()
