"""Count detections per frame in precomputed data."""
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
        print(f"  [stderr]: {err[:200]}")
    return out

print("=== Detection count distribution across ALL precomputed frames ===")
result = run(f"""{PY} -c "
import numpy as np, os, glob
from collections import Counter

precomputed_dir = '/root/red-light-prediction/data/precomputed'
npz_files = sorted(glob.glob(os.path.join(precomputed_dir, '*.npz')))

det_counts = Counter()
total_frames = 0

for npz_path in npz_files:
    try:
        raw = np.load(npz_path, allow_pickle=True)
        data = raw['data'].item()
    except:
        continue
    for frame_key, fd in data.items():
        n = len(fd.get('bboxes', []))
        det_counts[n] += 1
        total_frames += 1

print(f'Total npz files: {{len(npz_files)}}')
print(f'Total frames: {{total_frames}}')
print()
print('Detections per frame distribution:')
for n in sorted(det_counts.keys()):
    cnt = det_counts[n]
    pct = cnt / total_frames * 100
    bar = '#' * int(pct / 2)
    print(f'  {{n:3d}} detections: {{cnt:7d}} frames ({{pct:5.1f}}%) {{bar}}')
" """)
print(result)

c.close()
