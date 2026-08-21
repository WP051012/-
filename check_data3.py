"""Check data format on A1."""
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

# NPZ structure
first = run("ls /root/red-light-prediction/data/precomputed/*.npz 2>/dev/null | head -1")
print(f"=== NPZ: {first} ===")
result = run(f"""{PY} -c "
import numpy as np
d = np.load('{first}', allow_pickle=True)['data'].item()
ks = list(d.keys())[:5]
print('Sample keys:', ks)
k = ks[2] if len(ks) > 2 else ks[0]
v = d[k]
print('Frame', k)
print('  keys:', list(v.keys()))
for key in ['bboxes', 'positions', 'velocities', 'class_names', 'class_ids']:
    if key in v:
        arr = v[key]
        print(f'  {{key}}: shape={{arr.shape}} min={{arr.min():.1f}} max={{arr.max():.1f}}')
print('  class_names[:8]:', v['class_names'][:8])
" """)
print(result)

# Trajectory positions
traj = run("ls /root/red-light-prediction/data/processed/trajectories/ch01_00000000000000100_*/*.npz 2>/dev/null | head -1")
print(f"\n=== Traj: {traj} ===")
result = run(f"""{PY} -c "
import numpy as np
d = np.load('{traj}', allow_pickle=True)
ks = list(d.keys())[:3]
print('Keys:', ks)
k = ks[0]
v = d[k].item()
print('positions[:3]:', v['positions'][:3])
print('positions dtype:', v['positions'].dtype)
print('frames:', v.get('frames', 'N/A'))
" """)
print(result)

c.close()
