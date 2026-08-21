"""Check precomputed data structure on A1 (use python not python3)."""
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'

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

# Find python
print("=== Python ===")
for py in ['python', 'python3', '/usr/bin/python3', '/opt/conda/bin/python']:
    r = run(f"which {py} 2>/dev/null || echo 'not found: {py}'")
    print(r)

# Use python
print("\n=== NPZ structure ===")
first = run("ls /root/red-light-prediction/data/precomputed/*.npz 2>/dev/null | head -1")
print(f"File: {first}")
result = run(f"python -c \"import numpy as np; d=np.load('{first}',allow_pickle=True)['data'].item(); ks=list(d.keys())[:3]; print('Keys:', ks); k=ks[0]; v=d[k]; print('Frame', k, 'keys:', list(v.keys())); print('bboxes:', v['bboxes'].shape); print('positions[:3]:', v['positions'][:3]); print('class_names[:5]:', v['class_names'][:5])\"")
print(result if result else "(empty)")

print("\n=== Trajectory positions ===")
traj_npz = run("ls /root/red-light-prediction/data/processed/trajectories/ch01_00000000000000100_*/*.npz 2>/dev/null | head -1")
print(f"File: {traj_npz}")
result = run(f"python -c \"import numpy as np; d=np.load('{traj_npz}',allow_pickle=True); ks=list(d.keys())[:2]; print('Keys:', ks); k=ks[0]; v=d[k].item(); print('positions[:3]:', v['positions'][:3]); print('frames[:3]:', v.get('frames', 'N/A')[:3] if v.get('frames') is not None else 'no frames')\"")
print(result if result else "(empty)")

c.close()
