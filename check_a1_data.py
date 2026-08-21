"""Check precomputed data on A1."""
import paramiko, json, sys

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
        print(f"  stderr: {err[:200]}")
    return out

print("=== Precomputed data ===")
print(run("ls /root/red-light-prediction/data/precomputed/ 2>/dev/null | head -5"))
print(run("ls /root/red-light-prediction/data/precomputed/ 2>/dev/null | wc -l"))

print("\n=== Check one npz structure ===")
# Get first npz file
first = run("ls /root/red-light-prediction/data/precomputed/*.npz 2>/dev/null | head -1")
print(f"File: {first}")
if first:
    result = run(f"python3 -c \"import numpy as np; d=np.load('{first}',allow_pickle=True)['data'].item(); ks=list(d.keys())[:5]; print('Keys sample:', ks); k=ks[0]; v=d[k]; print('Frame', k, 'keys:', list(v.keys())); print('bboxes shape:', v['bboxes'].shape); print('positions[:3]:', v['positions'][:3]); print('class_names[:5]:', v['class_names'][:5])\"")
    print(result)

print("\n=== Domain map ===")
print(run("cat /root/red-light-prediction/data/domains/domain_labels_int.json 2>/dev/null | head -20"))

print("\n=== Trajectory sample check (positions coordinate system) ===")
print(run("python3 -c \"import numpy as np; d=np.load('/root/red-light-prediction/data/processed/trajectories/ch01_00000000000000100_20260127075320_20260127075548_198968/trajectories.npz',allow_pickle=True); ks=list(d.keys())[:3]; print('Keys:', ks); k=ks[0]; v=d[k].item(); print('positions[:3]:', v['positions'][:3]); print('frames[:3]:', v.get('frames', 'N/A')[:3])\""))

c.close()
