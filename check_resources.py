"""Check what detection/precomputation tools and raw data exist on A1."""
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

print("=== Raw videos? ===")
print(run("ls /root/red-light-prediction/data/raw_video* 2>/dev/null || echo 'no raw_video'"))
print(run("ls /root/red-light-prediction/D:/Red-Light*/ 2>/dev/null || echo 'no D: drive'"))
print(run("find /root/red-light-prediction -name '*.mp4' -o -name '*.avi' -o -name '*.mov' 2>/dev/null | head -5 || echo 'no video files'"))
print(run("du -sh /root/red-light-prediction/data/ 2>/dev/null"))

print("\n=== Detection/tracking scripts ===")
print(run("ls /root/red-light-prediction/scripts/ 2>/dev/null"))
print(run("ls /root/red-light-prediction/*.py 2>/dev/null | head -20"))

print("\n=== Precompute scripts ===")
print(run("find /root/red-light-prediction -name 'precompute*' -o -name '*scene*' 2>/dev/null | head -10"))

print("\n=== Disk space ===")
print(run("df -h /root/ 2>/dev/null"))

print("\n=== GPU ===")
print(run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null"))

print("\n=== labels/ directory ===")
print(run("ls /root/red-light-prediction/labels/ 2>/dev/null | head -5 || echo 'no labels dir'"))
print(run("find /root/red-light-prediction -maxdepth 2 -name 'labels' -type d 2>/dev/null"))

c.close()
