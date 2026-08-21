"""Check if precompute script is running and progress."""
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

print("=== Process status ===")
print(run("ps aux | grep precompute | grep -v grep"))

print("\n=== GPU status ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found'"))

print("\n=== Output file ===")
print(run("ls -lh /root/red-light-prediction/data/gat_conditions.pt 2>/dev/null || echo 'Not yet created'"))

c.close()
