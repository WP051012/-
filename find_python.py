"""Find Python on A1."""
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
    return stdout.read().decode().strip(), stderr.read().decode().strip()

print("=== Find Python ===")
for loc in [
    'find / -name python3.10 -type f 2>/dev/null | head -5',
    'find / -name python -type f 2>/dev/null | head -10',
    'ls /root/miniconda3/bin/python* 2>/dev/null',
    'ls /opt/conda/bin/python* 2>/dev/null',
    'which python3 2>/dev/null || which python 2>/dev/null || echo "not in PATH"',
    'ls /root/ 2>/dev/null | head -20',
    'cat ~/.bashrc 2>/dev/null | grep -i conda | head -5',
]:
    out, err = run(loc)
    if out:
        print(f"  [{loc[:60]}...] {out[:200]}")

# Check if the training script used a specific python
print("\n=== Check older script for python path ===")
out, _ = run("head -5 /root/red-light-prediction/scripts/train_fomaml.py 2>/dev/null || head -5 /root/red-light-prediction/train_fomaml.py 2>/dev/null")
print(f"  {out[:100]}")

# Check bash history for python usage
out, _ = run("tail -20 ~/.bash_history 2>/dev/null | grep python | head -5")
print(f"\n=== Bash history python ===\n  {out}")

c.close()
