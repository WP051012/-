"""Read-only: locate and tail the FOMAML v3 training log on the cloud instance.

Only runs read-only shell commands (ls / tail). Never triggers any training,
eval, or state-mutating script.
"""
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 29463
USER = 'root'
PW = 'UglX1RnukHl0'
REMOTE = '/root/red-light-prediction'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)

def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    print(f"$ {cmd}")
    if out:
        print(out)
    if err:
        print("[stderr]", err)
    print("-" * 70)

# Recently modified files (last 4 hours) anywhere shallow
run(f"find {REMOTE} -maxdepth 3 -type f -mmin -240 -printf '%T+ %s %p\\n' 2>/dev/null | sort -r | head -40")

# logs/ dir by recency
run(f"ls -lt {REMOTE}/logs/ 2>/dev/null | head -20")

c.close()
print("Done")
