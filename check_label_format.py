"""Check actual label file format."""
import paramiko, io

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

print(run("head -20 /root/red-light-prediction/labels/ch01_00000000001000000_20260127075543_20260127075842_685001.txt"))
print("---")
print(run("wc -l /root/red-light-prediction/labels/ | head -5"))
print("---")
# How does dataset read labels?
print(run("grep -n 'label_dir\\|_get_scene_data\\|label.*txt\\|with_scene' /root/red-light-prediction/data/dataset.py | head -20"))

c.close()
