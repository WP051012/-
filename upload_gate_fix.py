"""Upload signal-gate fix (direction 1) files to A1."""
import paramiko, os

HOST = 'connect.cqa1.seetacloud.com'
PORT = 29463
USER = 'root'
PW = 'UglX1RnukHl0'
BASE = r'C:\Users\wangj\Desktop\闯红灯预测'

FILES = [
    ('data/dataset.py', '/root/red-light-prediction/data/dataset.py'),
    ('scripts/eval_flowchain_domain.py', '/root/red-light-prediction/scripts/eval_flowchain_domain.py'),
    ('scripts/smoke_test_gatefix.py', '/root/red-light-prediction/scripts/smoke_test_gatefix.py'),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
print('Connected')

sftp = c.open_sftp()
for local_rel, remote in FILES:
    local = os.path.join(BASE, local_rel)
    if os.path.exists(local):
        sftp.put(local, remote)
        print(f'  Uploaded: {local_rel} -> {remote}')
    else:
        print(f'  MISSING: {local}')
sftp.close()
c.close()
print('Done')
