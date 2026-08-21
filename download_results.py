"""Download FOMAML v3 checkpoints from cloud to local for offline inspection."""
import os
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 29463
USER = 'root'
PW = 'UglX1RnukHl0'
REMOTE = '/root/red-light-prediction'
LOCAL = r'C:\Users\wangj\Desktop\闯红灯预测\_cloud_results'

os.makedirs(LOCAL, exist_ok=True)

FILES = [
    ('checkpoints/fomaml_v3/best_fomaml.pt', 'best_fomaml.pt'),
    ('checkpoints/fomaml_v3/fomaml_final.pt', 'fomaml_final.pt'),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
for remote_rel, local_name in FILES:
    remote = f'{REMOTE}/{remote_rel}'
    local = os.path.join(LOCAL, local_name)
    sftp.get(remote, local)
    print(f'Downloaded {remote_rel} -> {local} ({os.path.getsize(local)} bytes)')
sftp.close()
c.close()
print('Done')
