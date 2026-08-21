"""Upload agent-centric risk files to cloud instance."""
import paramiko
import os
import sys

host = 'connect.cqa1.seetacloud.com'
port = 46863
user = 'root'
password = '2Lgv+Pw0rD+q'

remote_base = '/root/red-light-prediction'
local_base = r'C:\Users\wangj\Desktop\闯红灯预测'

files = [
    'src/classification/agent_centric_risk.py',
    'scripts/run_ourmethod_v2.py',
]

transport = paramiko.Transport((host, port))
transport.connect(username=user, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)

for f in files:
    local_path = os.path.join(local_base, f.replace('/', '\\'))
    remote_path = f'{remote_base}/{f}'
    try:
        sftp.put(local_path, remote_path)
        print(f'Uploaded: {f}')
    except Exception as e:
        # Try creating remote directory
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            # mkdir recursively
            parts = remote_dir.strip('/').split('/')
            for i in range(1, len(parts) + 1):
                subdir = '/' + '/'.join(parts[:i])
                try:
                    sftp.stat(subdir)
                except FileNotFoundError:
                    sftp.mkdir(subdir)
        sftp.put(local_path, remote_path)
        print(f'Uploaded (retry): {f}')

sftp.close()
transport.close()
print('All files uploaded successfully!')
