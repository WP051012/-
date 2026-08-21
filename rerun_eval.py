"""Upload updated files and re-run eval with motion+traj only."""
import paramiko, os

host = 'connect.cqa1.seetacloud.com'; port = 46863
user = 'root'; password = '2Lgv+Pw0rD+q'

local_base = r'C:\Users\wangj\Desktop\闯红灯预测'
remote_base = '/root/red-light-prediction'

# Upload both files
transport = paramiko.Transport((host, port))
transport.connect(username=user, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)

files_to_upload = [
    r'src\classification\agent_centric_risk.py',
    r'scripts\eval_flowchain_agent.py',
]
for f in files_to_upload:
    local_path = os.path.join(local_base, f)
    remote_path = f'{remote_base}/{f.replace(chr(92), "/")}'
    sftp.put(local_path, remote_path)
    print(f'Uploaded: {f}')

sftp.close(); transport.close()
print('All files uploaded')

# Run eval
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=password)
cmd = f'cd {remote_base} && /root/miniconda3/bin/python scripts/eval_flowchain_agent.py 2>&1'
stdin, stdout, stderr = client.exec_command(cmd)
output = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(output)
if err:
    print('STDERR:', err[-1000:])
client.close()
