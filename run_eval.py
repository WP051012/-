"""Upload + run eval with nohup for reliability."""
import paramiko, os, time

host = 'connect.cqa1.seetacloud.com'; port = 46863
user = 'root'; password = '2Lgv+Pw0rD+q'
remote_base = '/root/red-light-prediction'
local_base = r'C:\Users\wangj\Desktop\闯红灯预测'

print('Connecting...')
transport = paramiko.Transport((host, port))
transport.connect(username=user, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)

# Upload files
for local_rel, remote_rel in [
    (r'src\classification\risk_estimator.py', 'src/classification/risk_estimator.py'),
    (r'src\classification\agent_centric_risk.py', 'src/classification/agent_centric_risk.py'),
    (r'scripts\eval_flowchain_agent.py', 'scripts/eval_flowchain_agent.py'),
]:
    local_path = os.path.join(local_base, local_rel)
    remote_path = f'{remote_base}/{remote_rel}'
    sftp.put(local_path, remote_path)
    print(f'  Uploaded: {remote_rel}')

sftp.close(); transport.close()
print('Upload done.')

# Run with nohup to survive SSH disconnect
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=password)

cmd = f'cd {remote_base} && nohup /root/miniconda3/bin/python scripts/eval_flowchain_agent.py > /tmp/eval_result.txt 2>&1 &'
stdin, stdout, stderr = client.exec_command(cmd)
print('Eval launched in background. Waiting 360s...')
client.close()

# Wait for completion (dataset loading ~3min + eval ~40s)
time.sleep(360)

# Fetch results
client2 = paramiko.SSHClient()
client2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client2.connect(host, port=port, username=user, password=password)
stdin, stdout, stderr = client2.exec_command('cat /tmp/eval_result.txt 2>&1')
output = stdout.read().decode('utf-8', errors='replace')
print(output)
err = stderr.read().decode('utf-8', errors='replace')
if err: print('STDERR:', err[-1000:])
client2.close()
