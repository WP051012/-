"""Upload analyze script and run on cloud."""
import paramiko
host = 'connect.cqa1.seetacloud.com'; port = 46863
user = 'root'; password = '2Lgv+Pw0rD+q'

# Upload
transport = paramiko.Transport((host, port))
transport.connect(username=user, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)
sftp.put(r'C:\Users\wangj\Desktop\闯红灯预测\analyze_env.py', '/root/red-light-prediction/analyze_env.py')
sftp.close(); transport.close()
print('Uploaded')

# Run
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=password)
cmd = 'cd /root/red-light-prediction && /root/miniconda3/bin/python analyze_env.py 2>&1'
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err: print('STDERR:', err[-500:])
client.close()
