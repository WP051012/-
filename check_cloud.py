"""Check cloud results."""
import paramiko, os

host = 'connect.cqa1.seetacloud.com'; port = 46863
user = 'root'

# Read password from the same source as upload_to_cloud.py
password = open(r'C:\Users\wangj\Desktop\闯红灯预测\upload_to_cloud.py').read()
for line in password.split('\n'):
    if 'password' in line and '=' in line:
        password = line.split('=')[1].strip().strip("'").strip('"')
        break

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password)

    # Check if CSV exists and get its modification time
    stdin, stdout, stderr = client.exec_command(
        'cd /root/red-light-prediction && ls -la flowchain_agent_features.csv 2>&1 && echo "---" && tail -20 /root/red-light-prediction/flowchain_agent_features.csv 2>&1'
    )
    print(stdout.read().decode('utf-8', errors='replace'))
    err = stderr.read().decode('utf-8', errors='replace')
    if err: print('STDERR:', err[:500])
    client.close()
except Exception as e:
    print(f'Error: {e}')
