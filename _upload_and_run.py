"""Upload and run _compare_flowchain.py on cloud."""
import paramiko
import sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=46863, username='root', password='2Lgv+Pw0rD+q', timeout=30)

# Upload
sftp = ssh.open_sftp()
with open('C:/Users/wangj/Desktop/闯红灯预测/_compare_flowchain.py', 'rb') as fh:
    sftp.putfo(fh, '/root/red-light-prediction/_compare_flowchain.py')
sftp.close()
print("Uploaded", flush=True)

# Kill old processes
ssh.exec_command('pkill -f _compare_flowchain 2>/dev/null; echo KILLED')

# Launch new - use channel to avoid blocking
channel = ssh.get_transport().open_session()
channel.exec_command('cd /root/red-light-prediction && nohup /root/miniconda3/bin/python _compare_flowchain.py > logs/compare_flowchain_v2.log 2>&1 &')
exit_status = channel.recv_exit_status()
print(f"Launched (exit={exit_status})", flush=True)

# Verify
stdin, stdout, stderr = ssh.exec_command('ps aux | grep _compare_flowchain | grep -v grep')
out = stdout.read().decode().strip()
print(f"Process check: {out[:200] if out else 'not found'}", flush=True)

# Check log after a few seconds
import time
time.sleep(5)
stdin, stdout, stderr = ssh.exec_command('head -5 /root/red-light-prediction/logs/compare_flowchain_v2.log 2>/dev/null || echo "no log yet"')
print(f"Log start: {stdout.read().decode().strip()[:300]}", flush=True)

ssh.close()
print("Done", flush=True)
