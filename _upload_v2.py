"""Upload lightweight compare script and run on cloud."""
import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=46863, username='root', password='2Lgv+Pw0rD+q', timeout=15)

# Upload
sftp = ssh.open_sftp()
with open('C:/Users/wangj/Desktop/闯红灯预测/_compare_flowchain_v2.py', 'rb') as fh:
    sftp.putfo(fh, '/root/red-light-prediction/_compare_flowchain_v2.py')
sftp.close()
print('Uploaded', flush=True)

# Kill old
ssh.exec_command('pkill -f _compare_flowchain 2>/dev/null; sleep 1; echo ok')

# Launch
channel = ssh.get_transport().open_session()
channel.exec_command('cd /root/red-light-prediction && nohup /root/miniconda3/bin/python _compare_flowchain_v2.py > logs/compare_flowchain_v4.log 2>&1 &')
channel.recv_exit_status()
print('Launched', flush=True)

# Wait for loading and check
time.sleep(120)
stdin, stdout, stderr = ssh.exec_command('grep -E \"(===|P_cross|Violations at|Test samples|Done)\" /root/red-light-prediction/logs/compare_flowchain_v4.log 2>/dev/null')
print('RESULTS:', stdout.read().decode().strip())
stdin2, stdout2, stderr2 = ssh.exec_command('ps aux | grep \"python.*compare\" | grep -v grep')
print('PROC:', stdout2.read().decode().strip()[:200])
ssh.close()
