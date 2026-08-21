"""Upload eval_flowchain_domain.py to instance A1."""
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'
LOCAL = r'C:\Users\wangj\Desktop\闯红灯预测\scripts\eval_flowchain_domain.py'
REMOTE = '/root/red-light-prediction/scripts/eval_flowchain_domain.py'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
print('Connected')

sftp = c.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print('Upload OK')

_, stdout, stderr = c.exec_command('wc -l ' + REMOTE)
print('Remote lines:', stdout.read().decode().strip())
c.close()
print('Done')
