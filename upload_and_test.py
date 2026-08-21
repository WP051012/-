"""Upload fixed eval_flowchain_domain.py to A1 and run test."""
import paramiko, time, sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'
LOCAL = r'C:\Users\wangj\Desktop\闯红灯预测\scripts\eval_flowchain_domain.py'
REMOTE = '/root/red-light-prediction/scripts/eval_flowchain_domain.py'
PY = '/root/miniconda3/bin/python'
WORKDIR = '/root/red-light-prediction'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)
print('Connected')

# Upload
sftp = c.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print('Upload OK')

# Check GPU
_, stdout, _ = c.exec_command(
    f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
print('GPU:', stdout.read().decode().strip())

# Run test
cmd = (f"cd {WORKDIR} && {PY} scripts/eval_flowchain_domain.py "
       "--train --epochs 3 --max-samples 2000 --num-mc 20 --save-ckpt checkpoints/flowchain_domain_test.pt")
print(f'\nRunning: {cmd}')
print('=' * 60)

# Non-blocking: stream output
channel = c.get_transport().open_session()
channel.exec_command(cmd)

try:
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode('utf-8', errors='replace')
            sys.stdout.write(data)
            sys.stdout.flush()
        if channel.recv_stderr_ready():
            err = channel.recv_stderr(4096).decode('utf-8', errors='replace')
            sys.stderr.write(err)
            sys.stderr.flush()
        if channel.exit_status_ready():
            break
        time.sleep(0.1)
except KeyboardInterrupt:
    print('\nInterrupted')
    channel.close()

exit_code = channel.recv_exit_status()
print(f'\nExit code: {exit_code}')
c.close()
