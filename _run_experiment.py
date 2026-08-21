"""Re-run two-stage experiment with traffic_lights.csv data on cloud."""
import paramiko, time, sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 46863
USER = 'root'
PWD = '2Lgv+Pw0rD+q'

# Build the training command
cmd = (
    'cd /root/red-light-prediction && '
    '/root/miniconda3/bin/python scripts/train_risk_regression.py '
    '--config configs/default.yaml '
    '--processed-dir data/processed/trajectories '
    '--label-dir labels/ '
    '--checkpoint checkpoints/flowchain_best.pt '
    '--save-path checkpoints/risk_regression_head_flowchain_v2.pt '
    '--epochs 30 '
    '--batch-size 32 '
    '--lr 1e-3 '
    '2>&1'
)

print("Starting two-stage training with real traffic light data...")
print(f"Command: {cmd[:150]}...")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

# Run with long timeout - training takes ~10-15 min
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=1800)

# Stream output
for line in stdout:
    print(line.strip())

err_output = stderr.read().decode()
if err_output:
    print("STDERR:", err_output[:2000])

ssh.close()
