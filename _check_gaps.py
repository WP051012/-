"""Quick check of critical gaps on cloud."""
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=46863, username='root', password='2Lgv+Pw0rD+q', timeout=30)

print("=== 1. Dirs without traffic_lights.csv (first 10) ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/red-light-prediction && "
    "for d in data/processed/trajectories/*/; do "
    "  if [ ! -f $d/traffic_lights.csv ]; then "
    "    bn=$(echo $d | sed 's|.*/||;s|/||'); echo $bn; "
    "  fi; "
    "done | head -10")
print(stdout.read().decode())

print("=== 2. Remote src/prediction/ ===")
stdin, stdout, stderr = ssh.exec_command("ls /root/red-light-prediction/src/prediction/ 2>&1")
print(stdout.read().decode())

print("=== 3. Remote run_ourmethod* ===")
stdin, stdout, stderr = ssh.exec_command("ls /root/red-light-prediction/scripts/run_ourmethod* 2>&1")
print(stdout.read().decode())

print("=== 4. Remote eval_ourmethod* ===")
stdin, stdout, stderr = ssh.exec_command("ls /root/red-light-prediction/scripts/eval_ourmethod* 2>&1")
print(stdout.read().decode())

print("=== 5. Remote run_experiments.py ===")
stdin, stdout, stderr = ssh.exec_command("ls -la /root/red-light-prediction/scripts/run_experiments.py 2>&1")
print(stdout.read().decode())

print("=== 6. Dirs without violation_labels.csv (count) ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/red-light-prediction && "
    "total=$(ls -d data/processed/trajectories/*/ | wc -l); "
    "with_vl=$(find data/processed/trajectories -name violation_labels.csv | wc -l); "
    "echo total=$total without_vl=$((total - with_vl))")
print(stdout.read().decode())

print("=== 7. Check: do labels/ files match? ===")
stdin, stdout, stderr = ssh.exec_command("ls /root/red-light-prediction/labels/ 2>&1 | head -5; echo ---; wc -l /root/red-light-prediction/labels/*.csv 2>&1 | tail -5")
print(stdout.read().decode())

import os
local_labels = "C:/Users/wangj/Desktop/闯红灯预测/labels"
if os.path.exists(local_labels):
    print("Local labels:", os.listdir(local_labels)[:5])
else:
    print("Local labels: NO DIR")

print("=== 8. Critical scripts missing? ===")
critical = [
    'scripts/train_risk_regression.py',
    'scripts/run_experiments.py',
    'scripts/diagnose_features.py',
    'src/classification/risk_regression_head.py',
    'src/classification/crossing_probability.py',
    'src/classification/agent_centric_risk.py',
    'src/baselines/baseline_models.py',
    'data/dataset.py',
]
for f in critical:
    stdin, stdout, stderr = ssh.exec_command(f"[ -f /root/red-light-prediction/{f} ] && echo EXISTS: {f} || echo MISSING: {f}")
    result = stdout.read().decode().strip()
    if 'MISSING' in result:
        print(f"  {result}")

ssh.close()
print("Done.")
