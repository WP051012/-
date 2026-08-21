"""Comprehensive comparison of local vs cloud project files."""
import paramiko
import os
from pathlib import Path

HOST = 'connect.cqa1.seetacloud.com'
PORT = 46863
USER = 'root'
PASSWORD = '2Lgv+Pw0rD+q'
REMOTE_BASE = '/root/red-light-prediction'
LOCAL_BASE = Path(r'C:\Users\wangj\Desktop\闯红灯预测')

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = ssh.open_sftp()

    # =========================================================
    # 1. Compare source code / config files (not data, not checkpoints)
    # =========================================================
    print("=" * 70)
    print("1. SOURCE CODE FILES")
    print("=" * 70)

    local_src = set()
    for root, dirs, files in os.walk(str(LOCAL_BASE)):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'data', 'checkpoints',
                                                  'tf-logs', 'debug_frames', '参考文献',
                                                  'autodl-tmp', 'autodl-pub')]
        for f in files:
            if f.endswith(('.py', '.yaml', '.yml', '.json', '.txt', '.md', '.sh', '.csv',
                          '.pt', '.tar.gz', '.zip', '.docx')):
                rel = os.path.relpath(os.path.join(root, f), str(LOCAL_BASE)).replace('\\', '/')
                local_src.add(rel)

    stdin, stdout, stderr = ssh.exec_command(
        "cd /root/red-light-prediction && find . -maxdepth 5 -type f "
        "\\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' "
        "-o -name '*.txt' -o -name '*.md' -o -name '*.sh' -o -name '*.csv' "
        "-o -name '*.pt' -o -name '*.tar.gz' -o -name '*.zip' -o -name '*.docx' \\) "
        "! -path '*/data/*' ! -path '*/__pycache__/*' ! -path '*/.git/*' "
        "! -path '*/checkpoints/*' ! -path '*/tf-logs/*' ! -path '*/debug_frames/*' "
        "! -path '*/autodl-*' ! -path '*/flow_chain*' 2>/dev/null | sort")
    remote_src = set(stdout.read().decode().strip().split('\n'))
    remote_src = {p.lstrip('./') for p in remote_src if p}

    only_local = local_src - remote_src
    only_remote = remote_src - local_src
    common = local_src & remote_src

    print(f"  Local files:  {len(local_src)}")
    print(f"  Remote files: {len(remote_src)}")
    print(f"  Common:       {len(common)}")

    if only_local:
        print(f"\n  ONLY LOCAL ({len(only_local)}):")
        for f in sorted(only_local):
            print(f"    MISSING on cloud: {f}")

    if only_remote:
        print(f"\n  ONLY REMOTE ({len(only_remote)}):")
        for f in sorted(only_remote):
            print(f"    Only on cloud:    {f}")

    # =========================================================
    # 2. Compare data/trajectories directories
    # =========================================================
    print("\n" + "=" * 70)
    print("2. DATA: traffic_lights.csv coverage")
    print("=" * 70)

    stdin, stdout, stderr = ssh.exec_command(
        "cd /root/red-light-prediction && "
        "total=$(ls -d data/processed/trajectories/*/ 2>/dev/null | wc -l); "
        "with_tl=$(find data/processed/trajectories -name 'traffic_lights.csv' 2>/dev/null | wc -l); "
        "with_roi=$(find data/processed/trajectories -name 'traffic_light_rois.json' 2>/dev/null | wc -l); "
        "with_vl=$(find data/processed/trajectories -name 'violation_labels.csv' 2>/dev/null | wc -l); "
        "echo \"total_dirs=$total\"; echo \"with_tl=$with_tl\"; echo \"with_roi=$with_roi\"; echo \"with_vl=$with_vl\"")
    out = stdout.read().decode()
    for line in out.strip().split('\n'):
        print(f"  {line}")

    # =========================================================
    # 3. Check data file types per directory
    # =========================================================
    print("\n" + "=" * 70)
    print("3. DATA: file types in trajectory directories (sample)")
    print("=" * 70)

    stdin, stdout, stderr = ssh.exec_command(
        "cd /root/red-light-prediction && "
        "ls data/processed/trajectories/ | head -3 | while read d; do "
        "echo \"--- \$d ---\"; ls data/processed/trajectories/\"\$d\"/ 2>/dev/null; done")
    print(stdout.read().decode())

    # =========================================================
    # 4. Check key directories exist
    # =========================================================
    print("=" * 70)
    print("4. KEY DIRECTORIES")
    print("=" * 70)
    for d in ['checkpoints', 'configs', 'src', 'scripts', 'data/processed',
              'data/processed/trajectories', 'data/annotations', 'labels']:
        stdin, stdout, stderr = ssh.exec_command(
            f"[ -d /root/red-light-prediction/{d} ] && echo 'EXISTS: {d}' || echo 'MISSING: {d}'")
        print(f"  {stdout.read().decode().strip()}")

    # =========================================================
    # 5. Check labels directory
    # =========================================================
    print("\n" + "=" * 70)
    print("5. LABELS")
    print("=" * 70)
    stdin, stdout, stderr = ssh.exec_command(
        "ls /root/red-light-prediction/labels/ 2>/dev/null || echo 'No labels dir'")
    print(f"  {stdout.read().decode().strip()}")

    # =========================================================
    # 6. Missing violation_labels.csv?
    # =========================================================
    print("\n" + "=" * 70)
    print("6. violation_labels.csv coverage")
    print("=" * 70)

    # Local count
    local_vl = 0
    for root, dirs, files in os.walk(str(LOCAL_BASE / 'data' / 'processed' / 'trajectories')):
        if 'violation_labels.csv' in files:
            local_vl += 1
    print(f"  Local dirs with violation_labels.csv: {local_vl}")

    sftp.close()
    ssh.close()

if __name__ == '__main__':
    main()
