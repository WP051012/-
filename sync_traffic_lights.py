"""Sync traffic_lights.csv & traffic_light_rois.json to cloud.

Run after cloud instance is started:
    python sync_traffic_lights.py
"""
import paramiko
import os
import sys
from pathlib import Path

HOST = 'connect.cqa1.seetacloud.com'
PORT = 46863
USER = 'root'
PASSWORD = '2Lgv+Pw0rD+q'
REMOTE_BASE = '/root/red-light-prediction'
LOCAL_BASE = Path(r'C:\Users\wangj\Desktop\闯红灯预测')
DATA_DIR = LOCAL_BASE / 'data' / 'processed' / 'trajectories'

FILES_TO_SYNC = ['traffic_lights.csv', 'traffic_light_rois.json']

def main():
    # Count local files
    local_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Local video directories: {len(local_dirs)}")

    # Count which have the target files
    local_stats = {f: 0 for f in FILES_TO_SYNC}
    for d in local_dirs:
        for f in FILES_TO_SYNC:
            if (d / f).exists():
                local_stats[f] += 1
    for f, count in local_stats.items():
        print(f"  Local dirs with {f}: {count}/{len(local_dirs)}")

    # Connect
    print(f"\nConnecting to {HOST}:{PORT}...")
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print("Connected.")

    # Check remote
    remote_data_dir = f"{REMOTE_BASE}/data/processed/trajectories"
    try:
        remote_dirs = sftp.listdir(remote_data_dir)
        print(f"Remote video directories: {len(remote_dirs)}")
    except FileNotFoundError:
        print(f"Remote data dir not found: {remote_data_dir}")
        sftp.close()
        transport.close()
        return

    # Sync each file
    for fname in FILES_TO_SYNC:
        print(f"\n--- Syncing {fname} ---")
        uploaded = 0
        skipped = 0
        for d in sorted(local_dirs):
            local_file = d / fname
            if not local_file.exists():
                continue
            remote_file = f"{remote_data_dir}/{d.name}/{fname}"
            try:
                # Check if already exists on remote
                try:
                    sftp.stat(remote_file)
                    skipped += 1
                    continue
                except FileNotFoundError:
                    pass

                # Upload
                sftp.put(str(local_file), remote_file)
                uploaded += 1
                if uploaded % 100 == 0:
                    print(f"  Uploaded {uploaded}...")
            except Exception as e:
                # Try creating remote directory first
                remote_dir_video = f"{remote_data_dir}/{d.name}"
                try:
                    sftp.stat(remote_dir_video)
                except FileNotFoundError:
                    try:
                        sftp.mkdir(remote_dir_video)
                    except Exception:
                        pass
                try:
                    sftp.put(str(local_file), remote_file)
                    uploaded += 1
                except Exception as e2:
                    print(f"  FAILED {d.name}/{fname}: {e2}")

        print(f"  {fname}: uploaded={uploaded}, skipped={skipped}")

    sftp.close()
    transport.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
