"""Check if label .txt files have pedestrian detections matching trajectory frames."""
import paramiko

HOST = 'connect.cqa1.seetacloud.com'
PORT = 44037
USER = 'root'
PW = 'qaaPg/iZDISX'
PY = '/root/miniconda3/bin/python'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15,
          look_for_keys=False, allow_agent=False)

def run(cmd):
    _, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        print(f"  [stderr]: {err[:200]}")
    return out

diag_script = '''
import numpy as np, os
from collections import Counter

BASE = "/root/red-light-prediction"

# Same video as diag2: pedestrian_6 has frames 1-424
video = "ch01_00000000001000000_20260127075543_20260127075842_685001"

# Check label file
label_path = os.path.join(BASE, "labels", video + ".txt")
if not os.path.exists(label_path):
    # Try other extensions
    for f in os.listdir(os.path.join(BASE, "labels")):
        if video[:60] in f:
            label_path = os.path.join(BASE, "labels", f)
            print(f"Found: {f}")
            break

print(f"Label file: {label_path}")
print(f"Exists: {os.path.exists(label_path)}")

if os.path.exists(label_path):
    # Read a few lines
    with open(label_path) as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")

    # Parse: frame_id, track_id, class_id, xc, yc, w, h, ...
    frame_classes = Counter()
    ped_frames = set()
    sample_lines = []

    for line in lines[:50000]:  # first 50k lines
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        frame_id = int(parts[0])
        class_id = int(parts[2]) if len(parts) > 2 else -1
        frame_classes[frame_id] += 1

        # class_id mapping: 0=pedestrian typically in COCO
        if class_id == 0:
            ped_frames.add(frame_id)
            if len(sample_lines) < 5:
                sample_lines.append(line.strip())

    print(f"\\nUnique frames: {len(frame_classes)}")
    if frame_classes:
        frames_sorted = sorted(frame_classes.keys())
        print(f"Frame range: [{frames_sorted[0]}, {frames_sorted[-1]}]")

    print(f"Frames with pedestrian (class_id=0): {len(ped_frames)}")
    if ped_frames:
        ped_sorted = sorted(ped_frames)
        print(f"  Range: [{ped_sorted[0]}, {ped_sorted[-1]}]")
        print(f"  Sample: {ped_sorted[:10]}")

    print(f"\\nSample pedestrian lines:")
    for sl in sample_lines:
        print(f"  {sl}")

    # Check overlap with trajectory pedestrian frames [1, 424]
    traj_ped_frames = set(range(1, 425))
    overlap = ped_frames & traj_ped_frames
    print(f"\\nOverlap with trajectory frames [1,424]: {len(overlap)}")
    if overlap:
        ov_sorted = sorted(overlap)
        print(f"  Range: [{ov_sorted[0]}, {ov_sorted[-1]}]")
        print(f"  First 10: {ov_sorted[:10]}")
else:
    print("Label file not found!")
'''

remote = '/root/red-light-prediction/check_labels.py'
sftp = c.open_sftp()
import io
sftp.putfo(io.BytesIO(diag_script.encode()), remote)
sftp.close()

print(run(f"{PY} {remote}"))
c.close()
