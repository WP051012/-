"""Deep dive: find frames where precomputed data HAS a pedestrian."""
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
        print(f"  [stderr]: {err[:300]}")
    return out

diag_script = '''
import numpy as np, os, glob
from collections import Counter

BASE = "/root/red-light-prediction"
PRECOMPUTED = os.path.join(BASE, "data/precomputed")
TRAJ_DIR = os.path.join(BASE, "data/processed/trajectories")

# Pick the second video from before (has frame overlap)
video = "ch01_00000000001000000_20260127075543_20260127075842_685001"

# --- Precomputed: find ALL class names across ALL frames ---
pc_path = os.path.join(PRECOMPUTED, f"{video}.npz")
pc = np.load(pc_path, allow_pickle=True)
pc_data = pc["data"].item()
pc_class_counts = Counter()
pc_ped_frames = []  # frames that HAVE pedestrian detections

for key, fd in pc_data.items():
    cns = list(fd.get("class_names", []))
    for cn in cns:
        pc_class_counts[cn] += 1
    if "pedestrian" in cns:
        pc_ped_frames.append(int(key))

pc_ped_frames.sort()
print(f"=== Video: {video[:70]}... ===")
print(f"Precomputed class counts: {dict(pc_class_counts)}")
print(f"Frames with pedestrian: {len(pc_ped_frames)} / {len(pc_data)}")
if pc_ped_frames:
    print(f"  Range: [{pc_ped_frames[0]}, {pc_ped_frames[-1]}]")
    print(f"  Sample: {pc_ped_frames[:20]}")

# --- Trajectory: find pedestrian tracks ---
traj_path = os.path.join(TRAJ_DIR, video, "trajectories.npz")
traj = np.load(traj_path, allow_pickle=True)
print(f"\\nTrajectory track keys: {list(traj.keys())[:10]}")

ped_tracks = {}
for k in traj.keys():
    v = traj[k].item()
    cn = v.get("class_name", "")
    if isinstance(cn, np.ndarray):
        cn = str(cn[0]) if len(cn) > 0 else ""
    if cn == "pedestrian":
        frames = v.get("frames", [])
        positions = v.get("positions", [])
        ped_tracks[k] = {"frames": frames, "positions": positions}

print(f"Pedestrian tracks: {len(ped_tracks)}")
for tk, tv in ped_tracks.items():
    frames = tv["frames"]
    pos = tv["positions"]
    ped_overlap = set(int(f) for f in frames) & set(pc_ped_frames)
    print(f"  {tk}: {len(frames)} frames, range=[{frames[0]},{frames[-1]}], "
          f"frames_with_ped_in_pc={len(ped_overlap)}")

# --- Find matched frames: both have pedestrian ---
if ped_tracks and pc_ped_frames:
    pc_ped_set = set(pc_ped_frames)
    for tk, tv in ped_tracks.items():
        frames = tv["frames"]
        positions = tv["positions"]
        matched = []
        for i, f in enumerate(frames):
            fi = int(f)
            if fi in pc_ped_set:
                # Check position
                fd = pc_data[str(fi)]
                pc_pos = fd["positions"]
                pc_cns = list(fd.get("class_names", []))
                ped_indices = [j for j, cn in enumerate(pc_cns) if cn == "pedestrian"]
                for pi in ped_indices:
                    dist = np.linalg.norm(pc_pos[pi] - positions[i])
                    matched.append((fi, positions[i], pc_pos[pi], dist))

        print(f"\\n  {tk}: {len(matched)} matched frames (both have pedestrian)")
        if matched:
            print(f"  First 10 matches:")
            for fi, tpos, ppos, dist in matched[:10]:
                print(f"    Frame {fi}: traj=[{tpos[0]:.1f}, {tpos[1]:.1f}]  "
                      f"det=[{ppos[0]:.1f}, {ppos[1]:.1f}]  dist={dist:.1f}")
            # Stats
            dists = [m[3] for m in matched]
            print(f"  Distance stats: min={min(dists):.1f}  max={max(dists):.1f}  "
                  f"mean={np.mean(dists):.1f}  median={np.median(dists):.1f}")
'''

remote = '/root/red-light-prediction/diag_frames2.py'
sftp = c.open_sftp()
import io
sftp.putfo(io.BytesIO(diag_script.encode()), remote)
sftp.close()

print(run(f"{PY} {remote}"))
c.close()
