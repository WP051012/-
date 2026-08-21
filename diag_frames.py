"""Diagnose frame ID mismatch between trajectory samples and precomputed npz."""
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
import numpy as np, json, os, glob, sys

BASE = "/root/red-light-prediction"
PRECOMPUTED = os.path.join(BASE, "data/precomputed")
TRAJ_DIR = os.path.join(BASE, "data/processed/trajectories")

# Load domain map to get the list of known videos
with open(os.path.join(BASE, "data/domains/domain_labels_int.json")) as f:
    domain_map = json.load(f)

# Get precomputed npz files (strip .npz)
pc_videos = set()
for f in glob.glob(os.path.join(PRECOMPUTED, "*.npz")):
    pc_videos.add(os.path.basename(f).replace(".npz", ""))

# Get trajectory directories
traj_videos = set()
for d in glob.glob(os.path.join(TRAJ_DIR, "*/")):
    traj_videos.add(os.path.basename(os.path.dirname(d)))

print(f"Precomputed npz count: {len(pc_videos)}")
print(f"Trajectory dir count:  {len(traj_videos)}")
print(f"Intersection:          {len(pc_videos & traj_videos)}")
print(f"Only in precomputed:   {len(pc_videos - traj_videos)}")
print(f"Only in trajectories:  {len(traj_videos - pc_videos)}")

# Pick 5 videos in the intersection
common = sorted(pc_videos & traj_videos)[:5]
print(f"\\n=== Analyzing {len(common)} common videos ===\\n")

for video in common[:3]:
    print(f"--- Video: {video[:70]}... ---")

    # 1. Trajectory data: sample frames
    traj_path = os.path.join(TRAJ_DIR, video, "trajectories.npz")
    traj = np.load(traj_path, allow_pickle=True)
    track_keys = list(traj.keys())

    # Find first pedestrian track
    ped_key = None
    for k in track_keys:
        v = traj[k].item()
        cn = v.get("class_name", "")
        if isinstance(cn, np.ndarray):
            cn = str(cn[0]) if len(cn) > 0 else ""
        if cn == "pedestrian":
            ped_key = k
            break

    if ped_key is None:
        print("  No pedestrian track found!")
        continue

    ped = traj[ped_key].item()
    frames = ped.get("frames", [])
    positions = ped.get("positions", [])

    print(f"  Pedestrian track: {ped_key}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Frame range: [{frames[0]}, {frames[-1]}]")
    print(f"  Sample frames (first 10): {frames[:10]}")
    print(f"  Position range: x=[{positions[:,0].min():.0f}, {positions[:,0].max():.0f}], y=[{positions[:,1].min():.0f}, {positions[:,1].max():.0f}]")

    # 2. Precomputed data: frame keys
    pc_path = os.path.join(PRECOMPUTED, f"{video}.npz")
    pc = np.load(pc_path, allow_pickle=True)
    pc_data = pc["data"].item()
    pc_keys = sorted(pc_data.keys(), key=lambda x: int(x))
    pc_keys_int = [int(k) for k in pc_keys]

    print(f"  Precomputed frame count: {len(pc_keys)}")
    print(f"  Precomputed key range: [{pc_keys_int[0]}, {pc_keys_int[-1]}]")
    print(f"  Precomputed sample keys: {pc_keys[:10]}")

    # 3. Overlap check
    traj_frame_set = set(int(f) for f in frames)
    pc_frame_set = set(int(k) for k in pc_keys)
    overlap = traj_frame_set & pc_frame_set
    print(f"  Overlap frames: {len(overlap)} / {len(traj_frame_set)}")

    # 4. Pick one overlapping frame, compare positions
    if overlap:
        overlap_frames = sorted(overlap)
        mid_f = overlap_frames[len(overlap_frames)//2]
        traj_idx = list(frames).index(mid_f) if mid_f in frames else -1

        pc_fd = pc_data[str(mid_f)]
        pc_positions = pc_fd["positions"]
        pc_class_names = list(pc_fd.get("class_names", []))

        if traj_idx >= 0:
            traj_pos = positions[traj_idx]
            print(f"\\n  Frame {mid_f}:")
            print(f"    Trajectory position: [{traj_pos[0]:.1f}, {traj_pos[1]:.1f}]")
            print(f"    Precomputed detections ({len(pc_positions)}):")
            for i, (pos, cn) in enumerate(zip(pc_positions, pc_class_names)):
                dist = np.linalg.norm(pos - traj_pos)
                marker = " <-- MATCH" if dist < 50 else ""
                print(f"      [{i}] {cn:15s} pos=[{pos[0]:.1f}, {pos[1]:.1f}]  dist={dist:.1f}{marker}")
        else:
            print(f"\\n  Frame {mid_f}: trajectory index not found (BUG)")

    # 5. Frame numbering comparison
    # Are trajectory frames 0-indexed within video or absolute?
    traj_f0 = int(frames[0])
    pc_k0 = pc_keys_int[0]
    offset = traj_f0 - pc_k0
    print(f"\\n  Frame offset (traj[0] - pc[0]): {offset}")
    print(f"  Trajectory frame range: [{frames[0]}, {frames[-1]}]")
    print(f"  PC key range:           [{pc_k0}, {pc_keys_int[-1]}]")

    # Check if relabeling fixes it
    if offset != 0:
        relabeled_overlap = set(int(f) - offset for f in frames) & pc_frame_set
        print(f"  Overlap AFTER offset correction ({offset}): {len(relabeled_overlap)} / {len(traj_frame_set)}")

    print()
'''

# Write script to instance and run
remote = '/root/red-light-prediction/diag_frames.py'
sftp = c.open_sftp()
# Write directly
import io
sftp.putfo(io.BytesIO(diag_script.encode()), remote)
sftp.close()

print(run(f"{PY} {remote}"))

c.close()
