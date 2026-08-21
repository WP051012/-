#!/usr/bin/env python3
"""Segmented training for OurMethod — 80GB container limit"""
import os, sys

BASE = "/root/miniconda3/bin/python"
SCRIPT = "scripts/run_experiments.py"
COMMON = "--config configs/default.yaml --processed-dir data/processed/trajectories --label-dir labels/"
TOTAL_EPOCHS = 10
SEGMENT = 2
CKPT = "checkpoints/perception_segment.pt"

print("=" * 60)
print("Phase 1: Trajectory Training")
msg = "Segments: {0}x{1} epochs = {2} total".format(TOTAL_EPOCHS // SEGMENT, SEGMENT, TOTAL_EPOCHS)
print(msg)
print("=" * 60)

for seg in range(0, TOTAL_EPOCHS, SEGMENT):
    resume = "--resume " + CKPT if seg > 0 else ""
    cmd = "{0} {1} {2} --exp trajectory --skip social-lstm,stgcnn,transformer,rnn,flowchain --epochs {3} --segment {4} {5}".format(
        BASE, SCRIPT, COMMON, TOTAL_EPOCHS, SEGMENT, resume)

    seg_num = seg // SEGMENT + 1
    info = "Segment {0}: epochs {1}->{2}".format(seg_num, seg, seg + SEGMENT)
    print(info)
    print("Command: " + cmd)
    rc = os.system(cmd)
    rc_val = rc >> 8 if rc > 255 else rc
    if rc_val != 0:
        print("WARNING: segment exit code {0}".format(rc_val))
    done_msg = "Segment {0} done.".format(seg_num)
    print(done_msg)

# Copy checkpoint for classification
os.system("cp " + CKPT + " checkpoints/ablation_fullmodel.pt 2>/dev/null")
print("Checkpoint copied to ablation_fullmodel.pt")

# Phase 2
print("=" * 60)
print("Phase 2: Classification")
print("=" * 60)
cmd2 = "{0} {1} {2} --exp classification --epochs 10 --segment 0".format(BASE, SCRIPT, COMMON)
rc = os.system(cmd2)
print("Phase 2 exit: {0}".format(rc))
print("Done!")
