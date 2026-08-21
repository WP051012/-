"""Inspect a FOMAML checkpoint to diagnose 'Modulation: no' and hyperparams.

Usage:
    python scripts/inspect_fomaml_ckpt.py --checkpoint checkpoints/fomaml_v2/best_fomaml.pt

Prints:
  - whether modulation_net / domain_conditions are present (root cause of "Modulation: no")
  - the FOMAML hyperparams stored in the checkpoint config
  - the trainable meta-param shapes
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_fomaml.pt")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top-level keys: {list(ckpt.keys())}")
    print()

    # ── Modulation ──
    has_mod = "modulation_net" in ckpt
    has_dom = "domain_conditions" in ckpt
    print(f"modulation_net    : {'PRESENT' if has_mod else 'MISSING  <- Modulation: no 的根因'}")
    print(f"domain_conditions : {'PRESENT' if has_dom else 'MISSING'}")
    if has_dom:
        dom = ckpt["domain_conditions"]
        print(f"  domains: {sorted(int(k) for k in dom.keys())}")
    if has_mod:
        sd = ckpt["modulation_net"]
        n = sum(v.numel() for v in sd.values())
        print(f"  modulation params: {n:,}")
        print(f"  keys (first 8): {list(sd.keys())[:8]}")
    print()

    # ── Config / hyperparams ──
    cfg = ckpt.get("config", {})
    print("config (训练时用到的 FOMAML 超参):")
    for k in ["inner_lr", "inner_steps", "outer_lr", "ade_weight", "batch_size",
              "use_adapter", "ada_alpha", "lambda_feat", "max_delta_norm",
              "modulation_lr", "epochs"]:
        print(f"  {k:16s} = {cfg.get(k, '<not set>')}")
    print()

    # ── Trainable meta-params ──
    tp = ckpt.get("trainable_params", {})
    if tp:
        n = sum(v.numel() for v in tp.values())
        print(f"trainable_params ({n:,} params):")
        for k, v in tp.items():
            print(f"  {k:24s} {tuple(v.shape)}")
    print()

    # ── Diagnosis ──
    print("Diagnosis:")
    if not has_mod:
        print("  -> checkpoint 里没有 modulation_net。原因二选一:")
        print("       1) 训练时加了 --no-modulation")
        print("       2) data/gat_conditions.pt 不存在 -> condition_map=None -> mod_net=None")
        print("     修复: 确认 gat_conditions.pt 存在，且重训时去掉 --no-modulation")
    else:
        print("  -> modulation 正常，eval 会显示 'Modulation: yes'")


if __name__ == "__main__":
    main()
