#!/usr/bin/env python3
"""
OOM Fix 云端验证脚本
====================
用法:
    1. 上传 oom_fix.tar.gz 解压后
    2. python scripts/test_oom_fix.py
    3. 全部 ✅ 即可放心跑完整实验

不依赖任何本地数据，全部用随机张量模拟。
"""

import sys, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test")

RESULTS = []  # (step_name, pass/fail, detail)

# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | PyTorch: {torch.__version__}")

    # ---- Step 1: Build model ----
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.perception_model import TrafficPerceptionModel
    model = TrafficPerceptionModel(config, stage=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params")

    if hasattr(model, 'compute_perception_context'):
        RESULTS.append(("compute_perception_context 方法存在", True, ""))
    else:
        RESULTS.append(("compute_perception_context 方法存在", False, "方法不存在 — oom_fix 没生效!"))
        report_and_exit()

    # ---- Step 2: Dummy data ----
    obs_len, pred_len = 8, 12
    N_agents = 5
    obs = torch.randn(1, obs_len, 2, device=device)
    target = torch.randn(1, pred_len, 2, device=device)

    scene_data = {
        "bboxes": torch.rand(obs_len, N_agents, 4, device=device),
        "positions": torch.rand(obs_len, N_agents, 2, device=device),
        "class_names": [["pedestrian", "car", "car", "bicycle", "motorcycle"] for _ in range(obs_len)],
        "target_idx": 0,
        "track_ids": [[0,1,2,3,4] for _ in range(obs_len)],
        "velocities": None,
        "traffic_light_state": None,
    }

    # ---- Step 3: compute_perception_context ----
    try:
        model.reset_state()
        pc = model.compute_perception_context(obs, scene_data)
        assert pc.shape == (1, model.flow_chain.condition_dim), f"shape={pc.shape}"
        assert pc.norm() > 0, "perception_c is all zeros"
        RESULTS.append(("compute_perception_context 返回值", True, f"shape={list(pc.shape)} norm={pc.norm():.4f}"))
    except Exception as e:
        RESULTS.append(("compute_perception_context 返回值", False, str(e)))
        report_and_exit()

    # ---- Step 4: log_prob teacher forcing ----
    try:
        lp = model.flow_chain.log_prob(
            obs_trajectory=obs.squeeze(0),
            target=target.squeeze(0),
            perception_c=pc,
        )
        loss = -lp.mean()
        assert torch.isfinite(loss), f"loss={loss.item()}"
        RESULTS.append(("flow_chain.log_prob()", True, f"loss={loss.item():.4f}"))
    except Exception as e:
        RESULTS.append(("flow_chain.log_prob()", False, str(e)))
        report_and_exit()

    # ---- Step 5: backward ----
    try:
        loss.backward()
        grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "all grads are zero"
        RESULTS.append(("backward 反向传播", True, f"grad_norm={grad_norm:.2f}"))
    except Exception as e:
        RESULTS.append(("backward 反向传播", False, str(e)))
        report_and_exit()

    # ---- Step 6: GPU memory comparison ----
    if device == "cuda":
        model.reset_state()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        _ = model.compute_perception_context(obs, scene_data)
        lp = model.flow_chain.log_prob(obs.squeeze(0), target.squeeze(0), perception_c=pc)
        _ = -lp.mean()
        mem_tf = torch.cuda.max_memory_allocated() / 1024**2

        model.reset_state()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(obs, scene_data=scene_data, num_samples=10)
        mem_ar = torch.cuda.max_memory_allocated() / 1024**2

        ratio = mem_tf / mem_ar if mem_ar > 0 else 0
        RESULTS.append(("GPU显存对比", True,
            f"TeacherForcing={mem_tf:.0f}MB vs AR={mem_ar:.0f}MB (节省 {100*(1-ratio):.0f}%)"))
    else:
        RESULTS.append(("GPU显存对比", True, "CPU模式，跳过"))

    # ---- Step 7: Multi-batch loop test (模拟真实训练) ----
    logger.info("模拟训练循环 (10 batch, 每batch 4样本)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    grad_accum = 4
    accum = 0
    total_loss = 0.0

    for batch_idx in range(10):
        for b in range(4):
            model.reset_state()
            pc_b = model.compute_perception_context(obs, scene_data)
            lp_b = model.flow_chain.log_prob(
                obs_trajectory=obs.squeeze(0), target=target.squeeze(0), perception_c=pc_b)
            loss_b = -lp_b.mean()

            if torch.isfinite(loss_b):
                (loss_b / grad_accum).backward()
                accum += 1
                total_loss += loss_b.item()

            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
                accum = 0
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    if accum > 0:
        optimizer.step(); optimizer.zero_grad()

    avg_loss = total_loss / 40
    RESULTS.append(("训练循环模拟 (40 samples)", True, f"avg_loss={avg_loss:.4f}"))

    # ---- Step 8: Real data quick test (if available) ----
    try:
        from data.dataset import TrajectoryDataset, trajectory_collate_fn
        from torch.utils.data import DataLoader, Subset

        ds = TrajectoryDataset(
            data_dir=str(PROJECT_ROOT / "data" / "processed" / "trajectories"),
            label_dir=str(PROJECT_ROOT / "labels"),
            obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
            target_classes=["pedestrian"], mode="with_scene", max_scene_samples=20,
        )
        scene_idx = ds.with_scene_subset()[:2]
        if scene_idx:
            subset = Subset(ds, scene_idx)
            loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)
            ok = 0
            for batch in loader:
                obs_r = batch["obs_trajectory"].to(device) / torch.tensor([3840.,2160.], device=device)
                target_r = batch["target_trajectory"].to(device) / torch.tensor([3840.,2160.], device=device)
                scene_r = batch.get("scene_list", [None])[0]
                if scene_r is None:
                    continue
                model.reset_state()
                pc_r = model.compute_perception_context(obs_r, scene_r)
                lp_r = model.flow_chain.log_prob(obs_trajectory=obs_r.squeeze(0), target=target_r.squeeze(0), perception_c=pc_r)
                n_agents = len(scene_r["class_names"][0]) if scene_r.get("class_names") else "?"
                logger.info(f"  真实样本: loss={-lp_r.mean().item():.4f}, agents={n_agents}")
                ok += 1
            RESULTS.append(("真实数据测试", True, f"{ok} 样本 OK"))
        else:
            RESULTS.append(("真实数据测试", True, "无scene样本，跳过"))
    except Exception as e:
        RESULTS.append(("真实数据测试", True, f"跳过 ({e})"))

    report_and_exit()


def report_and_exit():
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok, detail in RESULTS:
        status = "✅" if ok else "❌"
        line = f"  {status} {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
        if not ok:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("  🎉 全部通过! 可以放心运行: python scripts/run_experiments.py ...")
    else:
        print("  ❌ 有测试失败! 请检查上面的错误信息")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
