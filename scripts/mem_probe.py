#!/usr/bin/env python3
"""
内存压力测试 — OurMethod 实验一全量评估
=======================================
在 AutoDL 云端运行，模拟真实训练场景，评估：
  1. 全量 scene 数据集加载后的 CPU 内存
  2. 模型参数量 + GPU 显存占用
  3. 不同 agent 数量的 forward+backward 峰值 GPU 显存
  4. 持续训练 30 步，检测内存泄漏趋势

用法:
    python scripts/mem_probe.py

输出可直接贴给其他 AI 判断 80GB CPU + 24GB GPU 是否够用。
"""

import sys, gc, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mem_probe")

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# psutil fallback (AutoDL 镜像可能没装)
# ---------------------------------------------------------------------------
def _get_cpu_rss_bytes():
    """Cross-platform RSS in bytes."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        # Linux fallback: read /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return 0


def cpu_gb():
    return _get_cpu_rss_bytes() / GB


def gpu_gb():
    return torch.cuda.max_memory_allocated() / GB


def gpu_now_gb():
    return torch.cuda.memory_allocated() / GB


# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_total = None
    try:
        import psutil
        cpu_total = psutil.virtual_memory().total / GB
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        cpu_total = int(line.split()[1]) / (1024 * 1024)
        except Exception:
            cpu_total = 80  # assume

    gpu_total = torch.cuda.get_device_properties(0).total_memory / GB if device == "cuda" else 0

    print()
    print("=" * 65)
    print("  OurMethod 内存压力测试 (实验一)")
    print("=" * 65)
    print(f"  PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")
    print(f"  CPU 总内存: {cpu_total:.0f} GB  |  GPU 显存: {gpu_total:.0f} GB")
    print()

    # ================================================================
    # [1/5] 加载全量 scene 数据集
    # ================================================================
    print("=" * 65)
    print("[1/5] 加载全量 scene 数据集...")
    print("=" * 65)

    from data.dataset import TrajectoryDataset

    mem0 = cpu_gb()
    ds_scene = TrajectoryDataset(
        data_dir="data/processed/trajectories",
        label_dir="labels",
        obs_len=8, pred_len=12, stride=8, min_trajectory_len=20,
        target_classes=["pedestrian"],
        mode="with_scene",
        max_scene_samples=10000,
    )
    scene_idx = ds_scene.with_scene_subset()
    mem1 = cpu_gb()

    print(f"  轨迹总样本数 : {len(ds_scene):,}")
    print(f"  有 scene 数据 : {len(scene_idx):,}")
    print(f"  CPU 内存增量 : {mem1 - mem0:+.1f} GB  ({mem0:.1f} → {mem1:.1f})")

    # ================================================================
    # [2/5] 构建模型
    # ================================================================
    print()
    print("=" * 65)
    print("[2/5] 构建 TrafficPerceptionModel (stage=2)...")
    print("=" * 65)

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    from src.perception_model import TrafficPerceptionModel

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem0 = cpu_gb()

    model = TrafficPerceptionModel(config, stage=2).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem1 = cpu_gb()
    gpu_alloc = torch.cuda.memory_allocated() / GB

    # 逐模块参数分解
    print(f"  总参数量     : {n_total:,}  (可训练: {n_train:,})")
    print(f"  GPU 显存占用 : {gpu_alloc:.2f} GB  (仅模型参数)")
    print(f"  CPU 内存增量 : {mem1 - mem0:+.1f} GB")
    comp_names = []
    comp_params = []
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n > 0:
            comp_names.append(name)
            comp_params.append(n)
    # 按参数量降序
    order = sorted(range(len(comp_params)), key=lambda i: -comp_params[i])
    for i in order:
        print(f"    {comp_names[i]:30s} {comp_params[i]:>10,}")

    # ================================================================
    # [3/5] 真实样本压测 — 不同 agent 数量
    # ================================================================
    print()
    print("=" * 65)
    print("[3/5] 真实样本 forward+backward (不同 agent 数)...")
    print("=" * 65)

    from torch.utils.data import DataLoader, Subset
    from data.dataset import trajectory_collate_fn

    NORM = torch.tensor([3840., 2160.])

    subset = Subset(ds_scene, scene_idx[:80])
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)

    # 收集不同 agent 数量的代表样本
    agent_samples = {}
    for batch in loader:
        scene = batch.get("scene_list", [None])[0]
        if scene is None:
            continue
        cn = scene.get("class_names", [])
        na = max(len(c) for c in cn) if cn else 0
        if na not in agent_samples:
            agent_samples[na] = batch
        if len(agent_samples) >= 6:
            break

    print(f"  {'Agents':<8} {'GPU峰值':>10} {'loss':>10}  {'说明'}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*20}")

    max_gpu_seen = 0.0
    for na in sorted(agent_samples.keys()):
        batch = agent_samples[na]
        obs = batch["obs_trajectory"].to(device) / NORM.to(device)
        target = batch["target_trajectory"].to(device) / NORM.to(device)
        scene = batch["scene_list"][0]

        scene_data = {
            "bboxes": scene["bboxes"].unsqueeze(0).to(device),
            "positions": scene["positions"].unsqueeze(0).to(device),
            "class_names": scene["class_names"],
            "track_ids": scene.get("track_ids", []),
            "target_idx": 0,
        }

        model.reset_state()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            perception_c = model.compute_perception_context(obs.squeeze(0), scene_data)
            lp = model.flow_chain.log_prob(
                obs_trajectory=obs.squeeze(0), target=target.squeeze(0),
                perception_c=perception_c)
            loss = -lp.mean()
            loss.backward()
            gp = gpu_gb()
            tag = ""
            if gp > max_gpu_seen:
                max_gpu_seen = gp
                tag = " ← 峰值"
            print(f"  {na:<8} {gp:>8.2f} GB {loss.item():>10.4f}  {tag}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {na:<8} {'OOM!':>10}  {'':>10}  ❌ 显存不足!")
            else:
                print(f"  {na:<8} {'ERROR':>10}  {'':>10}  {str(e)[:40]}")
            torch.cuda.empty_cache()

        model.zero_grad()
        del perception_c, lp, loss, scene_data
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # [4/5] 持续训练模拟 — 检测内存泄漏
    # ================================================================
    print()
    print("=" * 65)
    print("[4/5] 持续训练模拟 (30步, 检测内存泄漏)...")
    print("=" * 65)

    # 选一个典型 agent 数的样本
    typical = None
    for n in [5, 6, 7, 8, 4, 3, 9]:
        if n in agent_samples:
            typical = agent_samples[n]
            break
    if typical is None and agent_samples:
        typical = list(agent_samples.values())[0]

    if typical is None:
        print("  ⚠️ 无可用样本, 跳过")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mem_trace = []
        t0 = time.time()

        for step in range(30):
            obs = typical["obs_trajectory"].to(device) / NORM.to(device)
            target = typical["target_trajectory"].to(device) / NORM.to(device)
            scene = typical["scene_list"][0]
            scene_data = {
                "bboxes": scene["bboxes"].unsqueeze(0).to(device),
                "positions": scene["positions"].unsqueeze(0).to(device),
                "class_names": scene["class_names"],
                "track_ids": scene.get("track_ids", []),
                "target_idx": 0,
            }

            model.reset_state()
            optimizer.zero_grad()
            perception_c = model.compute_perception_context(obs.squeeze(0), scene_data)
            lp = model.flow_chain.log_prob(
                obs_trajectory=obs.squeeze(0), target=target.squeeze(0),
                perception_c=perception_c)
            loss = -lp.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            cpu_now = cpu_gb()
            gpu_now = gpu_now_gb()
            mem_trace.append((cpu_now, gpu_now))
            if step % 10 == 0:
                elapsed = time.time() - t0
                print(f"  step {step:2d}  CPU={cpu_now:.1f}G  GPU={gpu_now:.2f}G  "
                      f"loss={loss.item():.4f}  {elapsed:.1f}s")

            del perception_c, lp, loss, scene_data

        elapsed = time.time() - t0

        # 趋势
        c0, g0 = mem_trace[0]
        c1, g1 = mem_trace[-1]
        print(f"\n  30步耗时: {elapsed:.1f}s  (均步 {elapsed/30:.1f}s)")
        print(f"  CPU 漂移: {c0:.1f} → {c1:.1f} GB (Δ={c1-c0:+.2f} GB)")
        print(f"  GPU 漂移: {g0:.2f} → {g1:.2f} GB (Δ={g1-g0:+.2f} GB)")

        if (c1 - c0) > 1.0:
            print(f"  ⚠️  CPU 内存持续增长 (+{c1-c0:.2f}GB/30步), 可能泄漏!")
        else:
            print(f"  ✅ CPU 内存稳定")
        if abs(g1 - g0) < 0.3:
            print(f"  ✅ GPU 显存稳定")
        else:
            print(f"  ⚠️  GPU 显存波动较大")

    # ================================================================
    # [5/5] 综合评估
    # ================================================================
    print()
    print("=" * 65)
    print("[5/5] 最终评估")
    print("=" * 65)

    cpu_final = cpu_gb()
    gpu_final = gpu_now_gb()
    gpu_peak = max_gpu_seen if max_gpu_seen > 0 else gpu_final

    # AdamW 优化器状态 ≈ 2 × params (momentum + variance)
    optim_gpu = n_train * 8 / GB  # fp32 × 2 states
    grad_gpu = n_train * 4 / GB
    total_gpu_est = gpu_alloc + optim_gpu + grad_gpu + (gpu_peak - gpu_alloc)

    cpu_pct = cpu_final / cpu_total * 100
    gpu_peak_pct = gpu_peak / gpu_total * 100

    print(f"  模型参数     : {n_total:,}")
    print(f"  模型权重     : {gpu_alloc:.2f} GB")
    print(f"  优化器状态   : ~{optim_gpu:.2f} GB (AdamW)")
    print(f"  梯度         : ~{grad_gpu:.2f} GB")
    print(f"  前向激活峰值 : ~{gpu_peak - gpu_alloc:.2f} GB")
    print(f"  ─────────────────────────────")
    print(f"  GPU 峰值合计 : ~{total_gpu_est:.2f} GB  ({gpu_peak_pct:.0f}% of {gpu_total:.0f} GB)")
    print(f"  CPU 当前占用 : {cpu_final:.1f} GB  ({cpu_pct:.0f}% of {cpu_total:.0f} GB)")
    print()

    # 判定
    if cpu_pct < 60 and gpu_peak_pct < 60:
        verdict = "✅ 充裕 — 可以安全运行实验一 (10 epochs × ~10000条)"
    elif cpu_pct < 75 and gpu_peak_pct < 75:
        verdict = "✅ 可行 — 内存够用，建议关掉其他GPU进程"
    elif cpu_pct < 85 and gpu_peak_pct < 85:
        verdict = "⚠️ 偏紧 — 可运行但建议降 max_scene_samples 至 5000"
    elif gpu_peak_pct >= 90:
        verdict = "❌ GPU 显存不足 — 需降 max_agents 或启用 gradient checkpointing"
    else:
        verdict = "❌ CPU 内存不足 — 需减少 max_scene_samples"

    print(f"  判定: {verdict}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
