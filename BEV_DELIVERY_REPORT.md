# 单目固定相机 → BEV 完整研究实现 · 交付报告

**任务定义**：Geometry-guided Weakly-supervised Monocular BEV
（Yang 风格 CVP/CVT 骨干 + Yan 风格跨视角循环一致性 + 时序一致性 + Homography+Detection 伪 BEV 弱监督）

本实现**完全新增**，未删除/简化/重构任何已有代码。已有 `src/detection/detector.py` 仅新增了一个非破坏性的 `bottom_center` 属性。

---

## 1. 模块总览

```
src/geometry/
  homography.py            单应矩阵 DLT(Hartley归一化)+RANSAC、正/逆变换、校验、往返误差
  coordinate.py            BEVGrid + ground⇄bev 坐标转换（X=横向 lateral，Y=纵向 longitudinal）
src/bev/
  pseudo_bev.py            检测底边中心→单应→伪 BEV 高斯热力图（命名始终 pseudo_bev，绝无 gt_bev）
  label_reader.py          读取已有 Ultralytics 追踪标签 .txt（复用检测+追踪，不重新检测）
  encoder.py               ResNet 相机编码器（torchvision weights= API）
  cvp.py                   Cycled View Projection（学习型 P_h/P_w 稠密视图投影 + 前向/反向/循环）
  cvt.py                   Cross-View Transformer（双向 cross-attention，非 concat）
  bev_decoder.py           BEV 特征 → 逐类热力图 logits
  camera_bev_projection.py 单应引导的可微相机⇄BEV 采样网格（grid_sample）
  monocular_bev.py         完整模型组装
  losses/
    pseudo_bev_loss.py     L_pseudo（focal / BCE / Dice）
    cycle_loss.py          L_cvp_cycle（特征循环）+ L_cycle（相机⇄BEV 循环）
    correspondence_loss.py L_corr（soft-argmax 目标位置 L2）
    temporal_loss.py       L_temporal（速度一致性）
    aggregate.py           compute_losses（全损失加权求和）
    presets.py             3 模式 + 消融 A0–A5 预设
  metrics.py               pseudo 监督评估指标 + 激活统计（防退化解）
  trainer.py               训练/验证循环（AMP/梯度累积/裁剪/NaN 守卫/防退化解检查）
  build.py                 config 驱动的构建助手
data/bev_dataset.py        BEV Dataset（帧→image/pseudo_bev/camera_mask）+ collate + 划分
configs/
  bev_proposed.yaml        完整模型（mode=proposed）
  bev_yang.yaml            Yang 基线（仅 pseudo 监督）
  bev_geometry.yaml        几何基线（无学习网络）
scripts/
  train_bev.py / evaluate_bev.py / inference_bev.py
tools/
  estimate_homography.py   单应标定（点对应/交互点击/planar_scale 占位）
  visualize_bev.py         BEV 热力图渲染
tests/
  test_geometry.py / test_pseudo_bev.py / test_models.py / test_losses.py / test_bev_dataset.py
```

## 2. 几何模块（STEP 2）

- `compute_homography(src, dst, method)`：`dlt`（Hartley 归一化 + SVD 零空间，最大重投影误差 ~1e-15）或 `ransac`（cv2）。
- `Homography`：`pixel_to_ground` / `ground_to_pixel` / `transform_points` / `validate_homography` / `round_trip_error` / `inverse` / `from_planar_scale` / 序列化。
- **非平凡解防护**：奇异/NaN/Inf 直接 assert；往返误差可度量。
- 测试：`tests/test_geometry.py` ✅（含往返、逆、RANSAC 分支、planar_scale）。

## 3. Detection → Pseudo-BEV（STEP 3）

- 投影点使用**底边中心** `((x1+x2)/2, y2)`（非框中心），见 `detector.DetectionResult.bottom_center` 与 `pseudo_bev.bottom_center_from_bbox`。
- 类名→BEV 通道：pedestrian→0，vehicle（bicycle/motorcycle/car/bus/truck）→1。
- 高斯热力图（σ 可配，峰值归一化 1，越界剔除）。
- **命名契约**：输出字段一律 `pseudo_bev`（`PseudoBEV.heatmap`），代码中不出现 `gt_bev`。
- 复用已有标签文件，**不重新检测**。
- 测试：`tests/test_pseudo_bev.py` ✅。

## 4. Encoder + CVP + CVT + Decoder（STEP 4）

- **CVP** 为学习型稠密视图投影（`P_h` 映射图像高度→BEV 纵向、`P_w` 映射宽度→BEV 横向，`einsum` 实现），**不是 Linear/reshape**；含反向投影与循环。
- **CVT** 为双向 cross-attention（BEV query 相机 / 相机 query BEV），**不是 concat+conv**。
- 测试：`tests/test_models.py` ✅（形状、有限性、循环可反传）。

## 5. 损失（STEP 5–10）

```
L = λ_pseudo·L_pseudo + λ_cvp·L_cvp_cycle + λ_cycle·L_cycle + λ_corr·L_corr + λ_temporal·L_temporal
```
所有权重均来自 config `loss:` 块（`compute_losses` 仅在权重>0 且字段存在时计算该项）。

- `L_pseudo`：focal（默认）/ BCE / Dice / 组合（弱监督目标 = pseudo_bev）。
- `L_cvp_cycle`：`||F_cam − F_cam_rec||₁`（CVP 特征循环）。
- `L_cycle`：`BCE + λ_dice·Dice( warp_back(pred_bev), pseudo_camera_mask )`（经 **H⁻¹** 可微采样，见 §6）。
- `L_corr`：soft-argmax 目标位置 L2（空场景 mask 掉）。
- `L_temporal`：`||v_pred − v_pseudo||₂`（连续帧速度一致性，仅两帧均有目标时计入）。
- 测试：`tests/test_losses.py` ✅（全部有限、`loss.backward()` 无 NaN、有非零梯度）。

## 6. 相机⇄BEV 可微循环（STEP 7–8）

`CameraBEVProjection` 用标定单应矩阵预计算两张 `grid_sample` 采样网格（固定相机只算一次，注册为 buffer）：

- `camera_to_bev`：BEV 每格 → 相机 mask 采样点；
- `bev_to_camera`：相机每像素 → BEV 采样点（H⁻¹）。

循环损失把 `pred_bev` 反投影回相机与**伪相机 mask** 比较，保证解码器与相机视角一致，全程无需 BEV GT。

## 7. 3 模式 + 消融 A0–A5（config 标志）

| 配置 | mode | 说明 |
|---|---|---|
| `bev_geometry.yaml` | geometry | 无学习网络，单应投影即输出（参考上界，仅评估） |
| `bev_yang.yaml` | yang | CVP/CVT 仅 L_pseudo 监督 |
| `bev_proposed.yaml` | proposed | 五项损失全开 |

消融（`ablation:` 字段，只把对应权重置零，不改图）：`a0`=完整，`a1`=去 L_cycle，`a2`=去 L_cvp_cycle，`a3`=去 L_corr，`a4`=去 L_temporal，`a5`=去 L_pseudo（纯循环+相关+时序）。

## 8. 数据与脚本（STEP 11–12）

- `data/bev_dataset.py`：逐帧 `image` + `pseudo_bev` + `camera_mask`，支持 `temporal` 连续帧；划分用 `split_ratio` 或显式 `train/val/test_videos`（不触碰已有轨迹数据划分）。
- `scripts/train_bev.py`（`--config --ablation --epochs --resume`）、`evaluate_bev.py`（测试集指标，vs pseudo_bev）、`inference_bev.py`（逐帧保存热力图 npz）。
- `tools/estimate_homography.py`（点对应/交互点击/planar_scale）、`tools/visualize_bev.py`。
- 测试：`tests/test_bev_dataset.py` ✅（presets、合成视频数据集、temporal、config 构建+前向）。

## 9. 防静默失败 / 防退化解

- 几何：奇异矩阵 / NaN / Inf 直接 assert；往返误差断言。
- 损失：`compute_losses` 与训练循环对 NaN/Inf 直接 `RuntimeError`；梯度有限性检查。
- 退化解：验证时输出 `pred_bev` 的 mean/std/min/max，`std < 1e-3` 时告警。

## 10. 内存策略（不降分辨率）

config 提供 `use_amp`（混合精度）、`grad_accum_steps`（梯度累积）、`grad_checkpointing`（梯度检查点）三个开关。**未**通过降低 BEV 分辨率或删减模块来规避 OOM。

## 11. 测试结果

```
ALL GEOMETRY TESTS PASSED
ALL PSEUDO-BEV TESTS PASSED
ALL MODEL SHAPE/FORWARD TESTS PASSED
ALL LOSS / CYCLE / MODEL TESTS PASSED
ALL BEV DATASET / PRESET / BUILD TESTS PASSED
```
另以合成视频端到端跑通 `BEVTrainer.train_epoch` / `validate`（CPU）。

## 12. 明确回答

> **是否真正使用了 BEV Ground Truth？**

**No.** 本项目没有任何 BEV 真值数据，也未伪造 BEV 真值。

监督信号类型为：

> **Geometry-based weak supervision（Homography + Detection 生成的 pseudo_bev）+ Cross-view Cycle consistency + Temporal consistency。**

单应矩阵仅作为「几何教师」产生**伪标签**（携带标定误差与检测误差），配合跨视角循环一致性与时序一致性共同约束 BEV 网络，全程不使用 BEV GT。
