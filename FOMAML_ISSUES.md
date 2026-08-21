# FOMAML v2 + FlowChain 元训练问题总结

## 背景
用 FOMAML v2 对 FlowChain 轨迹预测模型做 domain-adaptive 元学习。FlowChain = Transformer encoder-decoder + RealNVP flow（含内部 BatchNorm）。

## 当前状态
- **冻结 encoder（只训练 flow BN，16 参数）**：query loss 正常，训练可运行
- **不冻结 encoder（训练 attention + LayerNorm + flow BN，51K 参数）**：inner loop 正常，query loss 全部 NaN，outer loop 无有效梯度

## 已尝试的方案（全部失败）

| 方案 | 结果 |
|------|------|
| 1. eval 模式 inner loop + eval 模式 query loss | inner loop 正常，query loss NaN |
| 2. train 模式 inner loop（BN 用 batch stats）| inner loop 第一步就 NaN |
| 3. eval 模式 inner loop + train 模式 BN 校准 + eval 模式 query loss | 校准 pass 本身 NaN |
| 4. AdaBN（BN 内插值 batch/running stats，α=0.3）全程 eval 模式 | inner loop 正常，query loss 仍是 NaN |
| 5. AdaBN α=0.3 同时用于 inner loop + query loss | 同上 |
| 6. inner_lr=0.001, inner_steps=3, α=0.7 | 同上 |
| 7. 冻结 encoder（--freeze-encoder），只训练 flow BN | **可行**，但 encoder 无改编能力 |

## NaN 的精确位置
```
flow_chain_official.py line 355: Flow.log_prob()
  → self.forward(x, cond) → self.net(x, cond) → FlowSequential
  → LinearMaskedCoupling.forward() → t_net(ReLU) → 无上界的 translation t
  → u = x * exp(tanh(s)) + t  → u 含 INF
  → BatchNorm 无法修复 INF → base_dist.log_prob(u) → ValueError: invalid NaN values
```

NaN 链条：adapted encoder → decoder cross-attention → dist_args（16维flow条件）分布漂移 → coupling layer 的 **t_net**（Linear→ReLU→Linear→ReLU→Linear，最后 Linear 无激活）接收异常大的 dist_args → 输出极大的 t → u=INF → BN 归一化后仍是 INF。

## 核心瓶颈
1. **Flow 内部 coupling layer 的 t_net 无上界**：ReLU + 无激活的最后一个 Linear 层，输入稍大就会溢出到 INF
2. **Encoder 改编导致 dist_args 分布漂移**：即使 inner_lr=0.001、3步 SGD，encoder 权重改变仍然足以让 decoder 输出极端值
3. **BN 无法修复 INF**：AdaBN 插值解决了"分布不匹配"问题，但在 INF 面前无效
4. **dropout 在 train 模式下加剧不稳定**：train() 模式下 10% dropout 让情况更糟

## 可能的解决方向（尚未尝试）
1. **对 encoder 输出加 clamp/tanh**：在 encoder 输出后、decoder 前加 bounded activation
2. **对 dist_args 加 clamp**：在 `dist_args_proj` 输出后限制范围
3. **冻结 encoder 的 LayerNorm，只训练 attention**：LayerNorm 的缩放效应可能比 attention 更危险
4. **用 FOMAML 只改编 flow BN + encoder 最后一层 LayerNorm**：渐近式解冻
5. **在 coupling layer 的 t_net 最后加 Tanh**：从根本上限制 t 的输出范围
6. **用更高阶 MAML（create_graph=True）**：二阶梯度可能更稳定（MetaHTR 用的就是二阶）

## 代码位置
- 模型：`src/prediction/flow_chain_official.py`（BatchNorm L110-158, LinearMaskedCoupling L161-223, TransformerFlowChain.log_prob L738-784）
- 训练：`train_fomaml.py`（FOMAMLTrainer L329-580，compute_loss L301-322）
- AdaBN alpha 已实现在 BatchNorm.ada_alpha buffer
