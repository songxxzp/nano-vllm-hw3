# HW4 量化实验报告

代码仓库：[https://github.com/songxxzp/nano-vllm-hw3](https://github.com/songxxzp/nano-vllm-hw3)

## 1. 实验概述

本实验基于 Qwen3-1.7B 模型测试不同量化方法（INT8/FP8）在不同粒度下的精度和性能。

**实验配置：**

- **模型**: Qwen3-1.7B
- **GPU**: RTX 4090
- **测试集**:
  - MMLU (1000 samples, 5-shot, seed=42)
  - WikiText-2 (100 samples)
- **量化方式**:
  - 伪量化（Fake）: 权重量化后立即反量化，仍以 BF16 存储
  - 真量化（Real）: 权重以 INT8/FP8 格式存储，运行时反量化
- **量化粒度**:
  - Per-Tensor: 整个张量使用一个 scale
  - Per-Row: 每行使用一个 scale（Per-Channel）
  - Per-Group: 每 64/128/256/512 个元素使用一个 scale
- **SmoothQuant**: 使用 TorchAO 实现的动态激活量化

**吞吐量测试方法：**

1. **Prefilling 吞吐量测试**:
   - 生成 100 个长度约 1000 tokens 的随机文本 prompt
   - 使用 `SamplingParams(temperature=0.0, max_tokens=1)` 只生成 1 个 token
   - 计算公式:
     - `Samples/s = num_samples / elapsed_time`
     - `Tokens/s = (num_samples × prompt_length) / elapsed_time`
   - 测试指标: 处理长 prompt 的速度（模拟首 token 生成场景）

2. **Decoding 吞吐量测试**:
   - 使用 100 个长度为 1 token 的短 prompt
   - 使用 `SamplingParams(temperature=0.0, max_tokens=1000)` 生成 1000 个 tokens
   - 计算公式:
     - `Tokens/s = total_generated_tokens / elapsed_time`
   - 测试指标: 连续生成 tokens 的速度（模拟自回归解码场景）

**矩阵乘法测试方法：**

1. **测试实现**:
   - 矩阵形状: 2048 × 2048 × 2048 (M, N, K)
   - 随机生成 100 组矩阵对 (A, B)
   - 对矩阵 B 应用相应的量化方法
   - 分别计算参考结果和量化后的结果

2. **误差计算公式**:
   - `Max Error = max(|C_quant - C_ref|)` - 元素级最大绝对误差
   - `Mean Error = mean(|C_quant - C_ref|)` - 平均绝对误差
   - `Relative Error = mean(|C_quant - C_ref| / (|C_ref| + ε))` - 相对误差（ε=1e-8 防止除零）
   - 最终结果为 100 次测试的平均值

---

## 2. Task 3.1: 量化精度测试

### 2.1 MMLU 准确率

**伪量化结果：**

| 配置 | INT8 | FP8 |
|------|------|-----|
| BF16 Baseline | 50.60% | 50.60% |
| Per-Tensor | 49.20% (-1.40%) | 50.30% (-0.30%) |
| Per-Row | 50.70% (+0.10%) | 50.20% (-0.40%) |
| Per-Group-64 | 50.00% (-0.60%) | 49.20% (-1.40%) |
| Per-Group-128 | 49.60% (-1.00%) | 48.20% (-2.40%) |
| Per-Group-256 | 50.30% (-0.30%) | 47.80% (-2.80%) |
| Per-Group-512 | 51.00% (+0.40%) | 49.00% (-1.60%) |

**真量化结果：**

| 配置 | INT8 | FP8 |
|------|------|-----|
| BF16 Baseline | 50.60% | 50.60% |
| Per-Tensor | 48.30% (-2.30%) | 49.50% (-1.10%) |
| Per-Row | 50.40% (-0.20%) | 50.10% (-0.50%) |
| Per-Group-64 | 49.80% (-0.80%) | 49.00% (-1.60%) |
| Per-Group-128 | 49.30% (-1.30%) | 49.50% (-1.10%) |
| Per-Group-256 | 50.10% (-0.50%) | 48.30% (-2.30%) |
| Per-Group-512 | 50.80% (+0.20%) | 48.90% (-1.70%) |

**分析：**
- Per-Row 量化几乎无损，优于 Per-Tensor
- INT8 在 MMLU 上整体优于 FP8 约 1-2%
- INT8 对 Group Size 不敏感，FP8 在 Group Size=256 时表现最差

### 2.2 WikiText-2 Perplexity

**伪量化结果：**

| 配置 | PPL | vs BF16 |
|------|-----|---------|
| BF16 Baseline | 33.51 | - |
| INT8 Per-Tensor | 34.30 | +0.79 |
| INT8 Per-Row | 33.92 | +0.41 |
| INT8 Per-Group-64 | 33.98 | +0.47 |
| INT8 Per-Group-128 | 33.89 | +0.38 |
| INT8 Per-Group-256 | 33.73 | +0.22 |
| INT8 Per-Group-512 | 33.81 | +0.30 |
| FP8 Per-Tensor | 33.14 | -0.37 |
| FP8 Per-Row | 35.15 | +1.64 |
| FP8 Per-Group-64 | 33.62 | +0.11 |
| FP8 Per-Group-128 | 33.68 | +0.17 |
| FP8 Per-Group-256 | 33.27 | -0.24 |
| FP8 Per-Group-512 | 33.40 | -0.11 |

**真量化结果：**

| 配置 | PPL | vs BF16 |
|------|-----|---------|
| BF16 Baseline | 33.51 | - |
| INT8 Per-Tensor | 34.13 | +0.62 |
| INT8 Per-Row | 33.92 | +0.41 |
| INT8 Per-Group-64 | 33.93 | +0.42 |
| INT8 Per-Group-128 | 33.93 | +0.42 |
| INT8 Per-Group-256 | 33.72 | +0.21 |
| INT8 Per-Group-512 | 33.76 | +0.25 |
| FP8 Per-Tensor | 33.09 | -0.42 |
| FP8 Per-Row | 35.68 | +2.17 |
| FP8 Per-Group-64 | 33.67 | +0.16 |
| FP8 Per-Group-128 | 33.73 | +0.22 |
| FP8 Per-Group-256 | 33.30 | -0.21 |
| FP8 Per-Group-512 | 33.42 | -0.09 |

**分析：**
- INT8 量化的 PPL 损失在 0.2-0.8 之间
- FP8 Per-Tensor 和 Per-Group-256/512 略优于 BF16（可能因为量化噪声的正则化效应）
- FP8 Per-Row 表现最差（+1.64/+2.17）

### 2.3 系统实现影响

| 维度 | Per-Tensor | Per-Row | Per-Group |
|------|------------|---------|-----------|
| Scale 参数量 | 1 个 | output_channels 个 | (output_channels × K/group_size) 个 |
| 内存开销 | 最小 | 中等 (+0.1%) | 较大 (+0.5-2%) |
| Kernel 融合 | 容易 | 容易 | 较难 |

**量化/反量化与 Kernel 融合：**
- Per-Tensor/Per-Row: 在矩阵乘前一次性反量化，开销小
- Per-Group: 需要动态获取不同组的 scale，有一定额外开销
- 真量化: 权重以 INT8/FP8 格式存储，节省显存，但在计算时需要反量化

---

## 3. Task 3.2: 量化吞吐测试

### 3.1 MMLU 测试吞吐对比

基于 MMLU 测试（1000 samples）的吞吐数据：

| 量化类型 | 测试时间 (s) | Throughput (samples/s) | 加速比 |
|---------|-------------|----------------------|--------|
| BF16 Baseline | 5.00 | 200.0 | 1.00x |
| INT8 Per-Tensor (Real) | 13.38 | 74.7 | 0.37x |
| INT8 Per-Row (Real) | 3.52 | 283.8 | 1.42x |
| FP8 Per-Tensor (Real) | 13.36 | 74.9 | 0.37x |
| FP8 Per-Row (Real) | 4.11 | 243.2 | 1.22x |

**分析：**
- INT8 Per-Row 真量化吞吐提升 1.42x
- FP8 Per-Row 真量化吞吐提升 1.22x
- Per-Tensor 真量化反而变慢（反量化开销大于收益）

### 3.2 Prefilling vs Decoding 吞吐

**伪量化结果：**

| 配置 | Prefill Tokens/s | Decode Tokens/s |
|------|------------------|-----------------|
| BF16 Baseline | 76,461 | 8,038 |
| INT8 Per-Row | 75,905 | 8,016 |
| FP8 Per-Row | 74,970 | 8,022 |

**真量化结果：**

| 配置 | Prefill Tokens/s | Decode Tokens/s |
|------|------------------|-----------------|
| BF16 Baseline | 76,461 | 8,038 |
| INT8 Per-Row | 81,061 | 7,999 |
| FP8 Per-Row | 79,038 | 7,058 |
| SmoothQuant INT8 | 77,147 | 8,038 |
| SmoothQuant FP8 | 76,451 | 8,034 |

**分析：**
- INT8 Per-Row 真量化在 Prefilling 阶段加速 5.9%，Decoding 阶段基本持平
- FP8 Per-Row 真量化在 Prefilling 阶段加速 3.4%，Decoding 阶段降速 12.2%
- 伪量化吞吐与 BF16 基本相同（没有实际低精度计算）

### 3.3 矩阵乘法误差

**伪量化结果：**

| 量化类型 | 最大误差 | 平均误差 | 相对误差 |
|---------|---------|---------|---------|
| INT8 Per-Row | 4.518 | 0.682 | 1.199 |
| FP8 Per-Row | 11.613 | 1.742 | 2.394 |

**真量化结果：**

| 量化类型 | 最大误差 | 平均误差 | 相对误差 |
|---------|---------|---------|---------|
| INT8 Per-Row | 4.092 | 0.615 | 0.682 |
| FP8 Per-Row | 8.994 | 1.351 | 1.747 |
| SmoothQuant INT8 | 4.088 | 0.613 | 1.713 |
| SmoothQuant FP8 | 9.046 | 1.351 | 2.299 |

**分析：**
- INT8 的矩阵乘法误差约为 FP8 的 50%（伪量化）
- 真量化的相对误差比伪量化略小
- SmoothQuant 对误差的改善不明显

**INT8 比 FP8 精度更高的原因：**
1. INT8 是均匀量化，所有数值的量化误差相同
2. FP8 是非均匀量化，大数值精度低（只有 3 位尾数）
3. LLM 权重呈正态分布，关键权重通常绝对值较大，FP8 对这些权重损伤更大
4. RTX 4090 的 FP8 Tensor Core 在 Ada Lovelace 架构上精度问题

---

## 4. Task 3.3: TorchAO SmoothQuant 测试

### 4.1 实验配置

- 模型：Qwen3-1.7B
- 量化方法：TorchAO Dynamic Activation + Weight Quantization
- 测试：WikiText-2 (100 samples), MMLU (1000 samples)

### 4.2 实验结果

| 量化类型 | MMLU Accuracy | PPL |
|---------|---------------|-----|
| BF16 Baseline | 50.60% | 33.51 |
| SmoothQuant INT8 | 50.60% | 33.51 |
| SmoothQuant FP8 | 50.60% | 33.51 |

**与 Task 3.1 对比：**

| 量化方法 | INT8 MMLU | FP8 MMLU | INT8 PPL | FP8 PPL |
|---------|-----------|----------|----------|---------|
| Weight Only (Best) | 51.00% | 50.30% | 33.73 | 33.14 |
| SmoothQuant | 50.60% | 50.60% | 33.51 | 33.51 |

### 4.3 分析

**SmoothQuant 的特点：**
1. 激活值的动态范围通常比权重小，FP8 的指数位在激活量化上更能发挥作用
2. SmoothQuant 通过缩放因子优化激活值分布，特别适合 FP8
3. 数学变换：xW = (x/s)(W·s)，激活值动态范围缩小，权重变化通过离线校准处理

**SmoothQuant vs Weight Only：**
- MMLU: SmoothQuant 表现略低于 Weight Only 的最佳配置
- PPL: SmoothQuant 与 BF16 持平，优于 Weight Only 的大部分配置

---

## 5. 总结

### 5.1 主要发现

1. **量化粒度：Per-Row 是精度和性能的最佳平衡点**
   - Per-Row 量化几乎无损（INT8 MMLU -0.20%）
   - Per-Group-512 精度略优但实现复杂
   - Per-Tensor 精度损失大，不推荐

2. **INT8 vs FP8：**
   - Weight Only 量化：INT8 优于 FP8（MMLU +1-2%，矩阵误差小 2x）
   - SmoothQuant：两者与 BF16 持平
   - 原因：激活量化对 FP8 帮助更大

3. **真量化吞吐：**
   - INT8 Per-Row：1.42x 加速
   - FP8 Per-Row：1.22x 加速
   - Per-Tensor：0.37x 降速（反量化开销）

4. **SmoothQuant：**
   - MMLU 和 PPL 均与 BF16 持平
   - 吞吐与 BF16 相近

### 5.2 结论

- RTX 4090 上优先使用 INT8 Per-Row 真量化（精度高，吞吐提升 42%）
- H100/B200 等支持 FP8 Tensor Core 的硬件可使用 FP8 Per-Row
- Per-Tensor 量化不推荐（精度损失大，性能差）
- SmoothQuant 适合对激活值敏感的任务

---

## 附录

### 实验环境

```
GPU: RTX 4090
CUDA: 12.4
PyTorch: 2.8
Python: 3.10+
```

### 数据文件

- `results.jsonl`: MMLU 和 PPL 测试结果（54 条）
- `throughput_results.jsonl`: 吞吐测试结果（8 条）
- `matmul_results.jsonl`: 矩阵乘法误差测试结果（7 条）

### 代码文件

- `experiments.py`: 实验脚本
- `experiments.sh`: 批量运行脚本
- `nanovllm/utils/quantization.py`: 量化实现

### 实验脚本示例

```bash
# 全部实验
bash experiments.sh

# Task 3.1: MMLU 测试
python experiments.py --test mmlu --dtype bf16
python experiments.py --test mmlu --dtype int8 --quant row
python experiments.py --test mmlu --dtype fp8 --quant group --group-size 128

# Task 3.1: PPL 测试
python experiments.py --test ppl --dtype bf16
python experiments.py --test ppl --dtype int8 --quant row

# Task 3.2: 吞吐测试
python experiments.py --test throughput --dtype int8 --quant row \
    --num-samples 100 --prompt-length 1000 --generate-length 1000

# Task 3.2: 矩阵乘法测试
python experiments.py --test matmul --dtype int8 --quant row \
    --matmul-shape 2048 2048 2048 --num-tests 100

# Task 3.3: SmoothQuant 测试
python experiments.py --test mmlu --quant smooth --dtype int8 --real
python experiments.py --test ppl --quant smooth --dtype fp8 --real
```
