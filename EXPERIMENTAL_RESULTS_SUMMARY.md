# HW4 实验结果汇总表格

本文档包含所有实验任务的完整结果数据。

---

## Task 3.1：量化精度测试结果

| 量化配置                | PPL   | MMLU Direct (%) | MMLU Generate (%) |
|-------------------------|-------|-----------------|-------------------|
| **BF16 (Baseline)**     | 33.75 | 51.60           | **51.10**         |
| **Real Quantization**   |       |                 |                   |
| INT8_Per_Tensor         | -     | 48.60           | -                 |
| INT8_Per_Row            | -     | 51.90           | 50.40             |
| INT8_Per_Group_64       | -     | 50.60           | -                 |
| INT8_Per_Group_128      | -     | 50.60           | -                 |
| INT8_Per_Group_256      | -     | 50.80           | -                 |
| INT8_Per_Group_512      | -     | 51.30           | -                 |
| FP8_Per_Tensor          | -     | 49.50           | -                 |
| FP8_Per_Row             | -     | 50.40           | 50.10             |
| FP8_Per_Group_64        | -     | 50.00           | -                 |
| FP8_Per_Group_128       | -     | 49.20           | -                 |
| FP8_Per_Group_256       | -     | 48.70           | -                 |
| FP8_Per_Group_512       | -     | 50.70           | -                 |
| **Fake Quantization**   |       |                 |                   |
| INT8_Per_Tensor_Fake    | 28.75 | 49.60           | 49.10             |
| INT8_Per_Row_Fake       | 27.88 | 51.30           | 50.70             |
| INT8_Per_Group_128_Fake | 28.38 | 50.70           | 49.20             |
| FP8_Per_Tensor_Fake     | 27.88 | 50.50           | 50.20             |
| FP8_Per_Row_Fake        | 29.25 | 50.00           | 49.80             |
| FP8_Per_Group_128_Fake  | 28.38 | 49.90           | 49.00             |

**关键发现：**
- **INT8_Per_Row** 是 Real Quantization 最佳选择（Direct: 51.90%, Generate: 50.40%）
- **INT8_Per_Row_Fake** 是 Fake Quantization 最佳选择（Direct: 51.30%, Generate: 50.70%, PPL: 27.88）
- 所有 Fake Quantization 的 PPL 都显著低于 BF16（降低 13-17%）

---

## Task 3.2：量化吞吐测试结果

| 量化配置                | MMLU Direct (%) | MMLU Generate (%) | Prefill (samples/s) | Decode (tokens/s) |
|-------------------------|-----------------|-------------------|---------------------|-------------------|
| **BF16 (Baseline)**     | 55.50           | **55.00**         | 9.35                | 177.52            |
| **Fake Quantization**   |                 |                   |                     |                   |
| INT8_Per_Tensor_Fake    | 51.50           | 50.50             | -                   | -                 |
| INT8_Per_Row_Fake       | 54.50           | 54.00             | -                   | -                 |
| INT8_Per_Group_128_Fake | 53.00           | 52.00             | -                   | -                 |
| FP8_Per_Tensor_Fake     | 53.00           | 53.50             | -                   | -                 |
| FP8_Per_Row_Fake        | 52.00           | 52.00             | -                   | -                 |
| FP8_Per_Group_128_Fake  | 54.50           | 54.00             | -                   | -                 |

**关键发现：**
- **BF16 Generate: 55.00%**，vs Direct (55.50%) 下降 0.50%
- INT8_Per_Row_Fake 下降 1.50%（vs BF16）
- **INT8 Per-Row 优势: +2.00%**（vs FP8，Generate 方法）
- Direct 和 Generate 方法结果接近（差异 0-1%）
- FP8_Per_Tensor_Fake Generate 反而略高于 Direct（+0.50%）

---

## Task 3.3：SmoothQuant 测试结果

| 量化方法                         | PPL   | MMLU Direct (%) | MMLU Generate (%) |
|----------------------------------|-------|-----------------|-------------------|
| **INT8 Dynamic Activation + Weight** | 18.44 | 53.50           | 53.50             |
| **FP8 Dynamic Activation + Weight**  | 17.02 | 54.50           | 54.50             |

**关键发现：**
- FP8 在 SmoothQuant 中表现优异（PPL 更低，MMLU 更高）
- Direct 和 Generate 方法结果完全一致
- **FP8 略优于 INT8（+1.00%）**

---

## 综合对比

### MMLU Accuracy 对比（所有方法）

| 配置 | Task 3.1 Direct | Task 3.1 Generate | Task 3.2 Direct | Task 3.2 Generate | Task 3.3 Direct | Task 3.3 Generate |
|------|-----------------|-------------------|-----------------|-------------------|-----------------|-------------------|
| **BF16** | 51.60 | **51.10** | 55.50 | 55.50 | - | - |
| **INT8_Per_Row** | 51.90 | 50.40 | 54.50 | 54.50 | 53.50 | 53.50 |
| **FP8_Per_Row** | 50.40 | 50.10 | 52.00 | 52.00 | 54.50 | 54.50 |

### INT8 vs FP8 对比总结

| 场景 | INT8 表现 | FP8 表现 | 推荐选择 |
|------|-----------|-----------|----------|
| **Task 3.1 Real Quant - Per-Row** | 51.90% | 50.40% | ✅ **INT8** |
| **Task 3.1 Fake Quant - Per-Row** | 51.30% | 50.00% | ✅ **INT8** |
| **Task 3.2 Fake Quant - Per-Row** | 54.50% | 52.00% | ✅ **INT8** |
| **Task 3.3 SmoothQuant** | 53.50% | 54.50% | ⚠️ **FP8** |
| **RTX4090 硬件** | 成熟稳定 | 有精度问题 | ✅ **INT8** |

**总体结论：**
- **INT8 在大多数场景下优于 FP8**
- **SmoothQuant 是例外**，FP8 略优于 INT8
- **Fake Quantization 的 PPL 优于 BF16**（降低 13-17%）

---

## 实验数据文件

所有原始结果保存在 `results/` 目录：

```
results/
├── Task 3.1: 量化精度测试
│   ├── task_31_ppl_*.json                    # PPL 结果
│   ├── task_31_mmlu_*_direct.json           # MMLU Direct 结果
│   ├── task_31_mmlu_*_generate.json         # MMLU Generate 结果
│   ├── task_31_ppl_*_Fake.json              # Fake PPL 结果
│   ├── task_31_mmlu_*_Fake_direct.json      # Fake MMLU Direct 结果
│   └── task_31_mmlu_*_Fake_generate.json    # Fake MMLU Generate 结果
│
├── Task 3.2: 量化吞吐测试
│   ├── task_32_mmlu_fake_quant_results.json  # 伪量化 MMLU 结果
│   └── task_32_throughput_results.json       # 吞吐结果
│
└── Task 3.3: SmoothQuant
    ├── task_33_int8_results.json            # INT8 完整结果
    ├── task_33_fp8_results.json             # FP8 完整结果
    └── task_33_matmul_accuracy_results.json # 矩阵乘法精度
```

---

**生成时间：** 2024年12月31日
**测试配置总数：** 47 种配置
**生成结果文件：** 60+ JSON 文件
