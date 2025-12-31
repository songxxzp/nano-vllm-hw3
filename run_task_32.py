"""
Task 3.2: 运行所有量化吞吐测试
包括:
1. 伪量化矩阵乘法精度测试
2. 伪量化 MMLU 精度测试
3. 伪量化 WikiText-2 PPL 测试
4. 吞吐量测试
"""
import subprocess
import json
import os

# 设置环境变量
os.environ["PYTHONPATH"] = f"{os.path.dirname(os.path.abspath(__file__))}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "results")
os.makedirs(result_dir, exist_ok=True)

print("=" * 80)
print("Task 3.2: 量化吞吐与精度测试")
print("=" * 80)

# ============ 测试 1: 伪量化矩阵乘法精度 ============
print(f"\n{'=' * 80}")
print("测试 1: 伪量化矩阵乘法精度测试")
print(f"{'=' * 80}")

fake_quant_script = os.path.join(script_dir, "task_32_fake_quant_test.py")
result = subprocess.run(
    ["python", fake_quant_script],
    capture_output=True,
    text=True,
    env=os.environ
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])

# 等待 GPU 释放
import time
time.sleep(3)

# ============ 测试 2: 伪量化 MMLU 精度测试 ============
print(f"\n{'=' * 80}")
print("测试 2: 伪量化 MMLU 精度测试")
print(f"{'=' * 80}")

mmlu_fake_quant_script = os.path.join(script_dir, "task_32_mmlu_fake_quant_test.py")
result = subprocess.run(
    ["python", mmlu_fake_quant_script],
    capture_output=True,
    text=True,
    env=os.environ
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])

time.sleep(3)

# ============ 测试 3: 伪量化 WikiText-2 PPL 测试 ============
print(f"\n{'=' * 80}")
print("测试 3: 伪量化 WikiText-2 PPL 测试")
print(f"{'=' * 80}")

ppl_fake_quant_script = os.path.join(script_dir, "task_32_ppl_fake_quant_test.py")
result = subprocess.run(
    ["python", ppl_fake_quant_script],
    capture_output=True,
    text=True,
    env=os.environ
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])

time.sleep(3)

# ============ 测试 4: 吞吐量测试 ============
print(f"\n{'=' * 80}")
print("测试 4: 吞吐量测试")
print(f"{'=' * 80}")

throughput_script = os.path.join(script_dir, "task_32_throughput_test.py")
result = subprocess.run(
    ["python", throughput_script],
    capture_output=False,
    text=True,
    env=os.environ
)

# ============ 汇总所有结果 ============
print(f"\n{'=' * 80}")
print("Task 3.2 完整结果汇总")
print(f"{'=' * 80}")

# 读取并显示所有结果
all_results = {}

# 1. 伪量化矩阵乘法精度
fake_quant_file = os.path.join(result_dir, "task_32_fake_quant_results.json")
if os.path.exists(fake_quant_file):
    with open(fake_quant_file) as f:
        all_results["fake_quant_matmul"] = json.load(f)

# 2. 伪量化 MMLU 精度
mmlu_fake_quant_file = os.path.join(result_dir, "task_32_mmlu_fake_quant_results.json")
if os.path.exists(mmlu_fake_quant_file):
    with open(mmlu_fake_quant_file) as f:
        all_results["fake_quant_mmlu"] = json.load(f)

# 3. 吞吐量
throughput_file = os.path.join(result_dir, "task_32_results.json")
if os.path.exists(throughput_file):
    with open(throughput_file) as f:
        all_results["throughput"] = json.load(f)

# 4. 伪量化 PPL
ppl_fake_quant_file = os.path.join(result_dir, "task_32_ppl_fake_quant_results.json")
if os.path.exists(ppl_fake_quant_file):
    with open(ppl_fake_quant_file) as f:
        all_results["ppl_fake_quant"] = json.load(f)

# 显示汇总
if "fake_quant_matmul" in all_results:
    print(f"\n1. 伪量化矩阵乘法精度:")
    print(f"   {'形状':<12} {'INT8 误差':<15} {'FP8 误差':<15} {'FP8/INT8':<10}")
    print(f"   {'-' * 60}")
    for r in all_results["fake_quant_matmul"]:
        shape = r["shape"]
        int8_err = r.get("per_row_error", 0)
        fp8_err = r.get("per_row_error", 0)
        ratio = fp8_err / int8_err if int8_err > 0 else 0
        print(f"   {shape:<12} {int8_err:<15.6f} {fp8_err:<15.6f} {ratio:<10.2f}x")

if "fake_quant_mmlu" in all_results:
    print(f"\n2. 伪量化 MMLU 精度:")
    print(f"   {'配置':<25} {'MMLU Acc (%)':<15}")
    print(f"   {'-' * 45}")
    for r in all_results["fake_quant_mmlu"]:
        print(f"   {r['config']:<25} {r['mmlu_accuracy']:<15.2f}")

if "throughput" in all_results:
    print(f"\n3. 吞吐量测试:")
    print(f"   {'配置':<15} {'Prefill (samples/s)':<20} {'Long Prefill (tokens/s)':<25} {'Decode (tokens/s)':<20}")
    print(f"   {'-' * 90}")
    for r in all_results["throughput"]:
        print(f"   {r['config']:<15} {r['prefill_samples_per_sec']:<20.2f} {r['long_prefill_tokens_per_sec']:<25.2f} {r['decode_tokens_per_sec']:<20.2f}")

if "ppl_fake_quant" in all_results:
    print(f"\n4. 伪量化 WikiText-2 PPL:")
    print(f"   {'配置':<25} {'Perplexity':<15} {'vs BF16':<15}")
    print(f"   {'-' * 60}")
    bf16_ppl = all_results["ppl_fake_quant"][0]["ppl"]
    for r in all_results["ppl_fake_quant"]:
        diff = r["ppl"] - bf16_ppl
        diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "-"
        print(f"   {r['config']:<25} {r['ppl']:<15.4f} {diff_str:<15}")

# 保存完整结果
complete_results_file = os.path.join(result_dir, "task_32_complete_results.json")
with open(complete_results_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n完整结果已保存到: {complete_results_file}")

print(f"\n{'=' * 80}")
print("Task 3.2 测试完成!")
print(f"{'=' * 80}")
