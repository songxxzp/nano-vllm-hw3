"""
Task 3.1: 运行所有量化配置的 PPL 和 MMLU 测试
使用真正的量化方法测试精度
MMLU 测试同时使用 generate() 和 direct() 两种方法
"""
import subprocess
import json
import time
import os

# 设置环境变量
os.environ["PYTHONPATH"] = f"{os.path.dirname(os.path.abspath(__file__))}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, "results")
os.makedirs(result_dir, exist_ok=True)

# 所有配置
configs = [
    "BF16",
    "INT8_Per_Tensor",
    "INT8_Per_Row",
    "INT8_Per_Group_64",
    "INT8_Per_Group_128",
    "INT8_Per_Group_256",
    "INT8_Per_Group_512",
    "FP8_Per_Tensor",
    "FP8_Per_Row",
    "FP8_Per_Group_64",
    "FP8_Per_Group_128",
    "FP8_Per_Group_256",
    "FP8_Per_Group_512",
]

results = {"direct": [], "generate": []}

print("=" * 80)
print("Task 3.1: 量化精度测试 (使用真正的量化方法)")
print("=" * 80)

for config in configs:
    print(f"\n{'=' * 80}")
    print(f"测试配置: {config}")
    print(f"{'=' * 80}")

    # 运行 PPL 测试 (仅 direct 方法)
    print(f"\n[{config}] 运行 PPL 测试...")
    ppl_script = os.path.join(script_dir, "task_31_ppl_test.py")
    ppl_result = subprocess.run(
        ["python", ppl_script, config],
        capture_output=True,
        text=True,
        env=os.environ
    )
    print(ppl_result.stdout)
    if ppl_result.stderr:
        print("STDERR:", ppl_result.stderr)

    # 等待一下让GPU释放
    time.sleep(3)

    # 运行 MMLU 测试 - direct 方法
    print(f"\n[{config}] 运行 MMLU 测试 (direct 方法)...")
    mmlu_script = os.path.join(script_dir, "task_31_mmlu_test.py")
    mmlu_result = subprocess.run(
        ["python", mmlu_script, config, "direct"],
        capture_output=True,
        text=True,
        env=os.environ
    )
    print(mmlu_result.stdout)
    if mmlu_result.stderr:
        print("STDERR:", mmlu_result.stderr)

    # 等待一下让GPU释放
    time.sleep(3)

    # 运行 MMLU 测试 - generate 方法
    print(f"\n[{config}] 运行 MMLU 测试 (generate 方法)...")
    mmlu_result = subprocess.run(
        ["python", mmlu_script, config, "generate"],
        capture_output=True,
        text=True,
        env=os.environ
    )
    print(mmlu_result.stdout)
    if mmlu_result.stderr:
        print("STDERR:", mmlu_result.stderr)

    # 读取结果
    ppl_file = os.path.join(result_dir, f"task_31_ppl_{config}.json")
    mmlu_direct_file = os.path.join(result_dir, f"task_31_mmlu_{config}_direct.json")
    mmlu_generate_file = os.path.join(result_dir, f"task_31_mmlu_{config}_generate.json")

    ppl_val = None
    mmlu_direct_val = None
    mmlu_generate_val = None

    if os.path.exists(ppl_file):
        with open(ppl_file) as f:
            ppl_data = json.load(f)
            ppl_val = ppl_data.get("ppl")

    if os.path.exists(mmlu_direct_file):
        with open(mmlu_direct_file) as f:
            mmlu_data = json.load(f)
            mmlu_direct_val = mmlu_data.get("accuracy")

    if os.path.exists(mmlu_generate_file):
        with open(mmlu_generate_file) as f:
            mmlu_data = json.load(f)
            mmlu_generate_val = mmlu_data.get("accuracy")

    if mmlu_direct_val is not None and ppl_val is not None:
        results["direct"].append({
            "config": config,
            "ppl": ppl_val,
            "mmlu_accuracy": mmlu_direct_val
        })

    if mmlu_generate_val is not None and ppl_val is not None:
        results["generate"].append({
            "config": config,
            "ppl": ppl_val,
            "mmlu_accuracy": mmlu_generate_val
        })

    # 等待一下让GPU释放
    time.sleep(3)

# 打印汇总结果
print(f"\n{'=' * 80}")
print("测试结果汇总 (Direct 方法)")
print(f"{'=' * 80}")
print(f"{'配置':<25} {'Perplexity':<12} {'MMLU Acc (%)':<15}")
print(f"{'-' * 80}")
for result in results["direct"]:
    print(f"{result['config']:<25} {result['ppl']:<12.4f} {result['mmlu_accuracy']:<15.2f}")

print(f"\n{'=' * 80}")
print("测试结果汇总 (Generate 方法)")
print(f"{'=' * 80}")
print(f"{'配置':<25} {'Perplexity':<12} {'MMLU Acc (%)':<15}")
print(f"{'-' * 80}")
for result in results["generate"]:
    print(f"{result['config']:<25} {result['ppl']:<12.4f} {result['mmlu_accuracy']:<15.2f}")

# 打印分组汇总
print(f"\n{'=' * 80}")
print("INT8 不同量化粒度对比 (Direct)")
print(f"{'=' * 80}")
int8_results = [r for r in results["direct"] if "INT8" in r["config"]]
for r in int8_results:
    print(f"{r['config']:<25} {r['ppl']:<12.4f} {r['mmlu_accuracy']:<15.2f}")

print(f"\n{'=' * 80}")
print("FP8 不同量化粒度对比 (Direct)")
print(f"{'=' * 80}")
fp8_results = [r for r in results["direct"] if "FP8" in r["config"]]
for r in fp8_results:
    print(f"{r['config']:<25} {r['ppl']:<12.4f} {r['mmlu_accuracy']:<15.2f}")

print(f"\n{'=' * 80}")
print("INT8 不同量化粒度对比 (Generate)")
print(f"{'=' * 80}")
int8_results = [r for r in results["generate"] if "INT8" in r["config"]]
for r in int8_results:
    print(f"{r['config']:<25} {r['ppl']:<12.4f} {r['mmlu_accuracy']:<15.2f}")

print(f"\n{'=' * 80}")
print("FP8 不同量化粒度对比 (Generate)")
print(f"{'=' * 80}")
fp8_results = [r for r in results["generate"] if "FP8" in r["config"]]
for r in fp8_results:
    print(f"{r['config']:<25} {r['ppl']:<12.4f} {r['mmlu_accuracy']:<15.2f}")

# 保存完整结果
results_file_direct = os.path.join(result_dir, "task_31_results_direct.json")
results_file_generate = os.path.join(result_dir, "task_31_results_generate.json")
with open(results_file_direct, "w") as f:
    json.dump(results["direct"], f, indent=2)
print(f"\nDirect 结果已保存到: {results_file_direct}")
with open(results_file_generate, "w") as f:
    json.dump(results["generate"], f, indent=2)
print(f"Generate 结果已保存到: {results_file_generate}")
