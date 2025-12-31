"""
Task 3.3: 运行 SmoothQuant 测试
测试 Int8 和 FP8 动态激活 + 权重量化
包括 PPL、MMLU 和矩阵乘法精度测试
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
print("Task 3.3: SmoothQuant 量化测试")
print("=" * 80)

# 使用 --quant-type both 和 --method both 同时测试 INT8/FP8 和 Direct/Generate 方法
# 矩阵乘法精度测试只运行一次
smoothquant_script = os.path.join(script_dir, "task_33_smoothquant.py")
smoothquant_result = subprocess.run(
    ["python", smoothquant_script, "--quant-type", "both", "--method", "both"],
    capture_output=True,
    text=True,
    env=os.environ
)
print(smoothquant_result.stdout)
if smoothquant_result.stderr:
    print("STDERR:", smoothquant_result.stderr[-500:])

# 读取所有结果
results = {"direct": [], "generate": []}
for quant_type in ["int8", "fp8"]:
    for method in ["direct", "generate"]:
        result_file = os.path.join(result_dir, f"task_33_{quant_type}_{method}_results.json")
        try:
            with open(result_file) as f:
                result_data = json.load(f)
                results[method].append(result_data)
        except Exception as e:
            print(f"无法读取结果文件 {result_file}: {e}")

# 打印对比结果
for method in ["direct", "generate"]:
    if results[method]:
        print(f"\n{'=' * 80}")
        print(f"SmoothQuant 结果对比 ({method} 方法)")
        print(f"{'=' * 80}")
        print(f"{'量化类型':<20} {'PPL':<15} {'MMLU Acc (%)':<15}")
        print(f"{'-' * 80}")
        for r in results[method]:
            quant_name = r.get("quantization", "Unknown")
            ppl = r.get("ppl", 0)
            mmlu = r.get("mmlu_accuracy", 0)
            print(f"{quant_name:<20} {ppl:<15.4f} {mmlu:<15.2f}")

# 打印 Direct vs Generate 对比
if results["direct"] and results["generate"]:
    print(f"\n{'=' * 80}")
    print("Direct vs Generate 方法对比")
    print(f"{'=' * 80}")
    print(f"{'量化类型':<20} {'Direct Acc (%)':<15} {'Generate Acc (%)':<15} {'差异':<10}")
    print(f"{'-' * 80}")

    # 按量化类型配对
    direct_dict = {r["quantization"]: r for r in results["direct"]}
    generate_dict = {r["quantization"]: r for r in results["generate"]}

    for quant_name in direct_dict:
        if quant_name in generate_dict:
            direct_acc = direct_dict[quant_name].get("mmlu_accuracy", 0)
            generate_acc = generate_dict[quant_name].get("mmlu_accuracy", 0)
            diff = generate_acc - direct_acc
            diff_str = f"{diff:+.2f}%" if abs(diff) > 0.01 else "-"
            print(f"{quant_name:<20} {direct_acc:<15.2f} {generate_acc:<15.2f} {diff_str:<10}")

# 保存完整结果
all_results = {
    "direct": results["direct"],
    "generate": results["generate"]
}
results_file = os.path.join(result_dir, "task_33_results.json")
with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n完整结果已保存到: {results_file}")
