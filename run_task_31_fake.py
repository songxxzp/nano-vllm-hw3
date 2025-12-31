"""
Task 3.1: 批量运行 Fake Quantization 实验
测试所有三种 fake quantization (per-tensor, per-row, per-group) 的 PPL 和 MMLU
"""
import os
import subprocess
import sys

# 设置环境变量
os.environ["PYTHONPATH"] = "/datadisk/workspace/Quant/nano-vllm-hw3:" + os.environ.get("PYTHONPATH", "")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 测试配置
configs = [
    "INT8_Per_Tensor_Fake",
    "INT8_Per_Row_Fake",
    "INT8_Per_Group_128_Fake",
    "FP8_Per_Tensor_Fake",
    "FP8_Per_Row_Fake",
    "FP8_Per_Group_128_Fake",
]

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"SUCCESS: {description} completed")
        return True

def main():
    print("="*80)
    print("Task 3.1: Fake Quantization Batch Experiments")
    print("="*80)

    all_success = True

    # 运行 PPL 测试
    print("\n" + "="*80)
    print("PHASE 1: Running PPL Tests")
    print("="*80)

    for config in configs:
        cmd = f"conda run -n torch python task_31_ppl_fake_test.py {config}"
        description = f"PPL Test - {config}"
        if not run_command(cmd, description):
            all_success = False

    # 运行 MMLU 测试 (Direct 方法)
    print("\n" + "="*80)
    print("PHASE 2: Running MMLU Tests (Direct Method)")
    print("="*80)

    for config in configs:
        cmd = f"conda run -n torch python task_31_mmlu_fake_test.py {config} direct"
        description = f"MMLU Direct Test - {config}"
        if not run_command(cmd, description):
            all_success = False

    # 运行 MMLU 测试 (Generate 方法)
    print("\n" + "="*80)
    print("PHASE 3: Running MMLU Tests (Generate Method)")
    print("="*80)

    for config in configs:
        cmd = f"conda run -n torch python task_31_mmlu_fake_test.py {config} generate"
        description = f"MMLU Generate Test - {config}"
        if not run_command(cmd, description):
            all_success = False

    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

    if all_success:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the output above.")

    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
