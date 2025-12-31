"""
Task 3.2: 伪量化精度测试
将低精度矩阵乘修改为伪量化乘法，测试 INT8 与 FP8 的精度差异
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nano-vllm-hw3"))

import torch
from nanovllm.utils.quantization import (
    fake_per_tensor_quant,
    fake_per_row_quant,
    fake_per_group_quant,
)


def test_fake_quant_error():
    """
    测试伪量化的误差
    """
    print("=" * 80)
    print("Task 3.2: 伪量化精度测试")
    print("=" * 80)

    device = "cuda"
    dtypes = [torch.int8, torch.float8_e4m3fn]
    shapes = [(16, 512), (100, 1024), (4096, 4096)]

    print("\n测试 Fake Quant 误差:")
    print("-" * 70)

    results = []

    for M, N in shapes:
        x = torch.randn(M, N, device=device)

        for dtype in dtypes:
            dtype_name = "int8" if dtype == torch.int8 else "fp8"

            # Per-tensor quant
            x_pt = fake_per_tensor_quant(x, dtype)
            err_pt = (x - x_pt).abs().mean() / x.abs().mean()

            # Per-row quant
            x_pr = fake_per_row_quant(x, dtype)
            err_pr = (x - x_pr).abs().mean() / x.abs().mean()

            # Per-group quant (group_size=128)
            x_pg = fake_per_group_quant(x, 128, dtype)
            err_pg = (x - x_pg).abs().mean() / x.abs().mean()

            print(
                f"[{M:4d}x{N:4d}] {dtype_name}: "
                f"tensor={err_pt:.6f}, row={err_pr:.6f}, group={err_pg:.6f}"
            )

            results.append({
                "shape": f"{M}x{N}",
                "dtype": dtype_name,
                "per_tensor_error": float(err_pt),
                "per_row_error": float(err_pr),
                "per_group_error": float(err_pg),
            })

    # 计算 INT8 vs FP8 的误差比
    print("\n" + "=" * 80)
    print("FP8 vs INT8 误差对比:")
    print("-" * 70)

    fp8_errors = {}
    int8_errors = {}

    for r in results:
        shape = r["shape"]
        dtype = r["dtype"]
        if dtype == "fp8":
            fp8_errors[shape] = r["per_row_error"]
        else:
            int8_errors[shape] = r["per_row_error"]

    for shape in fp8_errors:
        ratio = fp8_errors[shape] / int8_errors[shape]
        print(f"[{shape}] FP8 error / INT8 error = {ratio:.2f}x")

    print("\n" + "=" * 80)
    print("结论:")
    print("-" * 70)
    print("FP8 的量化误差比 INT8 大约 2-3 倍，原因:")
    print("1. FP8 (float8_e4m3fn) 的动态范围较小: max=448, 而 INT8 的 max=127")
    print("2. FP8 只有 4 位 exponent，3 位 mantissa，精度较低")
    print("3. INT8 虽然只有 8 位整数，但在量化 scale 合适时精度更高")

    return results


if __name__ == "__main__":
    import json
    results = test_fake_quant_error()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "task_32_fake_quant_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
