"""
Task 3.2: 伪量化 PPL 测试
在 WikiText-2 上测试伪量化的 Perplexity
对比 INT8 vs FP8 的 PPL 表现
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.quantization import (
    fake_per_tensor_quant,
    fake_per_row_quant,
    fake_per_group_quant,
)


def apply_fake_quant_to_model(model, quant_type, linear_dtype, group_size=None):
    """
    对模型应用伪量化（量化后立即反量化，保持原 dtype）
    """
    print(f"应用伪量化: quant_type={quant_type}, dtype={linear_dtype}")

    for name, module in model.named_modules():
        # 只处理 Linear 层的权重
        if hasattr(module, 'weight') and 'Linear' in type(module).__name__:
            weight = module.weight.data

            if quant_type == "per_tensor":
                quantized_weight = fake_per_tensor_quant(weight, linear_dtype)
            elif quant_type == "per_row":
                # 对每一行进行量化
                weight_reshaped = weight.view(weight.size(0), -1)
                quantized_weight = torch.stack([
                    fake_per_row_quant(weight_reshaped[i], linear_dtype)
                    for i in range(weight_reshaped.size(0))
                ])
                quantized_weight = quantized_weight.view_as(weight)
            elif quant_type == "per_group":
                # 对每组进行量化
                weight_reshaped = weight.view(weight.size(0), -1)
                num_groups = weight_reshaped.size(1) // group_size
                weight_grouped = weight_reshaped.view(weight.size(0), num_groups, group_size)
                quantized_weight = torch.stack([
                    torch.stack([
                        fake_per_group_quant(weight_grouped[i, j], group_size, linear_dtype)
                        for j in range(num_groups)
                    ])
                    for i in range(weight.size(0))
                ])
                quantized_weight = quantized_weight.view_as(weight)
            else:
                continue

            # 替换权重（保持原 dtype）
            module.weight.data = quantized_weight

    print(f"伪量化应用完成")


def compute_ppl(model, tokenizer, num_samples=100):
    """
    计算 WikiText-2 上的 Perplexity
    参考 test_ppl.py 的实现
    """
    print("计算 WikiText-2 Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    total_loss = 0.0
    total_tokens = 0
    max_length = 4096
    texts = dataset["text"][:num_samples]

    with torch.no_grad():
        for text in texts:
            if len(text.strip()) == 0:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 2:
                continue

            # 分块处理
            for i in range(0, len(tokens) - 1, max_length):
                chunk = tokens[i : min(i + max_length + 1, len(tokens))]
                if len(chunk) < 2:
                    continue

                input_ids = torch.tensor([chunk[:-1]], device="cuda")
                targets = torch.tensor([chunk[1:]], device="cuda")
                positions = torch.arange(len(chunk) - 1, device="cuda").unsqueeze(0)

                hidden_states = model(input_ids, positions)
                logits = model.compute_logits(hidden_states)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += targets.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')
    return ppl


def main():
    model_path = "/root/Qwen3-1.7B"

    # 测试配置: 使用伪量化
    configs = [
        ("BF16", None, torch.bfloat16, None),
        ("INT8_Per_Row_Fake", "per_row", torch.int8, None),
        ("FP8_Per_Row_Fake", "per_row", torch.float8_e4m3fn, None),
        ("INT8_Per_Tensor_Fake", "per_tensor", torch.int8, None),
        ("FP8_Per_Tensor_Fake", "per_tensor", torch.float8_e4m3fn, None),
        ("INT8_Per_Group_128_Fake", "per_group", torch.int8, 128),
        ("FP8_Per_Group_128_Fake", "per_group", torch.float8_e4m3fn, 128),
    ]

    results = []

    print("=" * 80)
    print("Task 3.2: 伪量化 WikiText-2 PPL 测试")
    print("对比 INT8 vs FP8 在 WikiText-2 上的 Perplexity 表现")
    print("=" * 80)

    for config_name, quant_type, linear_dtype, group_size in configs:
        print(f"\n{'=' * 80}")
        print(f"测试配置: {config_name}")
        print(f"{'=' * 80}")

        # 初始化环境
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dist.init_process_group("nccl", f"tcp://localhost:2352", world_size=1, rank=0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        hf_config = Qwen3Config.from_pretrained(model_path)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        model = Qwen3ForCausalLM(hf_config, simple_attention=True)
        load_model(model, model_path)

        # 应用伪量化
        if quant_type is not None:
            apply_fake_quant_to_model(model, quant_type, linear_dtype, group_size)

        model.eval()

        # 计算 PPL
        t = time.time()
        ppl = compute_ppl(model, tokenizer, num_samples=100)
        elapsed = time.time() - t

        print(f"Perplexity: {ppl:.4f}")
        print(f"Time: {elapsed:.2f}s")

        results.append({
            "config": config_name,
            "quant_type": quant_type,
            "ppl": ppl,
            "time_sec": elapsed
        })

        # 清理
        del model
        torch.cuda.empty_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        dist.destroy_process_group()

    # 打印结果对比
    print(f"\n{'=' * 80}")
    print("WikiText-2 PPL 对比 (伪量化)")
    print(f"{'=' * 80}")
    print(f"{'配置':<25} {'Perplexity':<15} {'vs BF16':<15}")
    print(f"{'-' * 60}")

    bf16_ppl = results[0]["ppl"]
    for r in results:
        diff = r["ppl"] - bf16_ppl
        diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "-"
        print(f"{r['config']:<25} {r['ppl']:<15.4f} {diff_str:<15}")

    # INT8 vs FP8 对比
    print(f"\n{'=' * 80}")
    print("INT8 vs FP8 PPL 对比 (伪量化)")
    print(f"{'=' * 80}")

    int8_results = {r["quant_type"]: r for r in results if "INT8" in r["config"]}
    fp8_results = {r["quant_type"]: r for r in results if "FP8" in r["config"]}

    for quant_type in ["per_row", "per_tensor", "per_group"]:
        if quant_type in int8_results and quant_type in fp8_results:
            int8_ppl = int8_results[quant_type]["ppl"]
            fp8_ppl = fp8_results[quant_type]["ppl"]
            diff = fp8_ppl - int8_ppl
            better = "FP8" if diff < 0 else "INT8" if diff > 0 else "Same"
            print(f"{quant_type}: INT8={int8_ppl:.4f}, FP8={fp8_ppl:.4f}, 差异={diff:+.4f}, {better} 更优 (越低越好)")

    # 保存结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "task_32_ppl_fake_quant_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
