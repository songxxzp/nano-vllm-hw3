"""
Task 3.2: 量化吞吐测试
测试不同量化方法的吞吐量
支持 Per-Row 量化 (使用 LLM 接口)
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset

from nanovllm import LLM, SamplingParams


def test_throughput(config_name, quant_type=None, linear_dtype=None, group_size=None, model_path="/root/Qwen3-1.7B"):
    """
    测试不同量化配置的吞吐量
    """
    print(f"\n{'=' * 80}")
    print(f"测试 {config_name} 量化吞吐")
    print(f"{'=' * 80}")

    # 构建 LLM 参数
    llm_kwargs = {
        "model": model_path,
        "enforce_eager": True,
        "max_model_len": 4096,
    }
    if quant_type is not None:
        llm_kwargs["quant_type"] = quant_type
    if linear_dtype is not None:
        llm_kwargs["linear_dtype"] = linear_dtype
    if group_size is not None:
        llm_kwargs["group_size"] = group_size

    print(f"LLM kwargs: {llm_kwargs}")

    # 加载模型
    llm = LLM(**llm_kwargs)

    # 准备测试数据
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"][:100] if len(t.strip()) > 100][:20]

    # 生成不同长度的 prompts
    prompts_short = [t[:200] for t in texts[:10]]  # 短文本 (~50 tokens)
    prompts_long = [t[:800] for t in texts[10:]]   # 长文本 (~200 tokens)

    # 测试 prefilling (短文本，生成 1 token)
    print(f"\n[Prefilling] 测试短文本 ({len(prompts_short)} prompts)")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    t = time.time()
    outputs = llm.generate(prompts_short, sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    prefill_throughput = len(prompts_short) / elapsed
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    tokens_per_sec = total_tokens / elapsed
    print(f"Time: {elapsed:.2f}s, Throughput: {prefill_throughput:.2f} samples/s, {tokens_per_sec:.2f} tokens/s")

    # 测试 decoding (短文本，生成 128 tokens)
    print(f"\n[Decoding] 测试生成 128 tokens ({len(prompts_short[:5])} prompts)")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    t = time.time()
    outputs = llm.generate(prompts_short[:5], sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    decode_throughput = total_tokens / elapsed
    print(f"Time: {elapsed:.2f}s, Tokens: {total_tokens}, Throughput: {decode_throughput:.2f} tokens/s")

    # 测试长文本 prefill
    print(f"\n[Long Prefilling] 测试长文本 ({len(prompts_long)} prompts)")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    t = time.time()
    outputs = llm.generate(prompts_long, sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    long_prefill_throughput = len(prompts_long) / elapsed
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    long_tokens_per_sec = total_tokens / elapsed
    print(f"Time: {elapsed:.2f}s, Throughput: {long_prefill_throughput:.2f} samples/s, {long_tokens_per_sec:.2f} tokens/s")

    del llm

    return {
        "config": config_name,
        "prefill_samples_per_sec": prefill_throughput,
        "prefill_tokens_per_sec": tokens_per_sec,
        "decode_tokens_per_sec": decode_throughput,
        "long_prefill_samples_per_sec": long_prefill_throughput,
        "long_prefill_tokens_per_sec": long_tokens_per_sec,
    }


def main():
    results = []

    # 测试配置 - 只测试支持的量化类型
    configs = [
        ("BF16", None, torch.bfloat16, None),
        ("INT8_Per_Row", "per_row", torch.int8, None),
        ("FP8_Per_Row", "per_row", torch.float8_e4m3fn, None),
    ]

    for config_name, quant_type, linear_dtype, group_size in configs:
        try:
            result = test_throughput(config_name, quant_type, linear_dtype, group_size)
            results.append(result)
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # 打印对比结果
    print(f"\n{'=' * 80}")
    print("吞吐对比结果")
    print(f"{'=' * 80}")
    print(f"{'配置':<15} {'Prefill (samples/s)':<20} {'Long Prefill (tokens/s)':<25} {'Decode (tokens/s)':<20}")
    print(f"{'-' * 80}")
    for r in results:
        print(f"{r['config']:<15} {r['prefill_samples_per_sec']:<20.2f} {r['long_prefill_tokens_per_sec']:<25.2f} {r['decode_tokens_per_sec']:<20.2f}")

    # 保存结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "task_32_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
