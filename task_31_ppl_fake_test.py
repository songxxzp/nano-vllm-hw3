"""
Task 3.1: Fake Quantization PPL 测试脚本
使用 fake_per_tensor_quant, fake_per_row_quant, fake_per_group_quant 测试 PPL
"""
import os
import sys
import random
import json

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.quantization import (
    fake_per_tensor_quant,
    fake_per_row_quant,
    fake_per_group_quant,
    apply_weight_fake_quant,
)


def compute_ppl(model, tokenizer, num_samples=100):
    """计算 WikiText-2 Perplexity"""
    print("计算 WikiText-2 Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    random.seed(42)
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    total_loss = 0.0
    total_tokens = 0
    max_length = 4096

    model.eval()
    with torch.no_grad():
        for example in dataset:
            text = example["text"]
            if not text or len(text.strip()) < 10:
                continue

            # Tokenize without special tokens
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 2:
                continue

            # 分块处理
            for i in range(0, len(tokens) - 1, max_length):
                chunk = tokens[i : min(i + max_length + 1, len(tokens))]
                if len(chunk) < 2:
                    continue

                # input_ids: 所有 token 除了最后一个
                # targets: 所有 token 除了第一个（预测下一个 token）
                input_ids = torch.tensor([chunk[:-1]], device="cuda")
                targets = torch.tensor([chunk[1:]], device="cuda")
                positions = torch.arange(len(chunk) - 1, device="cuda").unsqueeze(0)

                # Forward pass
                hidden_states = model(input_ids, positions)
                logits = model.compute_logits(hidden_states)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="sum"
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    model_path = "/root/Qwen3-1.7B"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "INT8_Per_Tensor_Fake"

    # 量化配置映射
    quant_configs = {
        "INT8_Per_Tensor_Fake": {
            "quant_type": "per_tensor",
            "linear_dtype": torch.int8,
            "group_size": None,
            "fake_quant": lambda x, dtype=fake_per_tensor_quant: fake_per_tensor_quant(x, torch.int8)
        },
        "INT8_Per_Row_Fake": {
            "quant_type": "per_row",
            "linear_dtype": torch.int8,
            "group_size": None,
            "fake_quant": lambda x: fake_per_row_quant(x, torch.int8)
        },
        "INT8_Per_Group_128_Fake": {
            "quant_type": "per_group",
            "linear_dtype": torch.int8,
            "group_size": 128,
            "fake_quant": lambda x: fake_per_group_quant(x, 128, torch.int8)
        },
        "FP8_Per_Tensor_Fake": {
            "quant_type": "per_tensor",
            "linear_dtype": torch.float8_e4m3fn,
            "group_size": None,
            "fake_quant": lambda x: fake_per_tensor_quant(x, torch.float8_e4m3fn)
        },
        "FP8_Per_Row_Fake": {
            "quant_type": "per_row",
            "linear_dtype": torch.float8_e4m3fn,
            "group_size": None,
            "fake_quant": lambda x: fake_per_row_quant(x, torch.float8_e4m3fn)
        },
        "FP8_Per_Group_128_Fake": {
            "quant_type": "per_group",
            "linear_dtype": torch.float8_e4m3fn,
            "group_size": 128,
            "fake_quant": lambda x: fake_per_group_quant(x, 128, torch.float8_e4m3fn)
        },
    }

    print(f"Testing PPL for config: {config_name}")

    config = quant_configs.get(config_name)
    if config is None:
        print(f"Unknown config: {config_name}")
        sys.exit(1)

    fake_quant_fn = config["fake_quant"]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 使用动态端口避免冲突
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=1, rank=0)

    hf_config = Qwen3Config.from_pretrained(model_path)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")
    model = Qwen3ForCausalLM(hf_config, simple_attention=True)
    load_model(model, model_path)

    # 应用伪量化
    print(f"Applying fake quantization: {config_name}")
    apply_weight_fake_quant(model, fake_quant_fn)
    print(f"Applied fake {config['quant_type']} quantization with dtype={config['linear_dtype']}")

    model.eval()
    ppl = compute_ppl(model, tokenizer, num_samples=100)
    print(f"WikiText-2 Perplexity: {ppl:.2f}")

    del model
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
    dist.destroy_process_group()

    # 保存结果
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"task_31_ppl_{config_name}.json")
    with open(result_file, "w") as f:
        json.dump({
            "config": config_name,
            "ppl": ppl
        }, f)
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
