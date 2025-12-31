"""
Task 3.1: PPL 测试脚本
使用真正的量化方法测试精度
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm import LLM


def compute_ppl_direct(model, tokenizer, num_samples=100):
    """
    直接计算 WikiText-2 上的 Perplexity
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

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    model_path = "/root/Qwen3-1.7B"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "BF16"

    # 量化配置映射
    # quant_type: None, "per_tensor", "per_row", "per_group"
    quant_configs = {
        "BF16": {"quant_type": None, "linear_dtype": torch.bfloat16, "group_size": None},
        "INT8_Per_Tensor": {"quant_type": "per_tensor", "linear_dtype": torch.int8, "group_size": None},
        "INT8_Per_Row": {"quant_type": "per_row", "linear_dtype": torch.int8, "group_size": None},
        "INT8_Per_Group_64": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 64},
        "INT8_Per_Group_128": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 128},
        "INT8_Per_Group_256": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 256},
        "INT8_Per_Group_512": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 512},
        "FP8_Per_Tensor": {"quant_type": "per_tensor", "linear_dtype": torch.float8_e4m3fn, "group_size": None},
        "FP8_Per_Row": {"quant_type": "per_row", "linear_dtype": torch.float8_e4m3fn, "group_size": None},
        "FP8_Per_Group_64": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 64},
        "FP8_Per_Group_128": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 128},
        "FP8_Per_Group_256": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 256},
        "FP8_Per_Group_512": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 512},
    }

    print(f"Testing PPL for config: {config_name}")

    config = quant_configs.get(config_name)
    if config is None:
        print(f"Unknown config: {config_name}")
        sys.exit(1)

    quant_type = config["quant_type"]
    linear_dtype = config["linear_dtype"]
    group_size = config["group_size"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if quant_type is not None:
        # 使用 LLM + 真正的量化
        print(f"Using LLM with quant_type={quant_type}, linear_dtype={linear_dtype}, group_size={group_size}")
        llm = LLM(
            model_path,
            enforce_eager=True,
            max_model_len=4096,
            linear_dtype=linear_dtype,
            quant_type=quant_type,
            group_size=group_size,
        )
        # TODO: 实现LLM模式的PPL计算
        # 暂时使用直接模型方式
        del llm

    # 使用直接模型方式计算 PPL
    print("Loading model...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dist.init_process_group("nccl", "tcp://localhost:2350", world_size=1, rank=0)

    hf_config = Qwen3Config.from_pretrained(model_path)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")
    model = Qwen3ForCausalLM(hf_config, simple_attention=True)
    load_model(model, model_path)

    # 应用真正的量化
    if quant_type == "per_tensor":
        from nanovllm.utils.quantization import apply_tensor_quant
        apply_tensor_quant(model, linear_dtype)
        print(f"Applied per-tensor quantization with dtype={linear_dtype}")
    elif quant_type == "per_row":
        from nanovllm.utils.quantization import apply_per_row_quant
        apply_per_row_quant(model, linear_dtype)
        print(f"Applied per-row quantization with dtype={linear_dtype}")
    elif quant_type == "per_group":
        from nanovllm.utils.quantization import apply_group_quant
        apply_group_quant(model, linear_dtype, group_size)
        print(f"Applied per-group quantization with dtype={linear_dtype}, group_size={group_size}")

    model.eval()
    ppl = compute_ppl_direct(model, tokenizer, num_samples=100)
    print(f"Perplexity: {ppl:.4f}")

    del model
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
    dist.destroy_process_group()

    # 保存结果
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"task_31_ppl_{config_name}.json")
    with open(result_file, "w") as f:
        json.dump({"config": config_name, "ppl": ppl}, f)
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
