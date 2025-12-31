"""
Task 3.1: Fake Quantization MMLU 测试脚本
使用 fake_per_tensor_quant, fake_per_row_quant, fake_per_group_quant 测试 MMLU
支持 direct 和 generate 两种方法
"""
import os
import sys
import random
import json
import time

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


def compute_mmlu_accuracy_generate(model, tokenizer, num_samples=1000):
    """
    使用手动的 greedy decode 计算 MMLU 5-shot 准确率
    适用于 fake quantization（Qwen3ForCausalLM 没有 generate 方法）
    """
    print("计算 MMLU Accuracy (generate method with manual greedy decode)...")
    dataset = list(load_dataset("cais/mmlu", "all", split="test"))
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[:num_samples]
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")

    def format_example(question, choices, answer):
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{'ABCD'[i]}. {choice}\n"
        prompt += f"Answer: {answer}\n\n"
        return prompt

    # 构建 5-shot prompts
    few_shot_prompt = "The following are multiple choice questions (with answers).\n\n"
    for example in dev_dataset.select(range(5)):
        few_shot_prompt += format_example(
            example["question"], example["choices"], "ABCD"[example["answer"]]
        )

    prompts = []
    answers = []

    for example in dataset:
        prompt = few_shot_prompt + format_example(
            example["question"], example["choices"], ""
        ).replace("Answer: \n", "Answer:")
        prompts.append(prompt)
        answers.append("ABCD"[example["answer"]])

    # 手动 greedy decode 生成 1 个 token
    correct = 0
    total = len(prompts)

    t = time.time()
    with torch.no_grad():
        for i, (prompt, true_answer) in enumerate(zip(prompts, answers)):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            position_ids = torch.arange(input_ids.size(1), device="cuda").unsqueeze(0)

            # Forward pass 获取 logits
            hidden_states = model(input_ids, position_ids)
            logits = model.compute_logits(hidden_states)

            # Greedy sample: 选择 logits 最大的 token
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # 解码生成的 token
            generated_text = tokenizer.decode(next_token[0])
            pred = ""
            for char in generated_text:
                if char.upper() in "ABCD":
                    pred = char.upper()
                    break

            if pred == true_answer:
                correct += 1

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{total} samples...")

    elapsed = time.time() - t
    accuracy = correct / total * 100
    throughput = total / elapsed

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")
    return accuracy


def compute_mmlu_accuracy_direct(model, tokenizer, num_samples=1000):
    """
    直接使用模型计算 MMLU（用于 direct 方法）
    """
    print("计算 MMLU Accuracy (direct method)...")
    dataset = list(load_dataset("cais/mmlu", "all", split="test"))
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[:num_samples]
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")

    def format_example(question, choices, answer):
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{'ABCD'[i]}. {choice}\n"
        prompt += f"Answer: {answer}\n\n"
        return prompt

    few_shot_prompt = "The following are multiple choice questions (with answers).\n\n"
    for example in dev_dataset.select(range(5)):
        few_shot_prompt += format_example(
            example["question"], example["choices"], "ABCD"[example["answer"]]
        )

    correct = 0
    total = 0

    with torch.no_grad():
        for example in dataset:
            prompt = few_shot_prompt + format_example(
                example["question"], example["choices"], ""
            ).replace("Answer: \n", "Answer:")

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            outputs = model(input_ids, torch.arange(input_ids.size(1), device="cuda").unsqueeze(0))
            logits = model.compute_logits(outputs)

            # 找到ABCD对应的token
            choices_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in "ABCD"]
            choices_logits = [logits[0, -1, cid].item() for cid in choices_ids]
            pred = "ABCD"[choices_logits.index(max(choices_logits))]
            true_answer = "ABCD"[example["answer"]]

            if pred == true_answer:
                correct += 1
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy


def main():
    model_path = "/root/Qwen3-1.7B"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "INT8_Per_Tensor_Fake"
    method = sys.argv[2] if len(sys.argv) > 2 else "direct"  # direct, generate, both

    # 量化配置映射
    quant_configs = {
        "INT8_Per_Tensor_Fake": {
            "quant_type": "per_tensor",
            "linear_dtype": torch.int8,
            "group_size": None,
            "fake_quant": lambda x: fake_per_tensor_quant(x, torch.int8)
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

    print(f"Testing MMLU for config: {config_name}, method: {method}")

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

    # 保存结果
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    # 根据 method 运行测试
    if method in ["direct", "both"]:
        print("\n=== Running Direct Method ===")
        accuracy_direct = compute_mmlu_accuracy_direct(model, tokenizer, num_samples=1000)
        print(f"MMLU Accuracy (direct): {accuracy_direct:.2f}%")

        result_file = os.path.join(result_dir, f"task_31_mmlu_{config_name}_direct.json")
        with open(result_file, "w") as f:
            json.dump({
                "config": config_name,
                "method": "direct",
                "accuracy": accuracy_direct
            }, f)
        print(f"Result saved to: {result_file}")

    if method in ["generate", "both"]:
        print("\n=== Running Generate Method ===")
        accuracy_generate = compute_mmlu_accuracy_generate(model, tokenizer, num_samples=1000)
        print(f"MMLU Accuracy (generate): {accuracy_generate:.2f}%")

        result_file = os.path.join(result_dir, f"task_31_mmlu_{config_name}_generate.json")
        with open(result_file, "w") as f:
            json.dump({
                "config": config_name,
                "method": "generate",
                "accuracy": accuracy_generate
            }, f)
        print(f"Result saved to: {result_file}")

    del model
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
