"""
Task 3.1: Fake Quantization MMLU 测试脚本（使用LLM接口）
支持所有量化配置，BF16 baseline = 50.60%
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

from nanovllm import LLM, SamplingParams
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.quantization import (
    fake_per_tensor_quant,
    fake_per_row_quant,
    fake_per_group_quant,
    apply_weight_fake_quant,
)


def compute_mmlu_accuracy_llm(llm, tokenizer, num_samples=1000):
    """
    使用LLM接口计算MMLU准确率（与test_mmlu.py完全一致）
    BF16 baseline应该得到50.60%
    """
    print("计算 MMLU Accuracy (使用LLM接口)...")
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

    # 设置采样参数（与test_mmlu.py完全一致）
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    # 批量推理并计时
    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    # 计算准确率
    correct = 0
    for output, true_answer in zip(outputs, answers):
        pred = output["text"].strip()[0].upper() if output["text"] else ""
        if pred == true_answer:
            correct += 1

    accuracy = correct / len(answers) * 100
    throughput = len(prompts) / elapsed

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(answers)})")
    print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")
    return accuracy


def compute_mmlu_accuracy_direct_model(model, tokenizer, num_samples=1000):
    """
    使用直接forward计算MMLU准确率（适用于fake quantization）
    """
    print("计算 MMLU Accuracy (直接forward方式)...")
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

            # Greedy sample
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # 解码生成的 token
            generated_text = tokenizer.decode(next_token[0])
            generated_text = generated_text.strip()
            pred = generated_text[0].upper() if len(generated_text) > 0 else ""

            if pred == true_answer:
                correct += 1

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{total} samples...")

    elapsed = time.time() - t
    accuracy = correct / total * 100
    throughput = total / elapsed

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")
    return accuracy


def main():
    model_path = "/root/Qwen3-1.7B"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "BF16"

    # 量化配置映射
    quant_configs = {
        "BF16": {
            "quant_type": None,
            "fake_quant": None
        },
        "INT8_Per_Tensor_Fake": {
            "quant_type": "per_tensor",
            "use_llm": False,  # Fake quantization使用直接forward
            "fake_quant": lambda x: fake_per_tensor_quant(x, torch.int8)
        },
        "INT8_Per_Row_Fake": {
            "quant_type": "per_row",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_row_quant(x, torch.int8)
        },
        "INT8_Per_Group_64_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 64, torch.int8)
        },
        "INT8_Per_Group_128_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 128, torch.int8)
        },
        "INT8_Per_Group_256_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 256, torch.int8)
        },
        "INT8_Per_Group_512_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 512, torch.int8)
        },
        "FP8_Per_Tensor_Fake": {
            "quant_type": "per_tensor",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_tensor_quant(x, torch.float8_e4m3fn)
        },
        "FP8_Per_Row_Fake": {
            "quant_type": "per_row",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_row_quant(x, torch.float8_e4m3fn)
        },
        "FP8_Per_Group_64_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 64, torch.float8_e4m3fn)
        },
        "FP8_Per_Group_128_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 128, torch.float8_e4m3fn)
        },
        "FP8_Per_Group_256_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 256, torch.float8_e4m3fn)
        },
        "FP8_Per_Group_512_Fake": {
            "quant_type": "per_group",
            "use_llm": False,
            "fake_quant": lambda x: fake_per_group_quant(x, 512, torch.float8_e4m3fn)
        },
    }

    print(f"Testing MMLU for config: {config_name}")

    config = quant_configs.get(config_name)
    if config is None:
        print(f"Unknown config: {config_name}")
        sys.exit(1)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config["use_llm"]:
        # BF16使用LLM接口（与test_mmlu.py完全一致）
        print("Using LLM interface (enforce_eager=False)")
        llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
        accuracy = compute_mmlu_accuracy_llm(llm, None, num_samples=1000)
        del llm
    else:
        # Fake quantization使用直接forward
        print("Loading model for fake quantization...")

        # 使用动态端口避免冲突
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()

        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=1, rank=0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_config = Qwen3Config.from_pretrained(model_path)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        model = Qwen3ForCausalLM(hf_config, simple_attention=True)
        load_model(model, model_path)

        # 应用伪量化
        print(f"Applying fake quantization: {config_name}")
        apply_weight_fake_quant(model, config["fake_quant"])
        print(f"Applied fake {config['quant_type']} quantization")

        model.eval()
        accuracy = compute_mmlu_accuracy_direct_model(model, tokenizer, num_samples=1000)

        del model
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        dist.destroy_process_group()

    # 保存结果
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"task_31_mmlu_{config_name}_generate.json")
    with open(result_file, "w") as f:
        json.dump({
            "config": config_name,
            "method": "generate",
            "accuracy": accuracy
        }, f)
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
