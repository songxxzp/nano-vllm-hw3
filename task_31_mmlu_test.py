"""
Task 3.1: MMLU 测试脚本
使用真正的量化方法测试精度
支持 generate() 和 direct() 两种测试方法
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
from nanovllm import LLM


def compute_mmlu_accuracy_with_llm(llm, num_samples=1000):
    """
    使用 LLM.generate() 计算 MMLU 5-shot 准确率
    严格参照 test_mmlu.py
    """
    print("计算 MMLU Accuracy (generate method)...")
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
    prompts = []
    answers = []

    few_shot_prompt = "The following are multiple choice questions (with answers).\n\n"
    for example in dev_dataset.select(range(5)):
        few_shot_prompt += format_example(
            example["question"], example["choices"], "ABCD"[example["answer"]]
        )

    for example in dataset:
        prompt = few_shot_prompt + format_example(
            example["question"], example["choices"], ""
        ).replace("Answer: \n", "Answer:")
        prompts.append(prompt)
        answers.append("ABCD"[example["answer"]])

    # 设置采样参数：只生成1个token
    from nanovllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    # 批量推理并计时
    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    # 计算准确率
    correct = 0
    for output, true_answer in zip(outputs, answers):
        generated_text = output["text"].strip()
        # 提取生成的第一个非空字符，应该是 A/B/C/D
        pred = ""
        for char in generated_text:
            if char.upper() in "ABCD":
                pred = char.upper()
                break
        if not pred and generated_text:
            # 如果没找到ABCD，尝试取第一个字符
            pred = generated_text[0].upper() if generated_text[0].upper() in "ABCD" else ""
        if pred == true_answer:
            correct += 1

    accuracy = correct / len(answers) * 100
    throughput = len(prompts) / elapsed

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")
    return accuracy


def compute_mmlu_accuracy_direct(model, tokenizer, num_samples=1000):
    """
    直接使用模型计算 MMLU（用于非 LLM 模式）
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
    config_name = sys.argv[1] if len(sys.argv) > 1 else "BF16"
    method = sys.argv[2] if len(sys.argv) > 2 else "auto"  # auto, generate, direct

    # 量化配置映射
    quant_configs = {
        "BF16": {"quant_type": None, "linear_dtype": torch.bfloat16, "group_size": None, "use_llm": False},
        "INT8_Per_Tensor": {"quant_type": "per_tensor", "linear_dtype": torch.int8, "group_size": None, "use_llm": False},
        "INT8_Per_Row": {"quant_type": "per_row", "linear_dtype": torch.int8, "group_size": None, "use_llm": True},
        "INT8_Per_Group_64": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 64, "use_llm": False},
        "INT8_Per_Group_128": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 128, "use_llm": False},
        "INT8_Per_Group_256": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 256, "use_llm": False},
        "INT8_Per_Group_512": {"quant_type": "per_group", "linear_dtype": torch.int8, "group_size": 512, "use_llm": False},
        "FP8_Per_Tensor": {"quant_type": "per_tensor", "linear_dtype": torch.float8_e4m3fn, "group_size": None, "use_llm": False},
        "FP8_Per_Row": {"quant_type": "per_row", "linear_dtype": torch.float8_e4m3fn, "group_size": None, "use_llm": True},
        "FP8_Per_Group_64": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 64, "use_llm": False},
        "FP8_Per_Group_128": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 128, "use_llm": False},
        "FP8_Per_Group_256": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 256, "use_llm": False},
        "FP8_Per_Group_512": {"quant_type": "per_group", "linear_dtype": torch.float8_e4m3fn, "group_size": 512, "use_llm": False},
    }

    print(f"Testing MMLU for config: {config_name}")

    config = quant_configs.get(config_name)
    if config is None:
        print(f"Unknown config: {config_name}")
        sys.exit(1)

    quant_type = config["quant_type"]
    linear_dtype = config["linear_dtype"]
    group_size = config["group_size"]
    use_llm = config["use_llm"]

    # 决定使用哪种方法
    if method == "auto":
        method = "generate" if use_llm else "direct"
    elif method == "generate" and not use_llm:
        print(f"Warning: generate method requested but use_llm=False, forcing direct method")
        method = "direct"
    elif method == "direct" and use_llm:
        print(f"Note: Running direct method even though use_llm=True")

    accuracy = None
    actual_method = method

    if method == "generate":
        # 使用 LLM + 真正的 per-row 量化
        print(f"Using LLM.generate() with quant_type={quant_type}, linear_dtype={linear_dtype}")
        llm = LLM(
            model_path,
            enforce_eager=True,
            max_model_len=4096,
            linear_dtype=linear_dtype,
            quant_type=quant_type,
        )
        accuracy = compute_mmlu_accuracy_with_llm(llm, num_samples=1000)
        print(f"MMLU Accuracy: {accuracy:.2f}%")
        del llm
    else:
        # 使用直接模型 + 量化
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        accuracy = compute_mmlu_accuracy_direct(model, tokenizer, num_samples=1000)
        print(f"MMLU Accuracy: {accuracy:.2f}%")

        del model
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        dist.destroy_process_group()

    # 保存结果
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"task_31_mmlu_{config_name}_{actual_method}.json")
    with open(result_file, "w") as f:
        json.dump({
            "config": config_name,
            "method": actual_method,
            "accuracy": accuracy
        }, f)
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
