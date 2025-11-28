import os
import random
import time

import torch
from datasets import load_dataset

from nanovllm import LLM, SamplingParams


def format_example(question, choices, answer):
    """格式化单个样本为 prompt"""
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {choice}\n"
    prompt += f"Answer: {answer}\n\n"
    return prompt


def main():
    # 加载模型
    path = os.path.expanduser("./Qwen3-1.7B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 加载 MMLU 数据集
    dataset = list(load_dataset("cais/mmlu", "all", split="test"))
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[:1000]
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")

    # 构建 5-shot prompts
    prompts = []
    answers = []

    # 取前5个 dev 样本作为 few-shot examples
    few_shot_prompt = "The following are multiple choice questions (with answers).\n\n"
    for example in dev_dataset.select(range(5)):
        few_shot_prompt += format_example(
            example["question"], example["choices"], "ABCD"[example["answer"]]
        )

    # 为每个测试样本构建 prompt
    for example in dataset:
        prompt = few_shot_prompt + format_example(
            example["question"], example["choices"], ""
        ).replace("Answer: \n", "Answer:")
        prompts.append(prompt)
        answers.append("ABCD"[example["answer"]])

    # 设置采样参数：只生成1个token
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

    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(answers)})")
    print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/s")


if __name__ == "__main__":
    main()
