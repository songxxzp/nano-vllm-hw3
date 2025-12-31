import os
import random
import time
import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm import LLM, SamplingParams
from nanovllm.utils.loader import load_model
from nanovllm.utils.quantization import (
    fake_per_tensor_quant,
    fake_per_row_quant,
    fake_per_group_quant,
    per_tensor_quant,
    per_row_quant,
    per_group_quant,
)

from test_mmlu import format_example


def mmlu(args, linear_dtype, quant_type, weight_quant_fn):
    # 加载模型
    path = os.path.expanduser("./Qwen3-1.7B/")
    if args.real:
        llm = LLM(path, enforce_eager=False, max_model_len=4096, quant_type=quant_type, linear_dtype=linear_dtype, group_size=args.group_size)
    else:
        llm = LLM(path, enforce_eager=False, max_model_len=4096, weight_quant_fn=weight_quant_fn, linear_dtype=linear_dtype)

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


def wikitext(args, linear_dtype, quant_type, weight_quant_fn):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 加载模型
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    path = os.path.expanduser("./Qwen3-1.7B/")
    hf_config = Qwen3Config.from_pretrained(path)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")
    model = Qwen3ForCausalLM(hf_config, simple_attention=True)
    load_model(model, path)


    if weight_quant_fn is not None:
        from nanovllm.utils.quantization import apply_weight_fake_quant
        apply_weight_fake_quant(model, weight_quant_fn)
    elif quant_type is not None:
        from nanovllm.utils.quantization import (
            apply_tensor_quant,
            apply_per_row_quant,
            apply_group_quant,
        )
        if quant_type == "per_tensor":
            apply_tensor_quant(model, linear_dtype)
        elif quant_type == "per_row":
            apply_per_row_quant(model, linear_dtype)
        elif quant_type == "per_group":
            apply_group_quant(model, linear_dtype, args.group_size)
    elif linear_dtype != torch.bfloat16:
        from nanovllm.utils.quantization import apply_per_row_quant
        apply_per_row_quant(model, linear_dtype)

    model.eval()
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)

    # 加载 tokenizer 和数据集
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 计算 PPL
    total_loss = 0.0
    total_tokens = 0
    max_length = 4096
    texts = dataset["text"][:100]

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
    print(f"Perplexity: {ppl:.2f}")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', default=False)
    parser.add_argument('--quant', type=str, default=None, choices=['tensor', 'row', 'group'])
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'int8', 'fp8'])
    parser.add_argument('--group-size', type=int, default=64, choices=[64, 128, 256, 512])
    parser.add_argument('--test', type=str, default='mmlu', choices=['mmlu', 'ppl'])
    parser.add_argument('--save', type=str, default='result.json')

    args = parser.parse_args()

    print(f"test: {args.test}, real: {args.real}, quant: {args.quant}, dtype: {args.dtype}" + str(f"group size: {args.group_size}" if args.quant == "group" else ""))

    quant_type = None
    linear_dtype = torch.bfloat16
    weight_quant_fn = None

    if args.dtype == 'bf16':
        linear_dtype = torch.bfloat16
    elif args.dtype == 'int8':
        linear_dtype = torch.int8
    elif args.dtype == 'fp8':
        linear_dtype = torch.float8_e4m3fn

    if args.real:
        if args.quant == "tensor":
            quant_type = "per_tensor"
        elif args.quant == "row":
            quant_type = "per_row"
        elif args.quant == "group":
            quant_type = "per_group"
    else:
        if args.quant == "tensor":
            weight_quant_fn = partial(fake_per_tensor_quant, dtype=linear_dtype)
        elif args.quant == "row":
            weight_quant_fn = partial(fake_per_row_quant, dtype=linear_dtype)
        elif args.quant == "group":
            weight_quant_fn = partial(fake_per_group_quant, group_size=args.group_size, dtype=linear_dtype)

    if args.test == "mmlu":
        mmlu(args, linear_dtype, quant_type, weight_quant_fn)
    else:
        wikitext(args, linear_dtype, quant_type, weight_quant_fn)


if __name__ == "__main__":
    main()
