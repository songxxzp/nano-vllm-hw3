import os
import random
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

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
    apply_tensor_quant,
    apply_per_row_quant,
    apply_group_quant,
    apply_weight_fake_quant,
    apply_quant_torchao,
)

from test_mmlu import format_example


class ExperimentResults:
    """管理实验结果的保存和加载（JSONL格式）"""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.results = {}
        self._load()

    def _load(self):
        """从JSONL文件加载结果"""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        # 使用config作为唯一标识
                        config_key = self._get_config_key(result)
                        self.results[config_key] = result
            print(f"从 {self.filepath} 加载了 {len(self.results)} 条结果")

    def _get_config_key(self, result):
        """根据结果生成唯一的配置key"""
        # 包含所有配置相关的字段
        key_parts = [
            result.get('test', ''),
            result.get('real', False),
            result.get('quant', ''),
            result.get('dtype', ''),
            str(result.get('group_size', ''))
        ]
        return '|'.join(str(p) for p in key_parts)

    def save_or_update(self, result):
        """保存新结果或更新已有结果"""
        config_key = self._get_config_key(result)

        # 检查是否已存在
        is_new = config_key not in self.results

        # 添加时间戳
        result['timestamp'] = datetime.now().isoformat()

        # 更新内存中的结果
        self.results[config_key] = result

        # 写入JSONL文件（追加或更新）
        self._write_to_jsonl()

        if is_new:
            print(f"新结果已保存到 {self.filepath}")
        else:
            print(f"结果已更新到 {self.filepath}")

    def _write_to_jsonl(self):
        """将所有结果写入JSONL文件"""
        # 创建临时文件
        temp_file = self.filepath.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            for result in self.results.values():
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 替换原文件
        temp_file.replace(self.filepath)

    def exists(self, test, real, quant, dtype, group_size=None):
        """检查某个配置是否已存在结果"""
        key_parts = [test, real, quant, dtype, str(group_size)]
        config_key = '|'.join(str(p) for p in key_parts)
        return config_key in self.results


def mmlu(args, linear_dtype, quant_type, weight_quant_fn, results_manager):
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

    # 保存结果
    result = {
        'test': 'mmlu',
        'real': args.real,
        'quant': args.quant,
        'dtype': args.dtype,
        'group_size': args.group_size if args.quant == 'group' else None,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(answers),
        'time': elapsed,
        'throughput': throughput
    }
    results_manager.save_or_update(result)


def wikitext(args, linear_dtype, quant_type, weight_quant_fn, results_manager):
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
        apply_weight_fake_quant(model, weight_quant_fn)
    elif quant_type is not None:
        if quant_type == "per_tensor":
            apply_tensor_quant(model, linear_dtype)
        elif quant_type == "per_row":
            apply_per_row_quant(model, linear_dtype)
        elif quant_type == "per_group":
            apply_group_quant(model, linear_dtype, args.group_size)
        elif quant_type == "smoothquant":
            apply_quant_torchao(model, linear_dtype=linear_dtype)
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

    # 保存结果
    result = {
        'test': 'ppl',
        'real': args.real,
        'quant': args.quant,
        'dtype': args.dtype,
        'group_size': args.group_size if args.quant == 'group' else None,
        'perplexity': ppl,
        'total_tokens': total_tokens
    }
    results_manager.save_or_update(result)


def throughput(args, linear_dtype, quant_type, weight_quant_fn, results_manager):
    """测试 prefilling 和 decoding 的吞吐量（使用假数据）"""
    # 加载模型
    path = os.path.expanduser("./Qwen3-1.7B/")
    if args.real:
        llm = LLM(path, enforce_eager=False, max_model_len=4096,
                 quant_type=quant_type, linear_dtype=linear_dtype, group_size=args.group_size)
    else:
        llm = LLM(path, enforce_eager=False, max_model_len=4096,
                 weight_quant_fn=weight_quant_fn, linear_dtype=linear_dtype)

    # 生成假数据 - 随机token序列
    random.seed(42)
    num_samples = args.num_samples
    prompt_length = args.prompt_length
    generate_prompt_length = args.generate_prompt_length
    generate_length = args.generate_length

    # 生成随机prompts (使用随机token id构建)
    prompts = []
    for _ in range(num_samples):
        # 生成随机文本作为prompt（简单重复相同长度的文本）
        fake_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10 + 1)
        fake_text = fake_text[:prompt_length * 3]  # 粗略估计
        prompts.append(fake_text)

    # 测试 prefilling 阶段 (处理长prompt)
    print(f"测试 prefilling 吞吐量: {num_samples} samples, prompt长度 ~{prompt_length} tokens")
    sampling_params_prefill = SamplingParams(temperature=0.0, max_tokens=1)

    t = time.time()
    outputs_prefill = llm.generate(prompts, sampling_params_prefill, use_tqdm=True)
    elapsed_prefill = time.time() - t

    prefill_throughput = num_samples / elapsed_prefill
    prefill_tokens_per_sec = (num_samples * prompt_length) / elapsed_prefill

    print(f"\nPrefilling 吞吐量:")
    print(f"  时间: {elapsed_prefill:.2f}s")
    print(f"  Samples/s: {prefill_throughput:.2f}")
    print(f"  Tokens/s: {prefill_tokens_per_sec:.2f}")

    # 测试 decoding 阶段 (生成多个token)
    print(f"测试 decoding 吞吐量: {num_samples} samples, 生成 {generate_length} tokens")
    sampling_params_decode = SamplingParams(temperature=0.0, max_tokens=generate_length)

    # 使用较短的prompt用于decoding测试
    short_prompts = [p[:generate_prompt_length] for p in prompts[:num_samples]]

    t = time.time()
    outputs_decode = llm.generate(short_prompts, sampling_params_decode, use_tqdm=True)
    elapsed_decode = time.time() - t

    total_generated_tokens = sum(len(output["token_ids"]) for output in outputs_decode)
    decode_throughput = total_generated_tokens / elapsed_decode

    print(f"\nDecoding 吞吐量:")
    print(f"  时间: {elapsed_decode:.2f}s")
    print(f"  生成tokens: {total_generated_tokens}")
    print(f"  Tokens/s: {decode_throughput:.2f}")

    # 保存结果
    result = {
        'test': 'throughput',
        'real': args.real,
        'quant': args.quant,
        'dtype': args.dtype,
        'group_size': args.group_size if args.quant == 'group' else None,
        'num_samples': num_samples,
        'prompt_length': prompt_length,
        'generate_prompt_length': generate_prompt_length,
        'generate_length': generate_length,
        'prefill_time': elapsed_prefill,
        'prefill_samples_per_sec': prefill_throughput,
        'prefill_tokens_per_sec': prefill_tokens_per_sec,
        'decode_time': elapsed_decode,
        'decode_tokens_per_sec': decode_throughput,
        'total_generated_tokens': total_generated_tokens
    }
    results_manager.save_or_update(result)


def matmul_test(args, linear_dtype, quant_type, weight_quant_fn, results_manager):
    """测试矩阵乘法的伪量化误差"""
    import logging
    log = logging.getLogger(__name__)

    # 解析矩阵形状
    M, N, K = args.matmul_shape
    num_tests = args.num_tests

    log.info(f"测试矩阵乘法: M={M}, N={N}, K={K}, 测试次数={num_tests}")

    max_errors = []
    mean_errors = []
    relative_errors = []

    for test_idx in range(num_tests):
        # 生成随机矩阵
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        # 计算参考结果 (FP32)
        with torch.no_grad():
            C_ref = A @ B

        # 转换为目标数据类型
        if linear_dtype == torch.bfloat16:
            A_target = A.to(torch.bfloat16)
            B_target = B.to(torch.bfloat16)
        elif linear_dtype == torch.int8:
            # INT8 需要量化
            from nanovllm.utils.quantization import fake_per_tensor_quant
            A_target = fake_per_tensor_quant(A, dtype=torch.int8)
            B_target = fake_per_tensor_quant(B, dtype=torch.int8)
        elif linear_dtype == torch.float8_e4m3fn:
            from nanovllm.utils.quantization import fake_per_tensor_quant
            A_target = fake_per_tensor_quant(A, dtype=torch.float8_e4m3fn)
            B_target = fake_per_tensor_quant(B, dtype=torch.float8_e4m3fn)
        else:
            A_target = A
            B_target = B

        # 应用伪量化函数
        if weight_quant_fn is not None:
            # 对矩阵B应用伪量化
            B_reshaped = B_target.T.unsqueeze(0)  # (1, N, K)
            B_quant = weight_quant_fn(B_reshaped)
            B_target = B_quant.squeeze(0).T  # (K, N)

        # 计算量化后的结果
        with torch.no_grad():
            if linear_dtype == torch.int8 or linear_dtype == torch.float8_e4m3fn:
                # INT8/FP8需要先转换回FP32进行计算
                A_compute = A_target.to(torch.float32)
                B_compute = B_target.to(torch.float32)
            else:
                A_compute = A_target
                B_compute = B_target
            C_quant = A_compute @ B_compute

        # 计算误差
        error = torch.abs(C_quant - C_ref)
        max_error = error.max().item()
        mean_error = error.mean().item()
        relative_error = (error / (torch.abs(C_ref) + 1e-8)).mean().item()

        max_errors.append(max_error)
        mean_errors.append(mean_error)
        relative_errors.append(relative_error)

    # 统计结果
    avg_max_error = sum(max_errors) / len(max_errors)
    avg_mean_error = sum(mean_errors) / len(mean_errors)
    avg_relative_error = sum(relative_errors) / len(relative_errors)

    print(f"\n矩阵乘法误差统计 ({num_tests} 次测试平均):")
    print(f"  最大误差: {avg_max_error:.6f}")
    print(f"  平均误差: {avg_mean_error:.6f}")
    print(f"  相对误差: {avg_relative_error:.6f}")

    # 保存结果
    result = {
        'test': 'matmul',
        'real': args.real,
        'quant': args.quant,
        'dtype': args.dtype,
        'group_size': args.group_size if args.quant == 'group' else None,
        'matmul_shape': [M, N, K],
        'num_tests': num_tests,
        'avg_max_error': avg_max_error,
        'avg_mean_error': avg_mean_error,
        'avg_relative_error': avg_relative_error
    }
    results_manager.save_or_update(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', default=False)
    parser.add_argument('--quant', type=str, default=None, choices=['tensor', 'row', 'group', 'smooth'])
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'int8', 'fp8'])
    parser.add_argument('--group-size', type=int, default=64, choices=[64, 128, 256, 512])
    parser.add_argument('--test', type=str, default='mmlu', choices=['mmlu', 'ppl', 'throughput', 'matmul'])
    parser.add_argument('--save', type=str, default='results.jsonl', help='Path to JSONL file for saving results')

    # Throughput test arguments
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples for throughput test')
    parser.add_argument('--prompt-length', type=int, default=1000, help='Prompt length in tokens for throughput test')
    parser.add_argument('--generate-prompt-length', type=int, default=1, help='Prompt length in tokens for throughput test')
    parser.add_argument('--generate-length', type=int, default=1000, help='Number of tokens to generate for throughput test')

    # Matmul test arguments
    parser.add_argument('--matmul-shape', type=int, nargs=3, default=[4096, 4096, 4096],
                       help='Matrix shape for matmul test (M N K)')
    parser.add_argument('--num-tests', type=int, default=10, help='Number of tests for matmul error evaluation')

    args = parser.parse_args()

    print(f"test: {args.test}, real: {args.real}, quant: {args.quant}, dtype: {args.dtype}" + str(f" group size: {args.group_size}" if args.quant == "group" else ""))

    # 创建结果管理器
    results_manager = ExperimentResults(args.save)

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
        elif args.quant == "smooth":
            quant_type = "smoothquant"
    else:
        if args.quant == "tensor":
            weight_quant_fn = partial(fake_per_tensor_quant, dtype=linear_dtype)
        elif args.quant == "row":
            weight_quant_fn = partial(fake_per_row_quant, dtype=linear_dtype)
        elif args.quant == "group":
            weight_quant_fn = partial(fake_per_group_quant, group_size=args.group_size, dtype=linear_dtype)
        elif args.quant == "smooth":
            raise AssertionError("smoothquant must be real(add --real)")

    if args.test == "mmlu":
        mmlu(args, linear_dtype, quant_type, weight_quant_fn, results_manager)
    elif args.test == "ppl":
        wikitext(args, linear_dtype, quant_type, weight_quant_fn, results_manager)
    elif args.test == "throughput":
        throughput(args, linear_dtype, quant_type, weight_quant_fn, results_manager)
    elif args.test == "matmul":
        matmul_test(args, linear_dtype, quant_type, weight_quant_fn, results_manager)


if __name__ == "__main__":
    main()
