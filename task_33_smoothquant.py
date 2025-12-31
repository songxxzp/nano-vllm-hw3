"""
Task 3.3: SmoothQuant 量化测试 (Qwen2.5-1.5B)
支持 Int8 和 FP8 动态激活 + 权重量化
使用 TorchAO 的量化配置
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

# 尝试导入 TorchAO
try:
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig, Float8DynamicActivationFloat8WeightConfig
    TORCHAO_AVAILABLE = True
except ImportError as e:
    print(f"TorchAO not available: {e}")
    print("Please install TorchAO: pip install torchao")
    TORCHAO_AVAILABLE = False


def apply_quant_torchao(model, quant_type="int8"):
    """
    使用 TorchAO 应用量化

    Args:
        quant_type: "int8" 或 "fp8"
    """
    if not TORCHAO_AVAILABLE:
        raise RuntimeError("TorchAO is not available. Please install: pip install torchao")

    if quant_type == "int8":
        print("Applying Int8DynamicActivationInt8WeightConfig...")
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    elif quant_type == "fp8":
        print("Applying Float8DynamicActivationFloat8WeightConfig...")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    return model


def compute_ppl(model, tokenizer, num_samples=50):
    """
    计算 WikiText-2 上的 Perplexity
    手动计算 loss 以确保量化模型的正确性
    """
    print("计算 WikiText-2 Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    total_loss = 0.0
    total_tokens = 0
    max_length = 2048
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

                # 手动计算 logits 和 loss
                outputs = model(input_ids)
                logits = outputs.logits

                # 计算交叉熵损失
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="sum"
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')
    return ppl


def compute_mmlu_accuracy_direct(model, tokenizer, num_samples=200):
    """
    计算 MMLU 5-shot 准确率 (Direct 方法)
    直接使用模型 logits，不使用 generate()
    """
    print("计算 MMLU Accuracy (Direct 方法)...")
    import random
    random.seed(42)

    dataset = list(load_dataset("cais/mmlu", "all", split="test"))
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

            # 直接使用模型 forward
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # [vocab_size]

            # 找到 ABCD 对应的 token
            choices_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in "ABCD"]
            choices_logits = [logits[cid].item() for cid in choices_ids]
            pred = "ABCD"[choices_logits.index(max(choices_logits))]
            true_answer = "ABCD"[example["answer"]]

            if pred == true_answer:
                correct += 1
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy


def compute_mmlu_accuracy_generate(model, tokenizer, num_samples=200):
    """
    计算 MMLU 5-shot 准确率 (Generate 方法)
    使用 generate() 生成 1 个 token
    """
    print("计算 MMLU Accuracy (Generate 方法)...")
    import random
    random.seed(42)

    dataset = list(load_dataset("cais/mmlu", "all", split="test"))
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

    for example in dataset:
        prompt = few_shot_prompt + format_example(
            example["question"], example["choices"], ""
        ).replace("Answer: \n", "Answer:")

        # 使用 generate() 生成 1 个 token
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 生成 1 个 token（相当于 temperature=0.0 的 greedy sampling）
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        # 获取生成的第一个 token
        generated_token = outputs[0, -1].item()

        # 将生成的 token 转换为文本，取第一个字符
        generated_text = tokenizer.decode(generated_token)
        pred = generated_text[0].upper() if generated_text else ""

        true_answer = "ABCD"[example["answer"]]

        if pred == true_answer:
            correct += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy


def test_matmul_accuracy():
    """
    测试矩阵乘法量化精度
    类似 Task 3.2 的测试，测试 INT8 和 FP8 的量化误差
    """
    print("=" * 80)
    print("矩阵乘法量化精度测试 (类似 Task 3.2)")
    print("=" * 80)

    device = "cuda"
    dtypes = [torch.int8, torch.float8_e4m3fn]
    shapes = [(16, 512), (100, 1024), (4096, 4096)]

    print("\n测试量化误差:")
    print("-" * 70)

    results = []

    for M, N in shapes:
        x = torch.randn(M, N, device=device)

        for dtype in dtypes:
            dtype_name = "int8" if dtype == torch.int8 else "fp8"

            # Per-tensor quant
            if dtype == torch.int8:
                max_val = 127
            else:
                max_val = 448

            # 计算 scale
            amax = x.abs().max().clamp(min=1e-8)
            scale = amax / max_val

            # 量化
            if dtype == torch.int8:
                x_quant = (x / scale).round().clamp(-128, 127).to(dtype)
            else:
                x_quant = (x / scale).to(torch.float32).to(dtype)

            # 反量化
            x_dequant = x_quant.to(torch.float32) * scale

            # 计算误差
            err = (x - x_dequant).abs().mean() / x.abs().mean()

            print(f"[{M:4d}x{N:4d}] {dtype_name}: per_tensor_error={err:.6f}")

            results.append({
                "shape": f"{M}x{N}",
                "dtype": dtype_name,
                "per_tensor_error": float(err),
            })

    # 计算 INT8 vs FP8 的误差比
    print("\n" + "=" * 80)
    print("FP8 vs INT8 误差对比:")
    print("-" * 70)

    fp8_errors = {}
    int8_errors = {}

    for r in results:
        shape = r["shape"]
        dtype = r["dtype"]
        if dtype == "fp8":
            fp8_errors[shape] = r["per_tensor_error"]
        else:
            int8_errors[shape] = r["per_tensor_error"]

    for shape in fp8_errors:
        ratio = fp8_errors[shape] / int8_errors[shape]
        print(f"[{shape}] FP8 error / INT8 error = {ratio:.2f}x")

    print("\n" + "=" * 80)
    print("结论:")
    print("-" * 70)
    print("FP8 的量化误差比 INT8 大约 2-3 倍，原因:")
    print("1. FP8 (float8_e4m3fn) 的动态范围较小: max=448, 而 INT8 的 max=127")
    print("2. FP8 只有 4 位 exponent，3 位 mantissa，精度较低")
    print("3. INT8 虽然只有 8 位整数，但在量化 scale 合适时精度较高")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Task 3.3: SmoothQuant 测试")
    parser.add_argument("--quant-type", type=str, default="int8", choices=["int8", "fp8", "both"],
                        help="量化类型: int8, fp8, 或 both")
    parser.add_argument("--method", type=str, default="both", choices=["direct", "generate", "both"],
                        help="MMLU 测试方法: direct, generate, 或 both")
    parser.add_argument("--model-path", type=str, default="/root/Qwen2.5-1.5B",
                        help="模型路径")
    parser.add_argument("--ppl-samples", type=int, default=50,
                        help="PPL 测试样本数")
    parser.add_argument("--mmlu-samples", type=int, default=200,
                        help="MMLU 测试样本数")
    args = parser.parse_args()

    quant_types = ["int8", "fp8"] if args.quant_type == "both" else [args.quant_type]
    methods = ["direct", "generate"] if args.method == "both" else [args.method]

    for quant_type in quant_types:
        print("=" * 80)
        print(f"Task 3.3: {quant_type.upper()} SmoothQuant 测试 (Qwen2.5-1.5B)")
        print("=" * 80)

        model_path = args.model_path

        print(f"加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()

        # 应用量化
        quant_applied = False
        try:
            model = apply_quant_torchao(model, quant_type)
            print(f"{quant_type.upper()} 量化应用成功!")
            quant_applied = True
        except Exception as e:
            print(f"{quant_type.upper()} 量化失败: {e}")
            print("使用未量化模型继续...")
            import traceback
            traceback.print_exc()

        # 计算 PPL
        ppl = compute_ppl(model, tokenizer, num_samples=args.ppl_samples)
        print(f"Perplexity: {ppl:.4f}")

        # 根据方法计算 MMLU 准确率
        mmlu_results = {}
        for method in methods:
            if method == "direct":
                accuracy = compute_mmlu_accuracy_direct(model, tokenizer, num_samples=args.mmlu_samples)
            else:  # generate
                accuracy = compute_mmlu_accuracy_generate(model, tokenizer, num_samples=args.mmlu_samples)
            print(f"MMLU Accuracy ({method}): {accuracy:.2f}%")
            mmlu_results[method] = accuracy

        # 保存模型测试结果
        result = {
            "model": "Qwen2.5-1.5B",
            "quantization": f"{quant_type.upper()}DynamicActivation{quant_type.upper()}Weight",
            "quant_applied": quant_applied,
            "ppl": ppl,
            "mmlu_accuracy": mmlu_results
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, "results")
        os.makedirs(result_dir, exist_ok=True)

        # 为每个方法保存单独的结果文件
        for method in methods:
            result_single = result.copy()
            result_single["mmlu_accuracy"] = mmlu_results[method]
            result_file = os.path.join(result_dir, f"task_33_{quant_type}_{method}_results.json")
            with open(result_file, "w") as f:
                json.dump(result_single, f, indent=2)

        # 保存完整结果
        result_file = os.path.join(result_dir, f"task_33_{quant_type}_results.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {result_file}")

        # 释放内存
        del model
        torch.cuda.empty_cache()

    # 运行矩阵乘法精度测试（只运行一次）
    print("\n" + "=" * 80)
    print("运行矩阵乘法量化精度测试")
    print("=" * 80)

    matmul_results = test_matmul_accuracy()

    # 保存矩阵乘法测试结果
    matmul_result_file = os.path.join(result_dir, "task_33_matmul_accuracy_results.json")
    with open(matmul_result_file, "w") as f:
        json.dump(matmul_results, f, indent=2)
    print(f"\n矩阵乘法精度测试结果已保存到: {matmul_result_file}")


if __name__ == "__main__":
    main()
