import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


def main():
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


if __name__ == "__main__":
    main()
