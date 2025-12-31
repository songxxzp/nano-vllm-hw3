import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from nanovllm.layers.linear import LinearBase
from nanovllm.llm import LLM
from nanovllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM

from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig, Float8DynamicActivationFloat8WeightConfig


@triton.heuristics(values={"PAD_H": lambda args: triton.next_power_of_2(args["H"])})
@triton.jit
def _per_row_quant_kernel(
    X, Y, S, T, H, BLK_M: tl.constexpr, PAD_H: tl.constexpr, FP8: tl.constexpr
):
    bidx = tl.program_id(0)
    x_ptrs = (
        X
        + (tl.arange(0, BLK_M)[:, None] * H + tl.arange(0, PAD_H)[None, :])
        + bidx * BLK_M * H
    )
    s_mask = (tl.arange(0, BLK_M) + bidx * BLK_M) < T
    mask = ((tl.arange(0, PAD_H) < H)[None, :]) & (s_mask[:, None])
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)  # [BLK_M, PAD_H]
    row_max = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-8)  # [BLK_M]
    if FP8:
        scale = row_max / 448
        scale_inv = 448 / row_max

    else:
        scale = row_max / 127
        scale_inv = 127 / row_max

    scaled_x = x * scale_inv[:, None]

    if FP8:
        y = tl.cast(scaled_x, tl.float8e4nv)
    else:
        y = tl.cast(libdevice.round(scaled_x), tl.int8)

    y_ptrs = (
        Y
        + (tl.arange(0, BLK_M)[:, None] * H + tl.arange(0, PAD_H)[None, :])
        + bidx * BLK_M * H
    )
    s_ptrs = S + (tl.arange(0, BLK_M)) + bidx * BLK_M

    tl.store(y_ptrs, y, mask=mask)
    tl.store(s_ptrs, scale, mask=s_mask)


@triton.jit
def _per_row_post_process_mm(
    A,  # [M, K]
    sA,  # [M,]
    B,  # [N, K]
    sB,  # [N,]
    Bias,  # None or [N,]
    C,  # [M, N]
    M,
    N,
    K,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
    FP8: tl.constexpr,
):
    bidy = tl.program_id(0)
    bidx = tl.program_id(1)

    num_tiles = K // BLK_K

    a_block_ptr = tl.make_block_ptr(
        A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(bidx * BLK_M, 0),
        block_shape=(BLK_M, BLK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        B,
        shape=(N, K),
        strides=(K, 1),
        offsets=(bidy * BLK_N, 0),
        block_shape=(BLK_N, BLK_K),
        order=(0, 1),
    )

    if FP8:
        acc = tl.zeros((BLK_M, BLK_N), tl.float32)
    else:
        acc = tl.zeros((BLK_M, BLK_N), tl.int32)

    for _ in range(num_tiles):
        a = tl.load(
            a_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # [BLK_M, BLK_K]
        b = tl.load(
            b_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # [BLK_N, BLK_K]
        c = tl.dot(a, tl.trans(b))
        acc += c

        a_block_ptr = tl.advance(a_block_ptr, (0, BLK_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLK_K))

    acc = tl.cast(acc, tl.float32)
    sa_ptrs = sA + tl.arange(0, BLK_M) + bidx * BLK_M
    sa_mask = (tl.arange(0, BLK_M) + bidx * BLK_M) < M

    sb_ptrs = sB + tl.arange(0, BLK_N) + bidy * BLK_N
    sb_mask = (tl.arange(0, BLK_N) + bidy * BLK_N) < N

    sa = tl.load(sa_ptrs, mask=sa_mask, other=0)  # [BLK_M]
    sb = tl.load(sb_ptrs, mask=sb_mask, other=0)  # [BLK_N]

    c = acc * sa[:, None] * sb[None, :]

    if Bias:
        bias_ptrs = tl.make_block_ptr(
            Bias,
            shape=(N,),
            strides=(1,),
            offsets=(bidy * BLK_N,),
            block_shape=(BLK_N,),
            order=(0,),
        )
        bias = tl.load(bias_ptrs, boundary_check=(0,))  # [BLK_N]
        c += bias[:, None]

    c_block_ptr = tl.make_block_ptr(
        C,
        shape=(M, N),
        strides=(N, 1),
        offsets=(bidx * BLK_M, bidy * BLK_N),
        block_shape=(BLK_M, BLK_N),
        order=(0, 1),
    )

    tl.store(c_block_ptr, tl.cast(c, C.type.element_ty), boundary_check=(0, 1))


@torch.no_grad()
def per_row_quant(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dtype in [torch.int8, torch.float8_e4m3fn]
    assert x.dim() == 2

    qx = torch.empty_like(x, dtype=dtype)
    sx = torch.empty((x.size(0),), dtype=torch.float32, device=x.device)
    BLK_M = (16384) // triton.next_power_of_2(x.size(1))

    grid = (triton.cdiv(x.size(0), BLK_M),)
    _per_row_quant_kernel[grid](
        x, qx, sx, x.size(0), x.size(1), BLK_M=BLK_M, FP8=(dtype == torch.float8_e4m3fn)
    )

    return qx, sx


@torch.no_grad()
def per_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-tensor quantization: quantize entire tensor with single scale.
    Returns (quantized_tensor, scale).
    """
    assert dtype in [torch.int8, torch.float8_e4m3fn]

    if dtype == torch.int8:
        max_val = 127
    else:  # float8_e4m3fn
        max_val = 448

    # Compute scale for entire tensor
    amax = x.abs().max().clamp(min=1e-8)
    scale = amax / max_val

    # Quantize
    if dtype == torch.int8:
        qx = (x / scale).round().clamp(-128, 127).to(dtype)
    else:
        qx = (x / scale).to(torch.float32).to(dtype)

    return qx, scale


@torch.no_grad()
def per_group_quant(
    x: torch.Tensor, group_size: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-group quantization: quantize along K dimension with group_size.
    For weight tensor [out_features, in_features], groups are along in_features.
    Returns (quantized_tensor, scales).
    """
    assert dtype in [torch.int8, torch.float8_e4m3fn]
    assert x.dim() == 2
    assert x.size(1) % group_size == 0, f"K dim ({x.size(1)}) must be divisible by group_size ({group_size})"

    if dtype == torch.int8:
        max_val = 127
    else:  # float8_e4m3fn
        max_val = 448

    original_shape = x.shape
    x = x.float()  # Convert to float for computation

    # Reshape to [out_features, num_groups, group_size]
    num_groups = x.size(1) // group_size
    x = x.reshape(x.size(0), num_groups, group_size)

    # Compute scale per group
    amax = x.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scales = amax / max_val  # [out_features, num_groups, 1]

    # Quantize
    x_scaled = x / scales
    if dtype == torch.int8:
        x_quant = x_scaled.round().clamp(-128, 127).to(dtype)
    else:
        x_quant = x_scaled.to(dtype)

    # Reshape back
    qx = x_quant.reshape(original_shape)
    sx = scales.squeeze(-1)  # [out_features, num_groups]

    return qx, sx


@torch.no_grad()
def per_tensor_matmul(
    x: torch.Tensor,
    qw: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
):
    """
    Per-tensor quantized matrix multiplication.
    x: [..., K] activation (can have arbitrary leading dimensions)
    qw: [N, K] quantized weight
    sw: scale (scalar)
    bias: [N] or None
    """
    # Preserve leading dimensions
    x_dtype = x.dtype
    y_shape = list(x.shape)[:-1] + [qw.shape[0]]
    x = x.view(-1, x.shape[-1])  # [M, K]

    # Dequantize weight and compute
    w = qw.to(torch.float32) * sw
    y = (x.to(torch.float32) @ w.T).to(x_dtype)
    if bias is not None:
        y = y + bias
    return y.view(y_shape)


@torch.no_grad()
def per_group_matmul(
    x: torch.Tensor,
    qw: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    dtype: torch.dtype,
):
    """
    Per-group quantized matrix multiplication.
    x: [..., K] activation (can have arbitrary leading dimensions)
    qw: [N, K] quantized weight
    sw: [N, num_groups] scales
    bias: [N] or None
    group_size: size of each group
    """
    x_dtype = x.dtype
    # Preserve leading dimensions
    y_shape = list(x.shape)[:-1] + [qw.shape[0]]
    x = x.view(-1, x.shape[-1])  # [M, K]

    M, K = x.shape
    N = qw.shape[0]
    num_groups = K // group_size

    # Reshape for group-wise dequantization
    # qw: [N, K] -> [N, num_groups, group_size]
    qw_reshaped = qw.reshape(N, num_groups, group_size)

    # sw: [N, num_groups] -> [N, num_groups, 1]
    sw_reshaped = sw.unsqueeze(-1)

    # Dequantize per group and compute matmul
    w_dequant = qw_reshaped.to(torch.float32) * sw_reshaped  # [N, num_groups, group_size]
    w_dequant = w_dequant.reshape(N, K)  # [N, K]

    y = (x.to(torch.float32) @ w_dequant.T).to(x_dtype)  # [M, N]

    if bias is not None:
        y = y + bias

    return y.view(y_shape)


@torch.no_grad()
def per_row_matmul(
    x: torch.Tensor,
    qw: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
):
    x_dtype = x.dtype
    y_shape = list(x.shape)[:-1] + [
        qw.size(0),
    ]
    x = x.reshape(-1, x.size(-1))
    x, sx = per_row_quant(x, dtype)
    BLK_M, BLK_N, BLK_K = 128, 128, 128
    M = x.size(0)
    N = qw.size(0)
    K = x.size(1)
    assert K % BLK_K == 0, f"K({K}) must be multiple of BLK_K({BLK_K})"
    y = torch.empty(y_shape, dtype=x_dtype, device=x.device)
    grid = (triton.cdiv(N, 128), triton.cdiv(M, 128))
    _per_row_post_process_mm[grid](
        x,
        sx,
        qw,
        sw,
        bias,
        y,
        M,
        N,
        K,
        BLK_M=BLK_M,
        BLK_N=BLK_N,
        BLK_K=BLK_K,
        FP8=(dtype == torch.float8_e4m3fn),
    )
    return y


class QLinear(torch.nn.Module):
    """Per-row quantized Linear layer"""
    def __init__(self, dtype, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        assert dtype in [torch.float8_e4m3fn, torch.int8]
        self.dtype = dtype
        self.qw, self.sw = per_row_quant(weight, dtype)
        self.bias = bias

    @classmethod
    def from_linear(cls, other: LinearBase, dtype):
        assert other.tp_size == 1, "Not support TP."
        return cls(dtype, other.weight, other.bias)

    def forward(self, x: torch.Tensor):
        return per_row_matmul(x, self.qw, self.sw, self.bias, self.dtype)


class QTensorLinear(torch.nn.Module):
    """Per-tensor quantized Linear layer"""
    def __init__(self, dtype, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        assert dtype in [torch.float8_e4m3fn, torch.int8]
        self.dtype = dtype
        self.qw, self.sw = per_tensor_quant(weight, dtype)
        self.bias = bias

    @classmethod
    def from_linear(cls, other: LinearBase, dtype):
        assert other.tp_size == 1, "Not support TP."
        return cls(dtype, other.weight, other.bias)

    def forward(self, x: torch.Tensor):
        return per_tensor_matmul(x, self.qw, self.sw, self.bias, self.dtype)


class QGroupLinear(torch.nn.Module):
    """Per-group quantized Linear layer"""
    def __init__(self, dtype, weight: torch.Tensor, bias: torch.Tensor | None, group_size: int):
        super().__init__()
        assert dtype in [torch.float8_e4m3fn, torch.int8]
        self.dtype = dtype
        self.group_size = group_size
        self.qw, self.sw = per_group_quant(weight, group_size, dtype)
        self.bias = bias

    @classmethod
    def from_linear(cls, other: LinearBase, dtype, group_size: int):
        assert other.tp_size == 1, "Not support TP."
        return cls(dtype, other.weight, other.bias, group_size)

    def forward(self, x: torch.Tensor):
        return per_group_matmul(x, self.qw, self.sw, self.bias, self.group_size, self.dtype)


@torch.no_grad()
def _fake_per_block_quant(x: torch.Tensor, blk_m: int, blk_n: int, dtype: torch.dtype):
    assert dtype in [torch.float8_e4m3fn, torch.int8]
    original_shape = x.shape
    original_dtype = x.dtype
    x = x.reshape(-1, x.size(-1))
    m, n = x.shape
    if blk_m == -1:
        blk_m = m
    if blk_n == -1:
        blk_n = n

    assert m % blk_m == 0, f"m({m}) must be multiple of blk_m({blk_m})"
    assert n % blk_n == 0, f"n({n}) must be multiple of blk_n({blk_n})"

    x = x.reshape(m // blk_m, blk_m, n // blk_n, blk_n)
    amax = x.abs().max(dim=-1, keepdim=True).values.max(dim=1, keepdim=True).values
    amax = torch.clamp(amax, min=1e-8)
    if dtype == torch.int8:
        x = (torch.round((x.float() / amax * 127)) * (amax / 127)).to(original_dtype)
    else:
        x = (
            (x.float() / amax * 448).to(torch.float8_e4m3fn).to(torch.float)
            / 448
            * amax
        ).to(original_dtype)

    return x.reshape(original_shape)


def fake_per_tensor_quant(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _fake_per_block_quant(x, -1, -1, dtype)


def fake_per_row_quant(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _fake_per_block_quant(x, 1, -1, dtype)


def fake_per_group_quant(
    x: torch.Tensor, group_size: int, dtype: torch.dtype
) -> torch.Tensor:
    return _fake_per_block_quant(x, 1, group_size, dtype)


def apply_weight_fake_quant(model: Qwen3ForCausalLM, weight_fake_quant_fn: lambda x: x):
    for l in model.model.layers:
        l: Qwen3DecoderLayer
        l.self_attn.qkv_proj.weight.data = weight_fake_quant_fn(
            l.self_attn.qkv_proj.weight.data
        )
        l.self_attn.o_proj.weight.data = weight_fake_quant_fn(
            l.self_attn.o_proj.weight.data
        )
        l.mlp.gate_up_proj.weight.data = weight_fake_quant_fn(
            l.mlp.gate_up_proj.weight.data
        )
        l.mlp.down_proj.weight.data = weight_fake_quant_fn(l.mlp.down_proj.weight.data)


def apply_per_row_quant(model: Qwen3ForCausalLM, dtype):
    """Apply per-row quantization to model"""
    for l in model.model.layers:
        l: Qwen3DecoderLayer
        l.self_attn.qkv_proj = QLinear.from_linear(l.self_attn.qkv_proj, dtype)
        l.self_attn.o_proj = QLinear.from_linear(l.self_attn.o_proj, dtype)
        l.mlp.gate_up_proj = QLinear.from_linear(l.mlp.gate_up_proj, dtype)
        l.mlp.down_proj = QLinear.from_linear(l.mlp.down_proj, dtype)

    torch.cuda.empty_cache()


def apply_tensor_quant(model: Qwen3ForCausalLM, dtype):
    """Apply per-tensor quantization to model"""
    for l in model.model.layers:
        l: Qwen3DecoderLayer
        l.self_attn.qkv_proj = QTensorLinear.from_linear(l.self_attn.qkv_proj, dtype)
        l.self_attn.o_proj = QTensorLinear.from_linear(l.self_attn.o_proj, dtype)
        l.mlp.gate_up_proj = QTensorLinear.from_linear(l.mlp.gate_up_proj, dtype)
        l.mlp.down_proj = QTensorLinear.from_linear(l.mlp.down_proj, dtype)

    torch.cuda.empty_cache()


def apply_group_quant(model: Qwen3ForCausalLM, dtype, group_size: int):
    """Apply per-group quantization to model"""
    for l in model.model.layers:
        l: Qwen3DecoderLayer
        l.self_attn.qkv_proj = QGroupLinear.from_linear(l.self_attn.qkv_proj, dtype, group_size)
        l.self_attn.o_proj = QGroupLinear.from_linear(l.self_attn.o_proj, dtype, group_size)
        l.mlp.gate_up_proj = QGroupLinear.from_linear(l.mlp.gate_up_proj, dtype, group_size)
        l.mlp.down_proj = QGroupLinear.from_linear(l.mlp.down_proj, dtype, group_size)

    torch.cuda.empty_cache()


def apply_weight_quant(model: Qwen3ForCausalLM, quant_fn):
    """
    Apply custom quantization function to all linear layer weights.

    Args:
        model: Qwen3ForCausalLM model
        quant_fn: Function that takes a weight tensor and returns quantized tensor
                  Should have signature: (Tensor) -> Tensor
    """
    for l in model.model.layers:
        l: Qwen3DecoderLayer
        l.self_attn.qkv_proj.weight.data = quant_fn(l.self_attn.qkv_proj.weight.data)
        l.self_attn.o_proj.weight.data = quant_fn(l.self_attn.o_proj.weight.data)
        l.mlp.gate_up_proj.weight.data = quant_fn(l.mlp.gate_up_proj.weight.data)
        l.mlp.down_proj.weight.data = quant_fn(l.mlp.down_proj.weight.data)


def apply_quant_torchao(model, linear_dtype=torch.float8_e4m3fn):
    if linear_dtype == torch.int8:
        print("Applying Int8DynamicActivationInt8WeightConfig...")
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    elif linear_dtype == torch.float8_e4m3fn:
        print("Applying Float8DynamicActivationFloat8WeightConfig...")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
    else:
        raise ValueError(f"Unknown linear_dtype: {linear_dtype}")


def test_quant_mm():
    """Test per-row quantization"""
    shapes = [(1, 1536, 1536), (10, 1536, 1536), (4096, 4096, 4096)]
    dtypes = [torch.int8, torch.float8_e4m3fn]
    device = "cuda"

    print("Testing Per-Row Quantization:")
    print("-" * 70)

    for M, N, K in shapes:
        for dtype in dtypes:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            b = torch.randn(N, device=device, dtype=torch.bfloat16)
            qx, sx = per_row_quant(x, dtype)
            recon = qx.to(torch.float32) * sx[:, None]
            q_err = (x - recon).abs().mean() / x.abs().mean()

            w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
            qw, sw = per_row_quant(w, dtype)

            y = per_row_matmul(x, qw, sw, b, dtype)
            y_ref = x @ w.T + b[None, :]
            mm_err = (y - y_ref).abs().max() / y_ref.abs().max()

            dtype_name = "int8" if dtype == torch.int8 else "fp8"
            print(f"[{M}x{N}x{K}] {dtype_name}: q_err={q_err:.4f}, mm_err={mm_err:.4f}")
            assert mm_err < 0.08, f"Error too large: {mm_err}"

    print("✓ Per-row quantization tests passed!")


def test_tensor_quant_mm():
    """Test per-tensor quantization"""
    shapes = [(1, 512, 512), (10, 1024, 1024), (32, 2048, 2048)]
    dtypes = [torch.int8, torch.float8_e4m3fn]
    device = "cuda"

    print("\nTesting Per-Tensor Quantization:")
    print("-" * 70)

    for M, N, K in shapes:
        for dtype in dtypes:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            b = torch.randn(N, device=device, dtype=torch.bfloat16)

            # Test per-tensor quantization
            w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
            qw, sw = per_tensor_quant(w, dtype)

            # Test matmul
            y = per_tensor_matmul(x, qw, sw, b, dtype)
            y_ref = x @ w.T + b[None, :]
            mm_err = (y - y_ref).abs().max() / (y_ref.abs().max() + 1e-8)

            dtype_name = "int8" if dtype == torch.int8 else "fp8"
            print(f"[{M}x{N}x{K}] {dtype_name}: mm_err={mm_err:.4f}")
            assert mm_err < 0.1, f"Error too large: {mm_err}"

    print("✓ Per-tensor quantization tests passed!")


def test_group_quant_mm():
    """Test per-group quantization"""
    shapes = [(1, 512, 512), (10, 1024, 1024)]
    group_sizes = [64, 128, 256]
    dtypes = [torch.int8, torch.float8_e4m3fn]
    device = "cuda"

    print("\nTesting Per-Group Quantization:")
    print("-" * 70)

    for M, N, K in shapes:
        for group_size in group_sizes:
            if K % group_size != 0:
                continue
            for dtype in dtypes:
                x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
                b = torch.randn(N, device=device, dtype=torch.bfloat16)

                # Test per-group quantization
                w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
                qw, sw = per_group_quant(w, group_size, dtype)

                # Test matmul
                y = per_group_matmul(x, qw, sw, b, group_size, dtype)
                y_ref = x @ w.T + b[None, :]
                mm_err = (y - y_ref).abs().max() / (y_ref.abs().max() + 1e-8)

                dtype_name = "int8" if dtype == torch.int8 else "fp8"
                print(f"[{M}x{N}x{K}] {dtype_name} group={group_size}: mm_err={mm_err:.4f}")
                assert mm_err < 0.1, f"Error too large: {mm_err}"

    print("✓ Per-group quantization tests passed!")


def test_fake_quant():
    device = "cuda"
    dtypes = [torch.int8, torch.float8_e4m3fn]
    shapes = [(16, 512), (100, 1024), (4096, 4096)]

    print("Testing Fake Quant Error:")
    print("-" * 70)

    for M, N in shapes:
        x = torch.randn(M, N, device=device)

        for dtype in dtypes:
            dtype_name = "int8" if dtype == torch.int8 else "fp8"

            # Per-tensor quant
            x_pt = fake_per_tensor_quant(x, dtype)
            err_pt = (x - x_pt).abs().mean() / x.abs().mean()

            # Per-row quant
            x_pr = fake_per_row_quant(x, dtype)
            err_pr = (x - x_pr).abs().mean() / x.abs().mean()

            # Per-group quant (group_size=128)
            x_pg = fake_per_group_quant(x, 128, dtype)
            err_pg = (x - x_pg).abs().mean() / x.abs().mean()

            print(
                f"[{M:4d}x{N:4d}] {dtype_name}: "
                f"tensor={err_pt:.4f}, row={err_pr:.4f}, group={err_pg:.4f}"
            )

    print("-" * 70)
    print("✓ Fake quant test completed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Running Quantization Tests")
    print("=" * 80)

    test_quant_mm()
    test_tensor_quant_mm()
    test_group_quant_mm()
    test_fake_quant()

    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
