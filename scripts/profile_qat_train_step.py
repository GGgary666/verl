#!/usr/bin/env python3
"""Standalone QAT train-step micro benchmark for profiling.

This script simulates a minimal training loop with stacked QATLinear layers:
forward -> loss -> backward -> optimizer.step.

It is designed to isolate QAT overhead (especially W4A4 activation observer
all-reduce) without running full RL training.

Example (single GPU, synthetic QAT stack):
  python scripts/profile_qat_train_step.py --mode w4a4 --steps 30 --warmup 10

Example (HF causal LM + fused RMSNorm W4A4 activation path):
  python scripts/profile_qat_train_step.py --stack hf --model-path /path/to/model \\
    --mode w4a4 --hf-fuse-w4a4-rms-norm-activation --steps 30 --warmup 10

Example (Chrome trace for operator stacks, e.g. aten::copy_): add --profile-lite and either
  ``--profile-lite-export-trace`` (default path under ./qat_profile_traces/) or
  ``--profile-lite-export-trace /tmp/qat.json``.

Example (8 GPUs, distributed all-reduce path):
  torchrun --nproc_per_node=8 scripts/profile_qat_train_step.py \
    --mode w4a4 --activation-observer-sync-interval 1 --steps 30 --warmup 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import types
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.qat.core import (
    QATConfig,
    apply_qat,
    enable_batched_amax_sync,
    enable_qat_fuse,
    sync_activation_observer_amax,
)
from verl.utils.qat.fused_rms_norm_fake_quant import FusedRMSNormFakeQuant
from verl.utils.qat.linear import QATLinear, QATMode

# argparse ``const`` when ``--profile-lite-export-trace`` is passed without a path
_PROFILE_LITE_TRACE_AUTO = "__PROFILE_LITE_TRACE_AUTO__"


class QATStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        mode: QATMode,
        group_size: int,
        activation_observer: str,
        activation_observer_update_interval: int,
        activation_observer_freeze_after_steps: int,
        activation_observer_sync_interval: int,
        fake_quant_kernel_impl: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QATLinear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias=False,
                    mode=mode,
                    group_size=group_size,
                    activation_observer=activation_observer,
                    activation_observer_update_interval=activation_observer_update_interval,
                    activation_observer_freeze_after_steps=activation_observer_freeze_after_steps,
                    activation_observer_sync_interval=activation_observer_sync_interval,
                    fake_quant_kernel_impl=fake_quant_kernel_impl,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.norm(x)


class LinearStack(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.norm(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile QAT forward/backward/update micro loop.")
    parser.add_argument("--stack", choices=["qat", "linear", "hf"], default="qat")
    parser.add_argument("--mode", choices=["w4a4", "w4a16"], default="w4a4")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--activation-observer", type=str, default="static_minmax")
    parser.add_argument("--activation-observer-update-interval", type=int, default=1)
    parser.add_argument("--activation-observer-freeze-after-steps", type=int, default=-1)
    parser.add_argument("--activation-observer-sync-interval", type=int, default=1)
    parser.add_argument("--fake-quant-kernel-impl", choices=["legacy", "nvfp4", "torchao_real"], default="nvfp4")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", type=str, default="max-autotune")
    parser.add_argument("--profile-lite", action="store_true")
    parser.add_argument("--profile-lite-start-step", type=int, default=-1)
    parser.add_argument("--profile-lite-steps", type=int, default=10)
    parser.add_argument("--profile-lite-topk", type=int, default=20)
    parser.add_argument(
        "--profile-lite-export-trace",
        nargs="?",
        const=_PROFILE_LITE_TRACE_AUTO,
        default="",
        metavar="PATH",
        help=(
            "With --profile-lite: after the run, export a Chrome trace (chrome://tracing or perfetto.dev). "
            "Use bare --profile-lite-export-trace or PATH=auto for default file "
            "./qat_profile_traces/qat_lite_<stack>_<mode>_ws<WS>_r<RANK>_<timestamp>.json; "
            "or pass an explicit .json path. Useful to inspect stacks for aten::copy_, STEFP4QuantTriton, etc."
        ),
    )
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--forward-only-no-grad", action="store_true")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd", "none"],
        default="adamw",
        help=(
            "Optimizer for the micro train loop. Use 'sgd' or 'none' on memory-constrained "
            "single-GPU runs (e.g. 8B + AdamW needs ~96GB+ for fp32 Adam states alone)."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--hf-trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-enable-qat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hf-fuse-w4a4-rms-norm-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="W4A4 + HF stack only: replace decoder RMSNorms with FusedRMSNormFakeQuant (see QATConfig.fuse_w4a4_rms_norm_activation).",
    )
    parser.add_argument("--hf-use-monkey-patch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-use-tiled-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-tiled-mlp-shards", type=int, default=4)
    parser.add_argument("--hf-attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hf-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--verify-observer-cadence",
        action="store_true",
        help=(
            "Count _update_input_global_scale calls per train step and check they match "
            "activation_observer_update_interval (rank0 only). Implies W4A4 + at least one observer module."
        ),
    )
    parser.add_argument(
        "--verify-fuse-w4a4",
        action="store_true",
        help=(
            "After model init, verify fuse_w4a4_rms_norm_activation matches module structure "
            "(FusedRMSNormFakeQuant count, QATLinear skip flags; HF + W4A4 only, rank0)."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Build model, run structural checks (--verify-fuse-w4a4 / cadence setup), then exit before the train loop.",
    )
    parser.add_argument(
        "--enable-batched-amax-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "W4A4: defer per-module activation amax all_reduce to one batched sync after backward "
            "(mirrors dp_actor.update_policy)."
        ),
    )
    return parser.parse_args()


class _ObserverUpdateCounter:
    """Per-train-step tally of activation observer buffer updates."""

    def __init__(self) -> None:
        self.current_step: int = 0
        self.by_step: dict[int, int] = defaultdict(int)

    def set_step(self, step: int) -> None:
        self.current_step = step

    def record(self) -> None:
        self.by_step[self.current_step] += 1


def _install_observer_update_counters(
    modules: list[nn.Module],
    counter: _ObserverUpdateCounter,
) -> None:
    for module in modules:
        original = module._update_input_global_scale

        def _wrapped(self, x, _orig=original, _ctr=counter):
            _ctr.record()
            return _orig(x)

        module._update_input_global_scale = types.MethodType(_wrapped, module)


def _count_act_observer_modules(
    qat_modules: list[QATLinear],
    fused_norm_modules: list[FusedRMSNormFakeQuant],
) -> int:
    """Modules that may call _update_input_global_scale on a qualifying train step."""
    n = len(fused_norm_modules)
    n += sum(1 for m in qat_modules if not getattr(m, "skip_input_activation_fake_quant", False))
    return n


def _verify_observer_cadence(
    counter: _ObserverUpdateCounter,
    *,
    num_steps: int,
    update_interval: int,
    freeze_after_steps: int,
    modules_per_qualifying_step: int,
) -> list[str]:
    errors: list[str] = []
    for step in range(num_steps):
        if freeze_after_steps >= 0 and step > freeze_after_steps:
            expected = 0
        elif step % update_interval == 0:
            expected = modules_per_qualifying_step
        else:
            expected = 0
        actual = counter.by_step.get(step, 0)
        if actual != expected:
            errors.append(
                f"step={step}: expected {expected} observer updates, got {actual} "
                f"(update_interval={update_interval}, modules={modules_per_qualifying_step})"
            )
    return errors


def _print_observer_cadence_summary(
    counter: _ObserverUpdateCounter,
    *,
    num_steps: int,
    update_interval: int,
    sync_interval: int,
    modules_per_qualifying_step: int,
) -> None:
    qualifying = [s for s in range(num_steps) if s % update_interval == 0]
    total = sum(counter.by_step.values())
    print("\n=== Observer cadence verification ===")
    print(f"update_interval={update_interval}, sync_interval={sync_interval}")
    print(f"observer_modules_per_qualifying_step={modules_per_qualifying_step}")
    print(f"qualifying_train_steps (first 12): {qualifying[:12]}{'...' if len(qualifying) > 12 else ''}")
    print(f"total_observer_updates={total}")
    for step in qualifying[:8]:
        print(f"  step {step:3d}: updates={counter.by_step.get(step, 0)}")


_SKIP_ACT_FQ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_NON_SKIP_ACT_FQ_SUFFIXES = ("o_proj", "down_proj")


def _decoder_num_hidden_layers(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    n = getattr(config, "num_hidden_layers", None)
    return int(n) if n is not None else None


def _verify_fuse_w4a4_structure(
    model: nn.Module,
    qat_modules: list[QATLinear],
    fused_norm_modules: list[FusedRMSNormFakeQuant],
    *,
    expect_fuse: bool,
) -> list[str]:
    """Check fused RMSNorm + skip flags match ``fuse_w4a4_rms_norm_activation``."""
    errors: list[str] = []
    n_fused = len(fused_norm_modules)
    n_skip = sum(1 for m in qat_modules if getattr(m, "skip_input_activation_fake_quant", False))
    n_layers = _decoder_num_hidden_layers(model)

    if expect_fuse:
        if n_fused == 0:
            errors.append("expected FusedRMSNormFakeQuant modules > 0, got 0")
        if n_skip == 0:
            errors.append("expected QATLinear with skip_input_activation_fake_quant > 0, got 0")
        if n_layers is not None:
            expected_norms = 2 * n_layers
            expected_skip = 5 * n_layers
            if n_fused != expected_norms:
                errors.append(
                    f"expected fused_rms_norm_fake_quant_modules={expected_norms} "
                    f"(2 * num_hidden_layers={n_layers}), got {n_fused}"
                )
            if n_skip != expected_skip:
                errors.append(
                    f"expected qat_linears_skip_input_act={expected_skip} "
                    f"(5 * num_hidden_layers={n_layers}), got {n_skip}"
                )
        if getattr(model, "_qat_fuse_enabled", False) is not True:
            errors.append("expected model._qat_fuse_enabled=True after enable_qat_fuse()")

        for name, module in model.named_modules():
            if not isinstance(module, QATLinear):
                continue
            short = name.rsplit(".", 1)[-1]
            if short in _SKIP_ACT_FQ_SUFFIXES and not module.skip_input_activation_fake_quant:
                errors.append(f"expected {name} skip_input_activation_fake_quant=True")
            if short in _NON_SKIP_ACT_FQ_SUFFIXES and module.skip_input_activation_fake_quant:
                errors.append(f"expected {name} skip_input_activation_fake_quant=False")
    else:
        if n_fused != 0:
            errors.append(f"expected fused_rms_norm_fake_quant_modules=0, got {n_fused}")
        if n_skip != 0:
            errors.append(f"expected qat_linears_skip_input_act=0, got {n_skip}")

    return errors


def _print_fuse_w4a4_summary(
    model: nn.Module,
    qat_modules: list[QATLinear],
    fused_norm_modules: list[FusedRMSNormFakeQuant],
    *,
    expect_fuse: bool,
) -> None:
    n_skip = sum(1 for m in qat_modules if getattr(m, "skip_input_activation_fake_quant", False))
    n_layers = _decoder_num_hidden_layers(model)
    print("\n=== W4A4 fused RMSNorm verification ===")
    print(f"expect_fuse={expect_fuse}, hf_fuse_config_matches_structure={expect_fuse}")
    print(f"num_hidden_layers={n_layers}")
    print(f"fused_rms_norm_fake_quant_modules={len(fused_norm_modules)}")
    print(f"qat_linears_total={len(qat_modules)}, qat_linears_skip_input_act={n_skip}")
    print(f"model._qat_fuse_enabled={getattr(model, '_qat_fuse_enabled', False)}")
    if n_layers is not None and expect_fuse:
        print(f"expected_fused_norms={2 * n_layers}, expected_skip_linears={5 * n_layers}")


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
    return is_distributed, world_size, rank, local_rank


def to_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return mapping[dtype]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_hf_model(args: argparse.Namespace, dtype: torch.dtype, mode: QATMode) -> nn.Module:
    if not args.model_path:
        raise ValueError("--model-path is required when --stack=hf.")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.hf_trust_remote_code,
        attn_implementation=args.hf_attn_implementation,
    )

    if args.hf_use_monkey_patch:
        apply_monkey_patch(
            model=model,
            ulysses_sp_size=1,
            use_remove_padding=False,
            use_fused_kernels=False,
            fused_kernels_backend=None,
            use_prefix_grouper=False,
            use_tiled_mlp=args.hf_use_tiled_mlp,
            tiled_mlp_shards=args.hf_tiled_mlp_shards,
        )

    if args.hf_enable_qat:
        qat_cfg = QATConfig(
            enable=True,
            mode=mode.value,
            group_size=args.group_size,
            activation_observer=args.activation_observer,
            activation_observer_update_interval=args.activation_observer_update_interval,
            activation_observer_freeze_after_steps=args.activation_observer_freeze_after_steps,
            activation_observer_sync_interval=args.activation_observer_sync_interval,
            fake_quant_kernel_impl=args.fake_quant_kernel_impl,
            fuse_w4a4_rms_norm_activation=args.hf_fuse_w4a4_rms_norm_activation,
        )
        model = apply_qat(model, qat_cfg)
        enable_qat_fuse(model)
        if mode == QATMode.W4A4 and args.enable_batched_amax_sync:
            enable_batched_amax_sync(model)

    if args.hf_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model


def main() -> None:
    args = parse_args()
    if args.warmup >= args.steps:
        raise ValueError(
            f"--warmup ({args.warmup}) must be smaller than --steps ({args.steps}), "
            "otherwise no post-warmup samples are collected."
        )
    if args.forward_only_no_grad and not args.forward_only:
        raise ValueError("--forward-only-no-grad requires --forward-only.")
    if args.verify_observer_cadence and args.mode != "w4a4":
        raise ValueError("--verify-observer-cadence requires --mode w4a4.")
    if args.verify_observer_cadence and args.stack == "linear":
        raise ValueError("--verify-observer-cadence requires --stack qat or hf with QAT enabled.")
    if args.verify_observer_cadence and args.stack == "hf" and not args.hf_enable_qat:
        raise ValueError("--verify-observer-cadence requires --hf-enable-qat when --stack=hf.")
    if args.verify_fuse_w4a4 and args.mode != "w4a4":
        raise ValueError("--verify-fuse-w4a4 requires --mode w4a4.")
    if args.verify_fuse_w4a4 and args.stack != "hf":
        raise ValueError("--verify-fuse-w4a4 requires --stack hf (fused RMSNorm applies to HF decoder).")
    if args.verify_fuse_w4a4 and not args.hf_enable_qat:
        raise ValueError("--verify-fuse-w4a4 requires --hf-enable-qat.")
    if args.verify_only and not (args.verify_fuse_w4a4 or args.verify_observer_cadence):
        raise ValueError("--verify-only requires --verify-fuse-w4a4 and/or --verify-observer-cadence.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script (QAT Triton kernels are CUDA-only).")

    is_distributed, world_size, rank, local_rank = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    raw_export = (getattr(args, "profile_lite_export_trace", None) or "").strip()
    if raw_export in (_PROFILE_LITE_TRACE_AUTO, "auto", "AUTO", "default", "DEFAULT"):
        trace_dir = os.path.join(os.getcwd(), "qat_profile_traces")
        ts = time.strftime("%Y%m%d_%H%M%S")
        chrome_trace_path = os.path.join(
            trace_dir,
            f"qat_lite_{args.stack}_{args.mode}_ws{world_size}_r{rank}_{ts}.json",
        )
    else:
        chrome_trace_path = raw_export

    if chrome_trace_path and not args.profile_lite and rank == 0:
        print(
            "Warning: --profile-lite-export-trace is set but --profile-lite is false; "
            "no Chrome trace will be exported. Add --profile-lite."
        )

    mode = QATMode(args.mode)
    dtype = to_torch_dtype(args.dtype)

    qat_modules: list[QATLinear] = []
    if args.stack == "qat":
        model = QATStack(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            mode=mode,
            group_size=args.group_size,
            activation_observer=args.activation_observer,
            activation_observer_update_interval=args.activation_observer_update_interval,
            activation_observer_freeze_after_steps=args.activation_observer_freeze_after_steps,
            activation_observer_sync_interval=args.activation_observer_sync_interval,
            fake_quant_kernel_impl=args.fake_quant_kernel_impl,
        )
        qat_modules = [m for m in model.modules() if isinstance(m, QATLinear)]
        fused_norm_modules: list[FusedRMSNormFakeQuant] = []
        if mode == QATMode.W4A4 and args.enable_batched_amax_sync:
            enable_batched_amax_sync(model)
    elif args.stack == "hf":
        model = build_hf_model(args=args, dtype=dtype, mode=mode)
        qat_modules = [m for m in model.modules() if isinstance(m, QATLinear)]
        fused_norm_modules = [m for m in model.modules() if isinstance(m, FusedRMSNormFakeQuant)]
    else:
        model = LinearStack(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
        fused_norm_modules: list[FusedRMSNormFakeQuant] = []
    model = model.to(device=device, dtype=dtype)
    if args.use_torch_compile:
        model = torch.compile(
            model,
            fullgraph=False,
            dynamic=False,
            mode=args.torch_compile_mode,
        )
    model.train()

    observer_modules: list[nn.Module] = list(qat_modules) + list(fused_norm_modules)
    observer_counter: _ObserverUpdateCounter | None = None
    modules_per_qualifying_step = 0
    if args.verify_observer_cadence:
        modules_per_qualifying_step = _count_act_observer_modules(qat_modules, fused_norm_modules)
        if modules_per_qualifying_step == 0:
            raise RuntimeError(
                "--verify-observer-cadence: no activation observer modules found "
                "(enable W4A4 QAT / fused RMSNorm)."
            )
        observer_counter = _ObserverUpdateCounter()
        _install_observer_update_counters(observer_modules, observer_counter)

    if rank == 0 and args.verify_fuse_w4a4:
        expect_fuse = bool(args.hf_fuse_w4a4_rms_norm_activation)
        _print_fuse_w4a4_summary(
            model,
            qat_modules,
            fused_norm_modules,
            expect_fuse=expect_fuse,
        )
        fuse_errors = _verify_fuse_w4a4_structure(
            model,
            qat_modules,
            fused_norm_modules,
            expect_fuse=expect_fuse,
        )
        if fuse_errors:
            print("FUSE W4A4 CHECK: FAILED")
            for err in fuse_errors:
                print(f"  - {err}")
            if is_distributed:
                dist.destroy_process_group()
            raise SystemExit(1)
        print("FUSE W4A4 CHECK: PASSED")

    if args.verify_only:
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer | None = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = None

    if args.profile_lite:
        if args.profile_lite_steps <= 0:
            raise ValueError("--profile-lite-steps must be > 0 when --profile-lite is set.")
        if args.profile_lite_topk <= 0:
            raise ValueError("--profile-lite-topk must be > 0 when --profile-lite is set.")

    profile_start_step = args.profile_lite_start_step if args.profile_lite_start_step >= 0 else args.warmup
    profile_end_step = profile_start_step + args.profile_lite_steps
    enable_lite_profile = args.profile_lite and rank == 0
    lite_prof = None
    if enable_lite_profile:
        lite_prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=profile_start_step,
                warmup=0,
                active=args.profile_lite_steps,
                repeat=1,
            ),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        lite_prof.__enter__()

    if rank == 0:
        print("=== QAT Micro Benchmark Config ===")
        print(
            f"stack={args.stack}, mode={args.mode}, kernel={args.fake_quant_kernel_impl}, dtype={args.dtype}, "
            f"update_interval={args.activation_observer_update_interval}, "
            f"freeze_after_steps={args.activation_observer_freeze_after_steps}, "
            f"sync_interval={args.activation_observer_sync_interval}, "
            f"batched_amax_sync={args.enable_batched_amax_sync}, compile={args.use_torch_compile}, "
            f"optimizer={args.optimizer}, "
            f"forward_only={args.forward_only}, forward_only_no_grad={args.forward_only_no_grad}"
        )
        if args.stack == "hf":
            n_skip_act = sum(
                1 for m in qat_modules if getattr(m, "skip_input_activation_fake_quant", False)
            )
            print(
                f"model_path={args.model_path}, hf_enable_qat={args.hf_enable_qat}, "
                f"hf_fuse_w4a4_rms_norm_activation={args.hf_fuse_w4a4_rms_norm_activation}, "
                f"fused_rms_norm_fake_quant_modules={len(fused_norm_modules)}, "
                f"qat_linears_skip_input_act={n_skip_act}, "
                f"hf_use_monkey_patch={args.hf_use_monkey_patch}, "
                f"hf_use_tiled_mlp={args.hf_use_tiled_mlp}"
            )
        if args.profile_lite:
            _pte_msg = (
                f"lite_profile=True, "
                f"profile_window=[{profile_start_step}, {profile_end_step}), "
                f"topk={args.profile_lite_topk}"
            )
            if chrome_trace_path:
                _pte_msg += f", chrome_trace_export={os.path.abspath(chrome_trace_path)}"
            print(_pte_msg)
        print(
            f"layers={args.num_layers}, hidden={args.hidden_size}, "
            f"batch={args.batch_size}, seq_len={args.seq_len}"
        )
        print(f"steps={args.steps}, warmup={args.warmup}, world_size={world_size}")

    step_times_ms: list[float] = []
    fwd_times_ms: list[float] = []
    bwd_times_ms: list[float] = []
    opt_times_ms: list[float] = []

    autocast_enabled = dtype in (torch.bfloat16, torch.float16)
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        # Observer cadence is now train-step based in QATLinear. Keep it aligned
        # with loop step so interval knobs reflect "every N train steps".
        if qat_modules:
            for module in qat_modules:
                module.set_activation_observer_train_step(step)
        if fused_norm_modules:
            for module in fused_norm_modules:
                module.set_activation_observer_train_step(step)
        if observer_counter is not None:
            observer_counter.set_step(step)

        if args.stack == "hf":
            vocab_size = int(getattr(model.config, "vocab_size", 32000))
            input_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(args.batch_size, args.seq_len),
                device=device,
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
        else:
            x = torch.randn(
                args.batch_size,
                args.seq_len,
                args.hidden_size,
                device=device,
                dtype=dtype if autocast_enabled else torch.float32,
            )
            target = None if args.forward_only else torch.randn_like(x)

        if not args.forward_only:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        sync_cuda()
        t0 = time.perf_counter()

        if args.forward_only and args.forward_only_no_grad:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
                    if args.stack == "hf":
                        _ = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                    else:
                        _ = model(x)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
                if args.stack == "hf":
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    shift_logits = out.logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                else:
                    out = model(x)
                    loss = F.mse_loss(out, target)

        sync_cuda()
        t1 = time.perf_counter()

        if args.forward_only:
            t2 = t1
            t3 = t1
        else:
            loss.backward()
            if getattr(model, "_qat_batched_amax_sync", False):
                sync_activation_observer_amax(model)
            sync_cuda()
            t2 = time.perf_counter()

            if optimizer is not None:
                optimizer.step()
            sync_cuda()
            t3 = time.perf_counter()

        if step >= args.warmup:
            fwd_ms = (t1 - t0) * 1000
            bwd_ms = (t2 - t1) * 1000
            opt_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            fwd_times_ms.append(fwd_ms)
            bwd_times_ms.append(bwd_ms)
            opt_times_ms.append(opt_ms)
            step_times_ms.append(total_ms)

        if rank == 0 and step in {0, args.warmup, args.steps - 1}:
            if args.forward_only:
                print(f"step={step:03d} forward_only=1")
            else:
                print(f"step={step:03d} loss={loss.item():.6f}")
        if enable_lite_profile:
            lite_prof.step()

    if is_distributed:
        dist.barrier()
    if enable_lite_profile and lite_prof is not None:
        lite_prof.__exit__(None, None, None)

    if rank == 0:
        if not step_times_ms:
            print(
                "No post-warmup timing samples were collected. "
                "Please ensure warmup < steps."
            )
            if is_distributed:
                dist.destroy_process_group()
            return
        tokens_per_step = args.batch_size * args.seq_len
        mean_total = statistics.mean(step_times_ms)
        mean_fwd = statistics.mean(fwd_times_ms)
        mean_bwd = statistics.mean(bwd_times_ms)
        mean_opt = statistics.mean(opt_times_ms)
        print("\n=== Timing (post-warmup) ===")
        print(f"mean_step_ms: {mean_total:.3f}")
        print(f"mean_fwd_ms:  {mean_fwd:.3f}")
        print(f"mean_bwd_ms:  {mean_bwd:.3f}")
        print(f"mean_opt_ms:  {mean_opt:.3f}")
        print(f"p50_step_ms:  {statistics.median(step_times_ms):.3f}")
        print(f"max_step_ms:  {max(step_times_ms):.3f}")
        print(f"tokens_per_s: {tokens_per_step / (mean_total / 1000):.2f}")
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
        print(f"peak_mem_alloc_mb: {peak_alloc_mb:.2f}")
        print(f"peak_mem_reserved_mb: {peak_reserved_mb:.2f}")

        if observer_counter is not None:
            _print_observer_cadence_summary(
                observer_counter,
                num_steps=args.steps,
                update_interval=args.activation_observer_update_interval,
                sync_interval=args.activation_observer_sync_interval,
                modules_per_qualifying_step=modules_per_qualifying_step,
            )
            cadence_errors = _verify_observer_cadence(
                observer_counter,
                num_steps=args.steps,
                update_interval=args.activation_observer_update_interval,
                freeze_after_steps=args.activation_observer_freeze_after_steps,
                modules_per_qualifying_step=modules_per_qualifying_step,
            )
            if cadence_errors:
                print("OBSERVER CADENCE CHECK: FAILED")
                for err in cadence_errors[:20]:
                    print(f"  - {err}")
                if len(cadence_errors) > 20:
                    print(f"  ... and {len(cadence_errors) - 20} more")
                if is_distributed:
                    dist.destroy_process_group()
                raise SystemExit(1)
            print("OBSERVER CADENCE CHECK: PASSED")

        if enable_lite_profile and lite_prof is not None:
            print("\n=== Lite Profile (rank0) ===")
            print(
                lite_prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=args.profile_lite_topk,
                )
            )
            if chrome_trace_path:
                trace_parent = os.path.dirname(os.path.abspath(chrome_trace_path))
                if trace_parent:
                    os.makedirs(trace_parent, exist_ok=True)
                lite_prof.export_chrome_trace(chrome_trace_path)
                print(f"lite_profile_trace: {os.path.abspath(chrome_trace_path)}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()



"""
# Example commands below use placeholder model paths. Replace with your local checkpoint
# (or export MODEL_PATH) before running — do not commit cluster-specific directories.

python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --mode w4a4 \
  --hf-enable-qat \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer static_minmax \
  --activation-observer-update-interval 8 \
  --activation-observer-sync-interval 8 \
  --activation-observer-freeze-after-steps -1 \
  --batch-size 2 --seq-len 1024 \
  --steps 60 --warmup 20 \
  --forward-only --forward-only-no-grad \
  --profile-lite --profile-lite-steps 20

# HF + lite profile + Chrome trace（默认写入 ./qat_profile_traces/qat_lite_*.json，便于查 aten::copy_ 等栈）
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --mode w4a4 \
  --hf-enable-qat \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer static_minmax \
  --activation-observer-update-interval 8 \
  --activation-observer-sync-interval 8 \
  --activation-observer-freeze-after-steps -1 \
  --batch-size 2 --seq-len 1024 \
  --steps 60 --warmup 20 \
  --forward-only --forward-only-no-grad \
  --profile-lite --profile-lite-steps 20 \
  --profile-lite-export-trace

# 同上，显式指定输出文件
#   ... --profile-lite --profile-lite-steps 20 --profile-lite-export-trace /tmp/qat_lite.json

# HF + fused RMSNorm：反向传播烟测（无 forward-only、无 profiler）
# Qwen3-8B-Base-NVFP4-CALIB-ONLY
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-8B-Base-NVFP4-CALIB-ONLY \
  --mode w4a4 \
  --hf-enable-qat \
  --hf-fuse-w4a4-rms-norm-activation \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer static_minmax \
  --activation-observer-update-interval 8 \
  --activation-observer-sync-interval 8 \
  --activation-observer-freeze-after-steps -1 \
  --batch-size 2 --seq-len 1024 \
  --steps 30 --warmup 5

# HF + W4A16：反向烟测（无 fused norm；fuse 仅对 w4a4 生效，此处显式关闭）
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --mode w4a16 \
  --hf-enable-qat \
  --no-hf-fuse-w4a4-rms-norm-activation \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer static_minmax \
  --activation-observer-update-interval 8 \
  --activation-observer-sync-interval 8 \
  --activation-observer-freeze-after-steps -1 \
  --batch-size 2 --seq-len 1024 \
  --steps 30 --warmup 5

# HF + W4A16：forward-only + lite profile（与 w4a4 对比吞吐时可开）
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --mode w4a16 \
  --hf-enable-qat \
  --no-hf-fuse-w4a4-rms-norm-activation \
  --fake-quant-kernel-impl nvfp4 \
  --activation-observer static_minmax \
  --activation-observer-update-interval 8 \
  --activation-observer-sync-interval 8 \
  --activation-observer-freeze-after-steps -1 \
  --batch-size 2 --seq-len 1024 \
  --steps 60 --warmup 20 \
  --forward-only --forward-only-no-grad \
  --profile-lite --profile-lite-steps 20

# HF BF16 基线（关闭 QAT；需 BF16/常规权重 checkpoint，勿与仅 FP4 calib 混作精度基线）
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --dtype bf16 \
  --no-hf-enable-qat \
  --batch-size 2 --seq-len 1024 \
  --steps 30 --warmup 5

# HF BF16 基线：forward-only + lite profile
python scripts/profile_qat_train_step.py \
  --stack hf \
  --model-path /path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
  --dtype bf16 \
  --no-hf-enable-qat \
  --batch-size 2 --seq-len 1024 \
  --steps 60 --warmup 20 \
  --forward-only --forward-only-no-grad \
  --profile-lite --profile-lite-steps 20

"""