#!/usr/bin/env python3
"""Generic inference profiler for forward and generate workloads.

This script is intentionally adapter-driven so it can profile both:
1) regular classification / multimodal models that only expose forward()
2) HuggingFace-style generation models that expose generate()

Adapter contract
----------------
Pass --adapter module_or_file.py:function_name and return either a dict or an
object with the following attributes:

Required:
    model: torch.nn.Module
    dataloader: iterable yielding batches

Optional:
    task: "forward" or "generate" (default: "forward")
    prepare_batch(batch, device) -> inputs
        Move / convert a raw batch into model inputs. If omitted, tensors are
        recursively moved onto the selected device.
    static_inputs:
        Inputs used for static MACs/FLOPs analysis. If omitted, the script uses
        the first profiled batch and slices it down to a single sample.
    generate_kwargs: dict
        Extra kwargs forwarded to model.generate() when task == "generate".
    batch_size_fn(batch_or_inputs) -> int
        Custom batch-size inference for unusual batch structures.
    sample_splitter(inputs) -> list[single_sample_inputs]
        Custom splitter used for single-sample latency measurements.

Example:
    python profile_inference.py \
        --adapter profile_dummy_adapters.py:build_forward_adapter

    python profile_inference.py \
        --adapter profile_dummy_adapters.py:build_generate_adapter \
        --max-batches 8 \
        --single-sample-limit 12
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover - optional import for forward-only usage
    LogitsProcessor = object  # type: ignore[assignment]
    LogitsProcessorList = list  # type: ignore[assignment]


PrepareBatchFn = Callable[[Any, torch.device], Any]
BatchSizeFn = Callable[[Any], int]
SampleSplitterFn = Callable[[Any], Sequence[Any]]


@dataclass
class ProfileBundle:
    model: torch.nn.Module
    dataloader: Iterable[Any]
    task: str = "forward"
    prepare_batch: PrepareBatchFn | None = None
    static_inputs: Any | None = None
    generate_kwargs: dict[str, Any] | None = None
    batch_size_fn: BatchSizeFn | None = None
    sample_splitter: SampleSplitterFn | None = None


class LatencyProfiler(LogitsProcessor):
    """Token-level timestamp collector for model.generate()."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_time: float | None = None
        self.step_timestamps: list[float] = []

    def start(self) -> None:
        self.reset()
        self.start_time = time.perf_counter()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.step_timestamps.append(time.perf_counter())
        return scores

    def ttft_seconds(self) -> float | None:
        if self.start_time is None or not self.step_timestamps:
            return None
        return self.step_timestamps[0] - self.start_time

    def tpot_seconds(self) -> list[float]:
        if len(self.step_timestamps) < 2:
            return []
        return [
            current - previous
            for previous, current in zip(self.step_timestamps[:-1], self.step_timestamps[1:])
        ]

    def generated_tokens(self) -> int:
        return len(self.step_timestamps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inference performance")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter spec: module:function or /path/to/file.py:function",
    )
    parser.add_argument(
        "--adapter-kwargs",
        type=str,
        default="{}",
        help="JSON object passed as kwargs into the adapter function",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: auto/cpu/cuda/cuda:0",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=20,
        help="Maximum batches to profile for throughput stats",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="Warmup iterations before timing begins",
    )
    parser.add_argument(
        "--single-sample-limit",
        type=int,
        default=16,
        help="Maximum single-sample runs used for TTFT/TPOT or per-sample latency",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="",
        help="Optional path to save the profiling report as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_python_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Invalid adapter spec '{spec}'. Expected module:function")
    module_spec, object_name = spec.split(":", 1)
    if module_spec.endswith(".py") or os.path.sep in module_spec:
        module_path = Path(module_spec)
        if not module_path.is_absolute():
            module_path = Path.cwd() / module_path
        module_name = f"_profile_adapter_{module_path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Unable to import adapter file: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    current = module
    for part in object_name.split("."):
        current = getattr(current, part)
    return current


def to_namespace(bundle: Any) -> SimpleNamespace:
    if isinstance(bundle, ProfileBundle):
        return SimpleNamespace(**bundle.__dict__)
    if isinstance(bundle, Mapping):
        return SimpleNamespace(**bundle)
    return SimpleNamespace(**vars(bundle))


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def default_prepare_batch(batch: Any, device: torch.device) -> Any:
    return move_to_device(batch, device)


def infer_batch_size(value: Any) -> int | None:
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim > 0 else None
    if isinstance(value, Mapping):
        for item in value.values():
            batch_size = infer_batch_size(item)
            if batch_size is not None:
                return batch_size
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        tensor_sizes = [infer_batch_size(item) for item in value]
        for batch_size in tensor_sizes:
            if batch_size is not None:
                return batch_size
        if value and all(isinstance(item, str) for item in value):
            return len(value)
        return None
    return None


def slice_single_sample(value: Any, index: int, batch_size: int | None = None) -> Any:
    if batch_size is None:
        batch_size = infer_batch_size(value)
    if torch.is_tensor(value):
        if value.ndim == 0 or batch_size is None or value.shape[0] != batch_size:
            return value
        return value[index:index + 1]
    if isinstance(value, Mapping):
        return {
            key: slice_single_sample(item, index=index, batch_size=batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            slice_single_sample(item, index=index, batch_size=batch_size) for item in value
        )
    if isinstance(value, list):
        if batch_size is not None and len(value) == batch_size:
            return value[index:index + 1]
        return [slice_single_sample(item, index=index, batch_size=batch_size) for item in value]
    return value


def default_sample_splitter(inputs: Any) -> Sequence[Any]:
    batch_size = infer_batch_size(inputs)
    if batch_size is None:
        return [inputs]
    return [slice_single_sample(inputs, idx, batch_size=batch_size) for idx in range(batch_size)]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def call_model(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


def call_generate(model: torch.nn.Module, inputs: Any, generate_kwargs: Mapping[str, Any]) -> Any:
    if not hasattr(model, "generate"):
        raise AttributeError("Selected model does not expose generate()")
    if isinstance(inputs, Mapping):
        return model.generate(**inputs, **generate_kwargs)
    if isinstance(inputs, tuple):
        return model.generate(*inputs, **generate_kwargs)
    if isinstance(inputs, list):
        return model.generate(*inputs, **generate_kwargs)
    return model.generate(inputs, **generate_kwargs)


def summarize(values: Sequence[float], unit_scale: float = 1.0) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64) * unit_scale
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize_mean_and_p95(values: Sequence[float], unit_scale: float = 1.0) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64) * unit_scale
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
    }


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return str(value)
    if abs(value) >= 1_000_000:
        return f"{value:,.2f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def normalize_inputs_for_static_cost(inputs: Any) -> tuple[torch.nn.Module | None, tuple[Any, ...], str | None]:
    if isinstance(inputs, Mapping):
        key_order = list(inputs.keys())

        class KeywordForwardWrapper(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module, ordered_keys: list[str]) -> None:
                super().__init__()
                self.base_model = base_model
                self.ordered_keys = ordered_keys

            def forward(self, *args: Any) -> Any:
                kwargs = {key: value for key, value in zip(self.ordered_keys, args)}
                return self.base_model(**kwargs)

        values = tuple(inputs[key] for key in key_order)
        return KeywordForwardWrapper, values, ",".join(key_order)
    if isinstance(inputs, tuple):
        return None, inputs, None
    if isinstance(inputs, list):
        return None, tuple(inputs), None
    return None, (inputs,), None


def profile_static_cost(model: torch.nn.Module, inputs: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "single_sample_forward": True,
        "analyzer": None,
        "forward_macs": None,
        "forward_flops": None,
        "error": None,
    }
    wrapper_factory, positional_inputs, input_signature = normalize_inputs_for_static_cost(inputs)
    profiled_model: torch.nn.Module = model
    if wrapper_factory is not None:
        profiled_model = wrapper_factory(model, input_signature.split(","))  # type: ignore[arg-type]

    try:
        from thop import profile as thop_profile

        with torch.no_grad():
            macs, _ = thop_profile(profiled_model, inputs=positional_inputs, verbose=False)
        report["analyzer"] = "thop"
        report["forward_macs"] = float(macs)
        report["forward_flops"] = float(macs) * 2.0
        return report
    except Exception as exc:
        report["error"] = f"thop unavailable or failed: {exc}"

    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            flops = FlopCountAnalysis(profiled_model, positional_inputs).total()
        report["analyzer"] = "fvcore"
        report["forward_flops"] = float(flops)
        report["forward_macs"] = float(flops) / 2.0
        report["error"] = None
        return report
    except Exception as exc:
        report["error"] = f"{report['error']}; fvcore unavailable or failed: {exc}"
        return report


def get_parameter_stats(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(param.numel() for param in model.parameters())),
        "trainable": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }


def warmup_model(
    model: torch.nn.Module,
    bundle: SimpleNamespace,
    sample_inputs: Any,
    device: torch.device,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0:
        return
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            if bundle.task == "generate":
                call_generate(model, sample_inputs, bundle.generate_kwargs)
            else:
                call_model(model, sample_inputs)
            sync_device(device)


def profile_single_sample_latency(
    model: torch.nn.Module,
    bundle: SimpleNamespace,
    device: torch.device,
    max_samples: int,
    warmup_steps: int,
) -> dict[str, Any] | None:
    if max_samples <= 0:
        return None

    splitter = bundle.sample_splitter or default_sample_splitter
    iterator = iter(bundle.dataloader)
    sample_inputs: list[Any] = []

    while len(sample_inputs) < max_samples:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        prepared = bundle.prepare_batch(raw_batch, device)
        for single_inputs in splitter(prepared):
            sample_inputs.append(single_inputs)
            if len(sample_inputs) >= max_samples:
                break

    if not sample_inputs:
        return None

    warmup_model(model, bundle, sample_inputs[0], device=device, warmup_steps=warmup_steps)

    latencies: list[float] = []
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    token_rates: list[float] = []
    generated_tokens: list[int] = []

    model.eval()
    with torch.no_grad():
        for single_inputs in sample_inputs:
            if bundle.task == "generate":
                latency_probe = LatencyProfiler()
                processors = LogitsProcessorList()
                existing_processors = bundle.generate_kwargs.get("logits_processor")
                if existing_processors is not None:
                    processors.extend(existing_processors)
                processors.append(latency_probe)
                generate_kwargs = dict(bundle.generate_kwargs)
                generate_kwargs["logits_processor"] = processors

                sync_device(device)
                latency_probe.start()
                start_time = time.perf_counter()
                call_generate(model, single_inputs, generate_kwargs)
                sync_device(device)
                total_latency = time.perf_counter() - start_time
                latencies.append(total_latency)
                generated = latency_probe.generated_tokens()
                generated_tokens.append(generated)
                ttft = latency_probe.ttft_seconds()
                if ttft is not None:
                    ttft_values.append(ttft)
                tpot_values.extend(latency_probe.tpot_seconds())
                if generated > 0 and total_latency > 0:
                    token_rates.append(generated / total_latency)
            else:
                sync_device(device)
                start_time = time.perf_counter()
                call_model(model, single_inputs)
                sync_device(device)
                latencies.append(time.perf_counter() - start_time)

    report: dict[str, Any] = {
        "num_profiled_samples": len(latencies),
    }
    if bundle.task == "generate":
        report.update(
            {
                "ttft_ms": summarize_mean_and_p95(ttft_values, unit_scale=1000.0),
                "tpot_ms": summarize_mean_and_p95(tpot_values, unit_scale=1000.0),
                "generated_tokens_per_second": summarize_mean_and_p95(token_rates, unit_scale=1.0),
                "generate_latency_ms": summarize(latencies, unit_scale=1000.0),
                "generated_tokens": summarize(generated_tokens, unit_scale=1.0),
            }
        )
    else:
        report["forward_latency_ms"] = summarize(latencies, unit_scale=1000.0)
        if latencies:
            total_time = float(np.sum(latencies))
            report["forward_samples_per_second"] = len(latencies) / total_time if total_time > 0 else None
    return report


def profile_batch_throughput(
    model: torch.nn.Module,
    bundle: SimpleNamespace,
    device: torch.device,
    max_batches: int,
    warmup_steps: int,
) -> dict[str, Any]:
    iterator = iter(bundle.dataloader)
    operation_latencies: list[float] = []
    e2e_latencies: list[float] = []
    sample_latencies: list[float] = []
    batch_sizes: list[int] = []
    total_samples = 0

    warmed_up = False
    model.eval()
    measured_batches = 0
    with torch.no_grad():
        while measured_batches < max_batches:
            try:
                e2e_start = time.perf_counter()
                raw_batch = next(iterator)
            except StopIteration:
                break
            prepared = bundle.prepare_batch(raw_batch, device)
            batch_size = None
            if bundle.batch_size_fn is not None:
                batch_size = bundle.batch_size_fn(prepared)
            if batch_size is None:
                batch_size = infer_batch_size(prepared) or infer_batch_size(raw_batch) or 1

            if not warmed_up:
                warmup_model(model, bundle, prepared, device=device, warmup_steps=warmup_steps)
                warmed_up = True
                continue

            sync_device(device)
            op_start = time.perf_counter()
            if bundle.task == "generate":
                call_generate(model, prepared, bundle.generate_kwargs)
            else:
                call_model(model, prepared)
            sync_device(device)
            op_latency = time.perf_counter() - op_start
            e2e_latency = time.perf_counter() - e2e_start

            operation_latencies.append(op_latency)
            e2e_latencies.append(e2e_latency)
            sample_latencies.append(op_latency / max(batch_size, 1))
            batch_sizes.append(batch_size)
            total_samples += batch_size
            measured_batches += 1

    total_e2e = float(np.sum(e2e_latencies))
    throughput = total_samples / total_e2e if total_e2e > 0 else None
    operation_name = "generate" if bundle.task == "generate" else "forward"

    return {
        "num_profiled_batches": len(operation_latencies),
        "num_profiled_samples": total_samples,
        f"{operation_name}_batch_latency_ms": summarize(operation_latencies, unit_scale=1000.0),
        "e2e_batch_latency_ms": summarize(e2e_latencies, unit_scale=1000.0),
        f"{operation_name}_sample_latency_ms_mean": (
            float(np.mean(np.asarray(sample_latencies) * 1000.0)) if sample_latencies else None
        ),
        "eval_samples_per_second": throughput,
        "batch_size": summarize(batch_sizes, unit_scale=1.0),
    }


def measure_memory_peaks(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "cuda_max_memory_allocated_mb": float(allocated_mb),
        "cuda_max_memory_reserved_mb": float(reserved_mb),
    }


def print_report(report: Mapping[str, Any]) -> None:
    params = report["params"]
    compute = report["compute_cost"]
    print("=" * 80)
    print("Inference Profiling Report")
    print("=" * 80)
    print(f"Task: {report['task']}")
    print(f"Device: {report['device']}")
    print(f"Params (total): {format_number(params['total'])}")
    print(f"Params (trainable): {format_number(params['trainable'])}")
    print(f"Forward MACs: {format_number(compute['forward_macs'])}")
    print(f"Forward FLOPs: {format_number(compute['forward_flops'])}")
    if compute.get("analyzer"):
        print(f"Static analyzer: {compute['analyzer']}")
    if compute.get("error"):
        print(f"Static cost note: {compute['error']}")

    if report.get("single_sample_latency") is not None:
        single = report["single_sample_latency"]
        print("-" * 80)
        print("Single-sample latency")
        print(f"Profiled samples: {single['num_profiled_samples']}")
        if report["task"] == "generate":
            print(f"TTFT mean / p95 (ms): {single['ttft_ms']}")
            print(f"TPOT mean / p95 (ms): {single['tpot_ms']}")
            print(f"Generated tokens / sec: {single['generated_tokens_per_second']}")
            print(f"Generate latency (ms): {single['generate_latency_ms']}")
        else:
            print(f"Forward latency (ms): {single['forward_latency_ms']}")
            print(f"Forward samples / sec: {single['forward_samples_per_second']}")

    throughput = report["throughput"]
    print("-" * 80)
    print("Batch throughput")
    print(f"Profiled batches: {throughput['num_profiled_batches']}")
    print(f"Profiled samples: {throughput['num_profiled_samples']}")
    for key, value in throughput.items():
        if key in {"num_profiled_batches", "num_profiled_samples"}:
            continue
        print(f"{key}: {value}")

    if report.get("memory") is not None:
        print("-" * 80)
        print("CUDA memory peak")
        print(report["memory"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    adapter_fn = load_python_object(args.adapter)
    adapter_kwargs = json.loads(args.adapter_kwargs)
    if inspect.isclass(adapter_fn):
        bundle_raw = adapter_fn(**adapter_kwargs)
    else:
        bundle_raw = adapter_fn(**adapter_kwargs)
    bundle = to_namespace(bundle_raw)

    if not hasattr(bundle, "model") or not hasattr(bundle, "dataloader"):
        raise ValueError("Adapter must provide both 'model' and 'dataloader'")

    bundle.task = getattr(bundle, "task", "forward")
    if bundle.task not in {"forward", "generate"}:
        raise ValueError(f"Unsupported task '{bundle.task}'. Use 'forward' or 'generate'")
    bundle.prepare_batch = getattr(bundle, "prepare_batch", None) or default_prepare_batch
    bundle.generate_kwargs = dict(getattr(bundle, "generate_kwargs", None) or {})
    bundle.batch_size_fn = getattr(bundle, "batch_size_fn", None)
    bundle.sample_splitter = getattr(bundle, "sample_splitter", None)

    model = bundle.model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    params = get_parameter_stats(model)

    dataloader_iter = iter(bundle.dataloader)
    try:
        first_batch = next(dataloader_iter)
    except StopIteration as exc:
        raise RuntimeError("Adapter dataloader is empty") from exc

    first_inputs = bundle.prepare_batch(first_batch, device)
    static_inputs = bundle.static_inputs
    if static_inputs is None:
        static_inputs = slice_single_sample(first_inputs, index=0)
    else:
        static_inputs = move_to_device(static_inputs, device)

    compute_cost = profile_static_cost(model, static_inputs)
    single_sample_latency = profile_single_sample_latency(
        model=model,
        bundle=bundle,
        device=device,
        max_samples=args.single_sample_limit,
        warmup_steps=max(args.warmup_batches, 0),
    )
    throughput = profile_batch_throughput(
        model=model,
        bundle=bundle,
        device=device,
        max_batches=max(args.max_batches, 0),
        warmup_steps=max(args.warmup_batches, 0),
    )
    memory = measure_memory_peaks(device)

    report = {
        "task": bundle.task,
        "device": str(device),
        "params": params,
        "compute_cost": compute_cost,
        "single_sample_latency": single_sample_latency,
        "throughput": throughput,
        "memory": memory,
    }

    print_report(report)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
