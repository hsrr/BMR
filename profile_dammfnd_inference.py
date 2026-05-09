#!/usr/bin/env python3
"""Inference profiler for DAMMFND six-class classification models.

This script measures:
1. Static compute cost: params, MACs, FLOPs
2. Single-sample latency: forward latency and classification-style TTFT proxy
3. Dataset throughput: batch latency, end-to-end latency, samples/sec
4. CUDA memory peak
5. Optional six-class classification metrics on the evaluation set
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.dammfnd import DAMMFNDMODEL  # noqa: E402
from utils.custom_dataloader import bert_data as custom_data  # noqa: E402


CLASS_NAMES = [
    "real_news",
    "image_forgery",
    "entity_inconsistency",
    "event_inconsistency",
    "temporal_inconsistency",
    "invalid_visual",
]


class ProfilingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_names: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = list(input_names)
        self.eval()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        kwargs = {name: value for name, value in zip(self.input_names, inputs)}
        return extract_logits(self.model(**kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/map-vepfs/liniuniu/hesirui/datasets")
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--test-image-root", type=str, default=None)
    parser.add_argument("--bert", type=str, default="/map-vepfs/liniuniu/hesirui/bert-base-uncased")
    parser.add_argument("--clip-model", type=str, default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16")
    parser.add_argument("--use-cn-clip", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=197)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--mlp-dims", type=str, default="384")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--single-sample-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--flops-backend",
        type=str,
        default="auto",
        choices=("auto", "thop", "fvcore", "none"),
    )
    parser.add_argument("--print-classification-metrics", action="store_true", default=False)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def parse_mlp_dims(raw_dims: str) -> List[int]:
    dims = [segment.strip() for segment in raw_dims.split(",") if segment.strip()]
    return [int(dim) for dim in dims]


def discover_test_path(data_dir: Path) -> Optional[Path]:
    exact_candidates = [
        data_dir / "test.jsonl",
        data_dir / "test.json",
        data_dir / "test.ndjson",
    ]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.test_path:
        test_path = Path(args.test_path).expanduser().resolve()
    else:
        discovered_test_path = discover_test_path(data_dir)
        test_path = discovered_test_path if discovered_test_path is not None else data_dir / "test.jsonl"
    if args.test_image_root:
        test_image_root = Path(args.test_image_root).expanduser().resolve()
    elif args.image_root:
        test_image_root = Path(args.image_root).expanduser().resolve()
    else:
        test_image_root = data_dir / "AMG_MEDIA" / "test_imagesN"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    return {
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "test_path": test_path,
        "test_image_root": test_image_root,
        "output_json": output_json,
    }


def validate_paths(paths: Dict[str, Optional[Path]]) -> None:
    required = {
        "checkpoint": paths["checkpoint"],
        "test_path": paths["test_path"],
        "test_image_root": paths["test_image_root"],
    }
    missing = [f"{name}: {path}" for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def extract_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "network", "module"):
            if key in checkpoint_obj:
                return {k.replace("module.", "", 1): v for k, v in checkpoint_obj[key].items()}
        return {
            k.replace("module.", "", 1): v
            for k, v in checkpoint_obj.items()
            if torch.is_tensor(v)
        }
    return checkpoint_obj


def build_test_loader(args: argparse.Namespace, paths: Dict[str, Optional[Path]]) -> Any:
    category_dict = {name: index for index, name in enumerate(CLASS_NAMES)}
    loader_builder = custom_data(
        max_len=args.max_len,
        batch_size=args.batch_size,
        bert=args.bert,
        category_dict=category_dict,
        num_workers=args.num_workers,
        root_dir=str(paths["test_image_root"]),
        clip_model=args.clip_model,
    )
    return loader_builder.load_data(str(paths["test_path"]), shuffle=False)


def build_model(args: argparse.Namespace, device: torch.device) -> DAMMFNDMODEL:
    model = DAMMFNDMODEL(
        emb_dim=args.emb_dim,
        mlp_dims=parse_mlp_dims(args.mlp_dims),
        bert=args.bert,
        out_channels=320,
        dropout=args.dropout,
        num_classes=args.num_classes,
        use_cn_clip=args.use_cn_clip,
        clip_model=args.clip_model,
    )
    state_dict = extract_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    return {
        "mean_ms": float(statistics.mean(values)),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
    }


def current_cuda_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    return {
        "cuda_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "cuda_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }


def batch_to_device(batch: Tuple[Any, ...], device: torch.device) -> Dict[str, Any]:
    batch_data = {
        "content": batch[0].to(device, non_blocking=True),
        "content_masks": batch[1].to(device, non_blocking=True),
        "label": batch[2].to(device, non_blocking=True),
        "category": batch[3].to(device, non_blocking=True),
        "image": batch[4].to(device, non_blocking=True),
        "clip_image": batch[5].to(device, non_blocking=True),
        "clip_text": batch[6].to(device, non_blocking=True),
        "multi_category": batch[7].to(device, non_blocking=True),
    }
    if len(batch) > 8:
        batch_data["clip_attention_mask"] = batch[8].to(device, non_blocking=True)
    return batch_data


def make_forward_kwargs(batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    keys = ["content", "content_masks", "image", "clip_image", "clip_text", "multi_category"]
    kwargs = {key: batch_data[key] for key in keys}
    if "clip_attention_mask" in batch_data:
        kwargs["clip_attention_mask"] = batch_data["clip_attention_mask"]
    return kwargs


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "params_total": sum(param.numel() for param in model.parameters()),
        "params_trainable": sum(param.numel() for param in model.parameters() if param.requires_grad),
    }


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item
        raise TypeError("Model output tuple/list does not contain a tensor")
    if isinstance(model_output, dict):
        for key in ("logits", "output", "pred", "prediction"):
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]
        raise TypeError("Model output dict does not contain logits-like tensor")
    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def choose_profile_batch_size(sample_batch: Dict[str, Any], cap: int = 2) -> int:
    for value in sample_batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return max(1, min(cap, int(value.shape[0])))
    return 1


def profile_compute_cost(
    model: torch.nn.Module,
    sample_batch: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    model.eval()

    input_names = ["content", "content_masks", "image", "clip_image", "clip_text", "multi_category"]
    if "clip_attention_mask" in sample_batch:
        input_names.append("clip_attention_mask")

    bs_for_flop = choose_profile_batch_size(sample_batch, cap=2)
    sample_inputs = tuple(
        sample_batch[name][:bs_for_flop] if torch.is_tensor(sample_batch[name]) else sample_batch[name]
        for name in input_names
    )

    wrapper = ProfilingWrapper(model, input_names)
    wrapper.eval()
    report = {"backend": "none", "forward_macs": None, "forward_flops": None, "notes": []}

    with torch.inference_mode():
        if thop_profile is not None and backend in ("auto", "thop"):
            macs, _ = thop_profile(wrapper, inputs=sample_inputs, verbose=False)
            report["backend"] = "thop"
            report["forward_macs"] = float(macs) / bs_for_flop
            report["forward_flops"] = float(macs * 2.0) / bs_for_flop
            return report

        if FlopCountAnalysis is not None and backend in ("auto", "fvcore"):
            analysis = FlopCountAnalysis(wrapper, sample_inputs)
            total_ops = float(analysis.total())
            report["backend"] = "fvcore"
            report["forward_macs"] = total_ops / bs_for_flop
            report["forward_flops"] = (total_ops * 2.0) / bs_for_flop
            return report

    report["notes"].append("No FLOPs backend available. Install 'thop' or 'fvcore'.")
    return report


def benchmark_single_sample_latency(
    model: torch.nn.Module,
    single_batch: Dict[str, Any],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    model.eval()

    bs_for_latency = choose_profile_batch_size(single_batch, cap=2)
    sample_batch = {
        key: value[:bs_for_latency] if torch.is_tensor(value) else value
        for key, value in single_batch.items()
    }
    forward_kwargs = make_forward_kwargs(sample_batch)
    timings_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        for _ in range(max(warmup_steps, 0)):
            extract_logits(model(**forward_kwargs))
        sync_cuda(device)

        for _ in range(max(measure_steps, 1)):
            sync_cuda(device)
            start = time.perf_counter()
            extract_logits(model(**forward_kwargs))
            sync_cuda(device)
            timings_ms.append(((time.perf_counter() - start) * 1000.0) / bs_for_latency)

    metrics = summarize_ms(timings_ms)
    metrics["ttft_mean_ms"] = metrics["mean_ms"]
    metrics["ttft_p95_ms"] = metrics["p95_ms"]
    metrics["tpot_mean_ms"] = 0.0
    metrics["tpot_p95_ms"] = 0.0
    metrics["mean_output_seq_len"] = 1.0
    metrics["tokens_per_second"] = (
        1000.0 / metrics["mean_ms"] if metrics["mean_ms"] > 0 else float("nan")
    )
    return metrics


def compute_classification_metrics(
    labels: Sequence[int],
    logits: Sequence[Sequence[float]],
    num_classes: int,
) -> Dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(label_array, predictions)),
        "macro_precision": float(
            precision_score(label_array, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(label_array, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(label_array, predictions, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(label_array, predictions, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(label_array, predictions, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(label_array, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            label_array,
            predictions,
            labels=list(range(num_classes)),
        ).tolist(),
    }

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        label_array,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    metrics["per_class"] = {
        CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}": {
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(per_class_support[idx]),
        }
        for idx in range(num_classes)
    }

    one_hot_labels = np.eye(num_classes, dtype=np.int64)[label_array]
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(one_hot_labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None
    try:
        metrics["average_precision_macro"] = float(
            average_precision_score(one_hot_labels, probabilities, average="macro")
        )
    except ValueError:
        metrics["average_precision_macro"] = None

    return metrics


def evaluate_and_profile(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_batches: int,
    collect_metrics: bool,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    forward_batch_latency_ms: List[float] = []
    e2e_batch_latency_ms: List[float] = []
    forward_sample_latency_ms: List[float] = []
    total_samples = 0
    total_e2e_seconds = 0.0
    batches_measured = 0
    iterator = iter(dataloader)
    all_labels: List[int] = []
    all_logits: List[List[float]] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        while True:
            if max_batches and batches_measured >= max_batches:
                break
            e2e_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break

            if batch[0].shape[0] == 0:
                continue

            batch_data = batch_to_device(batch, device)
            forward_kwargs = make_forward_kwargs(batch_data)

            sync_cuda(device)
            forward_start = time.perf_counter()
            logits = extract_logits(model(**forward_kwargs))
            sync_cuda(device)
            forward_end = time.perf_counter()
            e2e_end = time.perf_counter()

            batch_size = int(batch_data["label"].shape[0])
            total_samples += batch_size
            total_e2e_seconds += max(e2e_end - e2e_start, 0.0)

            forward_ms = (forward_end - forward_start) * 1000.0
            forward_batch_latency_ms.append(forward_ms)
            e2e_batch_latency_ms.append((e2e_end - e2e_start) * 1000.0)
            forward_sample_latency_ms.append(forward_ms / max(batch_size, 1))
            batches_measured += 1

            if collect_metrics:
                all_labels.extend(batch_data["label"].detach().cpu().view(-1).tolist())
                all_logits.extend(logits.detach().cpu().tolist())

    report: Dict[str, Any] = {
        "forward_batch_latency_ms": summarize_ms(forward_batch_latency_ms),
        "e2e_batch_latency_ms": summarize_ms(e2e_batch_latency_ms),
        "forward_sample_latency_mean_ms": (
            float(statistics.mean(forward_sample_latency_ms)) if forward_sample_latency_ms else float("nan")
        ),
        "eval_samples_per_sec": (
            float(total_samples / total_e2e_seconds) if total_e2e_seconds > 0 else float("nan")
        ),
        "cuda_memory_peak": current_cuda_memory(device),
    }
    if collect_metrics and all_labels and all_logits:
        report["classification_metrics"] = compute_classification_metrics(
            labels=all_labels,
            logits=all_logits,
            num_classes=num_classes,
        )
    return report


def format_big_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value >= 1e12:
        return f"{value / 1e12:.4f} T"
    if value >= 1e9:
        return f"{value / 1e9:.4f} G"
    if value >= 1e6:
        return f"{value / 1e6:.4f} M"
    if value >= 1e3:
        return f"{value / 1e3:.4f} K"
    return f"{value:.4f}"


def safe_format(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.4f}"


def print_report(report: Dict[str, Any]) -> None:
    compute_cost = report["compute_cost"]

    print("\n[Profile] Computation cost")
    print(f"  Params (total):      {format_big_number(report['parameters']['params_total'])}")
    print(f"  Params (trainable):  {format_big_number(report['parameters']['params_trainable'])}")
    print(f"  Forward MACs:        {format_big_number(compute_cost['forward_macs'])}")
    print(f"  Forward FLOPs~:      {format_big_number(compute_cost['forward_flops'])}")
    print(f"  Backend:             {compute_cost['backend']}")
    if compute_cost["notes"]:
        print(f"  Note:                {compute_cost['notes'][0]}")

    single = report["single_sample_forward_latency_ms"]
    print("\n[Profile] Generate latency (single sample)")
    print(f"  Mean latency:        {safe_format(single['mean_ms'])} ms")
    print(f"  Std latency:         {safe_format(single['std_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(single['p50_ms'])} / {safe_format(single['p90_ms'])} / {safe_format(single['p95_ms'])} ms"
    )
    print(
        "  TTFT:                "
        f"{safe_format(single['ttft_mean_ms'])} ms (P95: {safe_format(single['ttft_p95_ms'])} ms)"
    )
    print(
        "  TPOT:                "
        f"{safe_format(single['tpot_mean_ms'])} ms (P95: {safe_format(single['tpot_p95_ms'])} ms)"
    )
    print(f"  Mean gen seq len:    {safe_format(single['mean_output_seq_len'])}")
    print(f"  Tokens / second:     {safe_format(single['tokens_per_second'])}")

    perf = report["dataset_eval"]
    print("\n[Profile] Full evaluation generate latency")
    print(f"  Mean batch latency:  {safe_format(perf['forward_batch_latency_ms']['mean_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(perf['forward_batch_latency_ms']['p50_ms'])} / "
        f"{safe_format(perf['forward_batch_latency_ms']['p90_ms'])} / "
        f"{safe_format(perf['forward_batch_latency_ms']['p95_ms'])} ms"
    )
    print(f"  Mean/sample latency: {safe_format(perf['forward_sample_latency_mean_ms'])} ms")
    print(f"  Samples / second:    {safe_format(perf['eval_samples_per_sec'])}")

    print("\n[Profile] Full evaluation end-to-end batch latency")
    print(f"  Mean batch latency:  {safe_format(perf['e2e_batch_latency_ms']['mean_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(perf['e2e_batch_latency_ms']['p50_ms'])} / "
        f"{safe_format(perf['e2e_batch_latency_ms']['p90_ms'])} / "
        f"{safe_format(perf['e2e_batch_latency_ms']['p95_ms'])} ms"
    )

    memory = perf["cuda_memory_peak"]
    print("\n[Profile] CUDA memory peak")
    print(f"  Max allocated:       {safe_format(memory['cuda_max_memory_allocated_mb'])} MB")
    print(f"  Max reserved:        {safe_format(memory['cuda_max_memory_reserved_mb'])} MB")

    if "classification_metrics" in perf:
        metrics = perf["classification_metrics"]
        print("\n[Profile] Classification metrics")
        print(f"  Accuracy:            {safe_format(metrics['accuracy'])}")
        print(f"  Balanced accuracy:   {safe_format(metrics['balanced_accuracy'])}")
        print(f"  Macro precision:     {safe_format(metrics['macro_precision'])}")
        print(f"  Macro recall:        {safe_format(metrics['macro_recall'])}")
        print(f"  Macro F1:            {safe_format(metrics['macro_f1'])}")
        print(f"  Weighted F1:         {safe_format(metrics['weighted_f1'])}")
        print(f"  ROC-AUC OVR macro:   {safe_format(metrics['roc_auc_ovr_macro'])}")
        print(f"  Avg precision macro: {safe_format(metrics['average_precision_macro'])}")


def main() -> None:
    args = parse_args()
    paths = resolve_input_paths(args)
    validate_paths(paths)
    args.checkpoint = str(paths["checkpoint"])

    device = torch.device(args.device)
    dataloader = build_test_loader(args, paths)
    first_batch_raw = next(iter(dataloader))
    first_batch = batch_to_device(first_batch_raw, device)

    model = build_model(args, device)

    report = {
        "paths": {key: str(value) if value is not None else None for key, value in paths.items()},
        "parameters": count_parameters(model),
        "compute_cost": profile_compute_cost(model, first_batch, args.flops_backend),
        "single_sample_forward_latency_ms": benchmark_single_sample_latency(
            model,
            first_batch,
            device,
            args.warmup_steps,
            args.single_sample_steps,
        ),
        "dataset_eval": evaluate_and_profile(
            model,
            dataloader,
            device,
            args.max_batches,
            args.print_classification_metrics,
            args.num_classes,
        ),
    }

    print_report(report)
    if paths["output_json"] is not None:
        output_path = paths["output_json"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
        print(f"\nSaved JSON report to {output_path}")


if __name__ == "__main__":
    main()
