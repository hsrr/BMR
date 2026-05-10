#!/usr/bin/env python3
"""
Inference efficiency profiling for UAMFD / UAMFDv2.

This script is designed for performance benchmarking rather than accuracy
evaluation. It can run:

1. On a real manifest file (JSON/JSONL/CSV/TSV) containing text-image samples.
2. On synthetic samples when you want to benchmark the model before training
   or before finalizing a new dataset pipeline.

For multi-class tasks (e.g. 6-way classification), the script attaches a small
linear classification head on top of the 64-d fused feature returned by UAMFD.
That makes it possible to benchmark end-to-end inference cost without retraining
the original binary head.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer


ENGLISH_DATASETS = {"gossip", "twitter", "politi"}


def parse_args():
    parser = argparse.ArgumentParser(description="Profile UAMFD inference efficiency.")
    parser.add_argument("--network-arch", choices=["UAMFD", "UAMFDv2"], default="UAMFDv2")
    parser.add_argument("--dataset-name", default="weibo", help="Used to select the backbone text encoder.")
    parser.add_argument("--manifest", default=None, help="JSON/JSONL/CSV/TSV manifest with text and image fields.")
    parser.add_argument("--text-column", default="text", help="Manifest field containing text.")
    parser.add_argument("--image-column", default="image", help="Manifest field containing image path.")
    parser.add_argument("--label-column", default="label", help="Manifest field containing label (optional).")
    parser.add_argument("--root-dir", default=".", help="Base directory for relative image paths.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum number of samples to profile.")
    parser.add_argument("--synthetic-samples", type=int, default=256, help="Synthetic sample count when no manifest is given.")
    parser.add_argument("--synthetic-text", default=None, help="Synthetic text used when no manifest is given.")
    parser.add_argument("--num-classes", type=int, default=6, help="Output class count for the profiling head.")
    parser.add_argument("--use-native-binary-head", action="store_true",
                        help="Use the original binary output instead of an attached multi-class head. Only meaningful for 1/2 classes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--single-batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--text-token-length", type=int, default=197)
    parser.add_argument("--image-token-length", type=int, default=197)
    parser.add_argument("--tokenizer-name", default=None, help="Override tokenizer name. Defaults from dataset name.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Use 0 for strict end-to-end timing without dataloader overlap.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--single-iters", type=int, default=50,
                        help="Number of batch=1 timed samples to use for single-sample latency.")
    parser.add_argument("--forward-batches", type=int, default=30,
                        help="Number of pure forward-only batches to use.")
    parser.add_argument("--end2end-batches", type=int, default=30,
                        help="Number of end-to-end batches to use.")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Benchmark device. CUDA is recommended for this repo.")
    parser.add_argument("--mae-checkpoint", default=None, help="Optional MAE checkpoint path.")
    parser.add_argument("--model-checkpoint", default=None, help="Optional full UAMFD state dict to load.")
    parser.add_argument("--strict-load", action="store_true", help="Use strict state-dict loading for the model checkpoint.")
    parser.add_argument("--report-json", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def mean_or_none(values):
    return statistics.mean(values) if values else None


def stdev_or_none(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def human_count(value):
    if value is None:
        return None
    if value >= 1e9:
        return f"{value / 1e9:.3f}B"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return str(value)


def human_bytes(value):
    if value is None:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def infer_tokenizer_name(dataset_name):
    if dataset_name.lower() in ENGLISH_DATASETS:
        return "bert-base-uncased"
    return "bert-base-chinese"


def infer_synthetic_text(tokenizer_name):
    if "uncased" in tokenizer_name.lower() or "english" in tokenizer_name.lower():
        return "Synthetic multimodal news sample used for inference benchmarking."
    return "这是一条用于推理性能测试的合成多模态新闻样本。"


def load_manifest_rows(manifest_path, max_samples, text_key, image_key, label_key):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    suffix = path.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("samples", [])
        rows = list(data[:max_samples] if max_samples else data)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                rows.append(row)
                if max_samples and len(rows) >= max_samples:
                    break
    else:
        raise ValueError("Unsupported manifest format. Use JSON/JSONL/CSV/TSV.")

    normalized = []
    for row in rows:
        normalized.append(
            {
                "text": row.get(text_key, ""),
                "image": row.get(image_key),
                "label": row.get(label_key, -1),
            }
        )
    return normalized


class ProfileDataset(Dataset):
    def __init__(self, samples, synthetic=False, synthetic_text=None):
        self.samples = samples
        self.synthetic = synthetic
        self.synthetic_text = synthetic_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        if self.synthetic:
            sample["text"] = self.synthetic_text
            sample["image"] = None
        return sample


class BatchCollator:
    def __init__(self, tokenizer, image_size, text_token_length, root_dir):
        self.tokenizer = tokenizer
        self.root_dir = Path(root_dir)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.text_token_length = text_token_length

    def _load_image(self, image_path):
        if image_path is None:
            return torch.rand(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.image_transform(image)

    def __call__(self, examples):
        texts = [str(example.get("text", "")) for example in examples]
        images = torch.stack([self._load_image(example.get("image")) for example in examples], dim=0)
        labels = []
        for example in examples:
            value = example.get("label", -1)
            try:
                labels.append(int(value))
            except (TypeError, ValueError):
                labels.append(-1)

        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_token_length,
            return_tensors="pt",
        )
        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(tokenized["input_ids"])

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": token_type_ids,
            "image": images,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def load_model_state(module, checkpoint_path, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    clean_state = OrderedDict()
    for key, value in checkpoint.items():
        clean_state[key[7:] if key.startswith("module.") else key] = value
    module.load_state_dict(clean_state, strict=strict)


class UAMFDProfileWrapper(nn.Module):
    def __init__(self, backbone, num_classes=6, use_native_binary_head=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_native_binary_head = use_native_binary_head and num_classes <= 2
        self.classifier = None if self.use_native_binary_head else nn.Linear(64, num_classes)

    def forward(self, batch):
        if self.use_native_binary_head:
            outputs = self.backbone(
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["image"],
                no_ambiguity=True,
            )
            return outputs[0]

        outputs = self.backbone(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["image"],
            no_ambiguity=True,
            return_features=True,
        )
        final_feature_main_task = outputs[-1][0]
        return self.classifier(final_feature_main_task)


def instantiate_model(args, device):
    if args.network_arch == "UAMFDv2":
        from models.UAMFDv2_Net import UAMFD_Net as Backbone
    else:
        from models.UAMFD_Net import UAMFD_Net as Backbone

    backbone = Backbone(
        dataset=args.dataset_name,
        text_token_len=args.text_token_length,
        image_token_len=args.image_token_length,
        is_use_bce=True,
        batch_size=args.batch_size,
        thresh=args.thresh,
        mae_checkpoint_path=args.mae_checkpoint,
    )

    if args.model_checkpoint:
        load_model_state(backbone, args.model_checkpoint, strict=args.strict_load)

    model = UAMFDProfileWrapper(
        backbone=backbone,
        num_classes=args.num_classes,
        use_native_binary_head=args.use_native_binary_head,
    )
    model = model.to(device)
    model.eval()
    return model


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def run_forward(model, batch, device, precision):
    with torch.inference_mode():
        with autocast_context(device, precision):
            logits = model(batch)
    return logits


def cycle_collect_gpu_batches(loader, device, limit_batches):
    batches = []
    if limit_batches <= 0:
        return batches

    while len(batches) < limit_batches:
        made_progress = False
        for batch in loader:
            batches.append(move_batch_to_device(batch, device))
            made_progress = True
            if len(batches) >= limit_batches:
                break
        if not made_progress:
            break
    return batches


def measure_forward_latencies(model, gpu_batches, device, precision, warmup_iters):
    if not gpu_batches:
        return {
            "latencies_ms": [],
            "mean_batch_latency_ms": None,
            "mean_sample_latency_ms": None,
            "samples_per_second": None,
            "total_samples": 0,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    for warmup_idx in range(warmup_iters):
        batch = gpu_batches[warmup_idx % len(gpu_batches)]
        run_forward(model, batch, device, precision)
    sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    total_samples = 0
    total_time_s = 0.0
    for batch in gpu_batches:
        sync_device(device)
        start_time = time.perf_counter()
        _ = run_forward(model, batch, device, precision)
        sync_device(device)
        elapsed_s = time.perf_counter() - start_time
        latencies_ms.append(elapsed_s * 1000.0)
        total_time_s += elapsed_s
        total_samples += int(batch["input_ids"].shape[0])

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    mean_batch_latency = mean_or_none(latencies_ms)
    mean_sample_latency = (total_time_s * 1000.0 / total_samples) if total_samples else None
    samples_per_second = (total_samples / total_time_s) if total_time_s > 0 else None
    return {
        "latencies_ms": latencies_ms,
        "mean_batch_latency_ms": mean_batch_latency,
        "mean_sample_latency_ms": mean_sample_latency,
        "samples_per_second": samples_per_second,
        "total_samples": total_samples,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def measure_end_to_end_latencies(model, loader, device, precision, warmup_iters, limit_batches):
    if limit_batches <= 0:
        return {
            "latencies_ms": [],
            "mean_batch_latency_ms": None,
            "mean_sample_latency_ms": None,
            "samples_per_second": None,
            "total_samples": 0,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    loader_iter = iter(loader)
    for _ in range(warmup_iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = move_batch_to_device(batch, device)
        run_forward(model, batch, device, precision)
    sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    total_samples = 0
    total_time_s = 0.0
    loader_iter = iter(loader)
    for _ in range(limit_batches):
        try:
            sync_device(device)
            start_time = time.perf_counter()
            batch = next(loader_iter)
        except StopIteration:
            break
        batch = move_batch_to_device(batch, device)
        _ = run_forward(model, batch, device, precision)
        sync_device(device)
        elapsed_s = time.perf_counter() - start_time
        latencies_ms.append(elapsed_s * 1000.0)
        total_time_s += elapsed_s
        total_samples += int(batch["input_ids"].shape[0])

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    mean_batch_latency = mean_or_none(latencies_ms)
    mean_sample_latency = (total_time_s * 1000.0 / total_samples) if total_samples else None
    samples_per_second = (total_samples / total_time_s) if total_time_s > 0 else None
    return {
        "latencies_ms": latencies_ms,
        "mean_batch_latency_ms": mean_batch_latency,
        "mean_sample_latency_ms": mean_sample_latency,
        "samples_per_second": samples_per_second,
        "total_samples": total_samples,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def summarize_single_sample_latency(latencies_ms):
    mean_latency = mean_or_none(latencies_ms)
    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": stdev_or_none(latencies_ms),
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p90_latency_ms": percentile(latencies_ms, 90),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "ttft_ms": mean_latency,
        "tpot_ms": None,
        "mean_gen_seq_len": 1.0,
        "samples_measured": len(latencies_ms),
    }


def estimate_flops(model, sample_batch, device, precision):
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as profiler:
                with autocast_context(device, precision):
                    _ = model(sample_batch)
                sync_device(device)

        flops = 0
        for event in profiler.key_averages():
            if getattr(event, "flops", 0):
                flops += event.flops
        if flops == 0:
            return None, None, "PyTorch profiler did not return FLOPs for this model/runtime."
        return int(flops / 2), int(flops), None
    except Exception as exc:
        return None, None, f"{exc.__class__.__name__}: {exc}"


def build_dataloader(dataset, collator, batch_size, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False,
    )


def merge_peak_memory(*results):
    allocated = [item.get("peak_allocated_bytes") for item in results if item.get("peak_allocated_bytes") is not None]
    reserved = [item.get("peak_reserved_bytes") for item in results if item.get("peak_reserved_bytes") is not None]
    return (max(allocated) if allocated else None, max(reserved) if reserved else None)


def print_report(report):
    print("\n[Profile] Computation cost (static)")
    print(f"  Params (total):      {report['computation_cost']['params_total']} ({report['computation_cost']['params_total_human']})")
    print(f"  Params (trainable):  {report['computation_cost']['params_trainable']} ({report['computation_cost']['params_trainable_human']})")
    print(f"  Forward MACs:        {report['computation_cost']['forward_macs']} ({report['computation_cost']['forward_macs_human']})")
    print(f"  Forward FLOPs:       {report['computation_cost']['forward_flops']} ({report['computation_cost']['forward_flops_human']})")
    if report["computation_cost"]["flops_note"]:
        print(f"  Note:                {report['computation_cost']['flops_note']}")

    print("\n[Profile] Generate latency (single sample)")
    single = report["single_sample_latency"]
    print(f"  Mean latency:        {single['mean_latency_ms']:.3f} ms" if single["mean_latency_ms"] is not None else "  Mean latency:        N/A")
    print(f"  P50 / P90 / P95:     {single['p50_latency_ms']:.3f} / {single['p90_latency_ms']:.3f} / {single['p95_latency_ms']:.3f} ms"
          if single["p50_latency_ms"] is not None else "  P50 / P90 / P95:     N/A")
    print(f"  TTFT:                {single['ttft_ms']:.3f} ms (classifier proxy)" if single["ttft_ms"] is not None else "  TTFT:                N/A")
    print(f"  TPOT:                {single['tpot_ms']}")
    print(f"  Mean gen seq len:    {single['mean_gen_seq_len']}")

    print("\n[Profile] Full evaluation generate latency (forward-only)")
    full = report["full_eval_generate_latency"]
    print(f"  Mean batch latency:  {full['mean_batch_latency_ms']:.3f} ms" if full["mean_batch_latency_ms"] is not None else "  Mean batch latency:  N/A")
    print(f"  Mean/sample latency: {full['mean_sample_latency_ms']:.3f} ms" if full["mean_sample_latency_ms"] is not None else "  Mean/sample latency: N/A")
    print(f"  Samples / second:    {full['samples_per_second']:.3f}" if full["samples_per_second"] is not None else "  Samples / second:    N/A")

    print("\n[Profile] Full evaluation end-to-end batch latency")
    end2end = report["full_eval_end_to_end_batch_latency"]
    print(f"  Mean batch latency:  {end2end['mean_batch_latency_ms']:.3f} ms" if end2end["mean_batch_latency_ms"] is not None else "  Mean batch latency:  N/A")
    print(f"  Mean/sample latency: {end2end['mean_sample_latency_ms']:.3f} ms" if end2end["mean_sample_latency_ms"] is not None else "  Mean/sample latency: N/A")
    print(f"  Samples / second:    {end2end['samples_per_second']:.3f}" if end2end["samples_per_second"] is not None else "  Samples / second:    N/A")
    print(f"  Overhead vs forward: {end2end['overhead_vs_forward_ms']:.3f} ms / batch"
          if end2end["overhead_vs_forward_ms"] is not None else "  Overhead vs forward: N/A")

    print("\n[Profile] CUDA memory peak")
    memory = report["cuda_memory_peak"]
    print(f"  Max allocated:       {memory['max_allocated_bytes']} ({memory['max_allocated_human']})")
    print(f"  Max reserved:        {memory['max_reserved_bytes']} ({memory['max_reserved_human']})")

    print("\n[Notes]")
    for note in report["notes"]:
        print(f"  - {note}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark command, but no CUDA device is available.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This repository's UAMFD models use CUDA-only code paths during inference. Please benchmark on CUDA.")
    tokenizer_name = args.tokenizer_name or infer_tokenizer_name(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if args.manifest:
        samples = load_manifest_rows(
            manifest_path=args.manifest,
            max_samples=args.max_samples,
            text_key=args.text_column,
            image_key=args.image_column,
            label_key=args.label_column,
        )
        synthetic = False
        synthetic_text = None
    else:
        synthetic_text = args.synthetic_text or infer_synthetic_text(tokenizer_name)
        sample_count = args.synthetic_samples if args.max_samples is None else min(args.synthetic_samples, args.max_samples)
        samples = [{"text": synthetic_text, "image": None, "label": -1} for _ in range(sample_count)]
        synthetic = True

    if not samples:
        raise RuntimeError("No samples available for profiling.")

    dataset = ProfileDataset(samples=samples, synthetic=synthetic, synthetic_text=synthetic_text)
    collator = BatchCollator(
        tokenizer=tokenizer,
        image_size=args.image_size,
        text_token_length=args.text_token_length,
        root_dir=args.root_dir,
    )

    single_loader = build_dataloader(
        dataset=dataset,
        collator=collator,
        batch_size=args.single_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    batch_loader = build_dataloader(
        dataset=dataset,
        collator=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    torch.backends.cudnn.benchmark = True
    model = instantiate_model(args, device)

    params_total, params_trainable = count_parameters(model)

    single_batches = cycle_collect_gpu_batches(single_loader, device, args.single_iters)
    if not single_batches:
        raise RuntimeError("Failed to build single-sample batches for profiling.")
    forward_batches = cycle_collect_gpu_batches(batch_loader, device, args.forward_batches)
    if not forward_batches:
        raise RuntimeError("Failed to build batched samples for profiling.")

    forward_macs, forward_flops, flops_note = estimate_flops(
        model=model,
        sample_batch=single_batches[0],
        device=device,
        precision=args.precision,
    )

    single_result = measure_forward_latencies(
        model=model,
        gpu_batches=single_batches,
        device=device,
        precision=args.precision,
        warmup_iters=args.warmup_iters,
    )
    batch_forward_result = measure_forward_latencies(
        model=model,
        gpu_batches=forward_batches,
        device=device,
        precision=args.precision,
        warmup_iters=args.warmup_iters,
    )
    end2end_result = measure_end_to_end_latencies(
        model=model,
        loader=batch_loader,
        device=device,
        precision=args.precision,
        warmup_iters=args.warmup_iters,
        limit_batches=args.end2end_batches,
    )

    peak_allocated, peak_reserved = merge_peak_memory(single_result, batch_forward_result, end2end_result)

    report = {
        "config": {
            "network_arch": args.network_arch,
            "dataset_name": args.dataset_name,
            "tokenizer_name": tokenizer_name,
            "num_classes": args.num_classes,
            "head_mode": "native_binary_head" if model.use_native_binary_head else "attached_linear_probe",
            "device": str(device),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "single_batch_size": args.single_batch_size,
            "sample_count": len(dataset),
            "synthetic_data": synthetic,
            "manifest": args.manifest,
        },
        "computation_cost": {
            "params_total": params_total,
            "params_total_human": human_count(params_total),
            "params_trainable": params_trainable,
            "params_trainable_human": human_count(params_trainable),
            "forward_macs": forward_macs,
            "forward_macs_human": human_count(forward_macs),
            "forward_flops": forward_flops,
            "forward_flops_human": human_count(forward_flops),
            "flops_note": flops_note,
        },
        "single_sample_latency": summarize_single_sample_latency(single_result["latencies_ms"]),
        "full_eval_generate_latency": {
            "mean_batch_latency_ms": batch_forward_result["mean_batch_latency_ms"],
            "mean_sample_latency_ms": batch_forward_result["mean_sample_latency_ms"],
            "samples_per_second": batch_forward_result["samples_per_second"],
            "batches_measured": len(batch_forward_result["latencies_ms"]),
            "samples_measured": batch_forward_result["total_samples"],
        },
        "full_eval_end_to_end_batch_latency": {
            "mean_batch_latency_ms": end2end_result["mean_batch_latency_ms"],
            "mean_sample_latency_ms": end2end_result["mean_sample_latency_ms"],
            "samples_per_second": end2end_result["samples_per_second"],
            "batches_measured": len(end2end_result["latencies_ms"]),
            "samples_measured": end2end_result["total_samples"],
            "overhead_vs_forward_ms": (
                end2end_result["mean_batch_latency_ms"] - batch_forward_result["mean_batch_latency_ms"]
                if end2end_result["mean_batch_latency_ms"] is not None and batch_forward_result["mean_batch_latency_ms"] is not None
                else None
            ),
            "overhead_ratio_vs_forward": (
                end2end_result["mean_batch_latency_ms"] / batch_forward_result["mean_batch_latency_ms"]
                if end2end_result["mean_batch_latency_ms"] is not None
                and batch_forward_result["mean_batch_latency_ms"] not in (None, 0)
                else None
            ),
        },
        "cuda_memory_peak": {
            "max_allocated_bytes": peak_allocated,
            "max_allocated_human": human_bytes(peak_allocated),
            "max_reserved_bytes": peak_reserved,
            "max_reserved_human": human_bytes(peak_reserved),
        },
        "notes": [
            "This script is intended for efficiency benchmarking, not accuracy evaluation.",
            "When num_classes > 2, the script attaches a random linear classification head on top of the fused 64-d feature.",
            "Without a trained checkpoint, the output logits are not semantically meaningful, but the latency/memory metrics are still useful.",
            "TTFT is reported as a classifier proxy: time-to-first-logits. TPOT is not applicable to this non-autoregressive model and is reported as null.",
            "Set --num-workers 0 if you want strict end-to-end latency that includes CPU-side loading/tokenization without overlap.",
        ],
    }

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)


if __name__ == "__main__":
    main()
