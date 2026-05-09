"""Small adapter examples for validating profile_inference.py."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


class RandomTokenDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        input_ids = torch.randint(
            low=3,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        masked = embedded * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.proj(pooled)


@dataclass
class AdapterBundle:
    model: nn.Module
    dataloader: DataLoader
    task: str
    generate_kwargs: dict | None = None


def build_forward_adapter(
    num_samples: int = 64,
    batch_size: int = 8,
    seq_len: int = 32,
    vocab_size: int = 256,
    hidden_size: int = 64,
    num_classes: int = 6,
) -> AdapterBundle:
    dataset = RandomTokenDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = TinyClassifier(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    )
    return AdapterBundle(model=model, dataloader=dataloader, task="forward")


def build_generate_adapter(
    num_samples: int = 32,
    batch_size: int = 4,
    seq_len: int = 24,
    vocab_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_new_tokens: int = 12,
) -> AdapterBundle:
    dataset = RandomTokenDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len + max_new_tokens + 8,
        n_ctx=seq_len + max_new_tokens + 8,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": 0,
        "eos_token_id": None,
        "use_cache": True,
    }
    return AdapterBundle(
        model=model,
        dataloader=dataloader,
        task="generate",
        generate_kwargs=generate_kwargs,
    )
