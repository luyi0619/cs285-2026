from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:

    N = batch.input_ids.shape[0]
    batch_device = batch.input_ids.device

    if shuffle:
        if generator is not None:
            gen_device = generator.device
            all_indices = torch.randperm(N, generator=generator, device=gen_device).to(batch_device)
        else:
            all_indices = torch.randperm(N, generator=torch.default_generator, device="cpu").to(batch_device)
    else:
        all_indices = torch.arange(N, device=batch_device)

    for i in range(0, N, minibatch_size):
        indices = all_indices[i:i+minibatch_size]
        indices_list = indices.tolist()
        task_names = [batch.task_names[idx] for idx in indices_list] if batch.task_names is not None else None
        completion_texts = [batch.completion_texts[idx] for idx in indices_list] if batch.completion_texts is not None else None

        b = RolloutBatch(
                input_ids=batch.input_ids[indices],
                attention_mask=batch.attention_mask[indices],
                completion_mask=batch.completion_mask[indices],
                old_logprobs=batch.old_logprobs[indices],
                ref_logprobs=batch.ref_logprobs[indices],
                rewards=batch.rewards[indices],
                advantages=batch.advantages[indices],
                task_names=task_names,
                completion_texts=completion_texts,
        )

        if device is not None:
            b = b.to(device)
        yield b