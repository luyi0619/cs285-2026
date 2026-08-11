from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

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

    # Optional debug
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
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    N = batch.input_ids.shape[0]

    if shuffle:
        perm = torch.randperm(N, generator=generator)
    else:
        perm = torch.arange(N)

    for i in range(0, N, minibatch_size):
        perm_slice = perm[i:min(N,i+minibatch_size)]

        # Convert indices to a list for indexing standard Python lists
        indices = perm_slice.tolist()

        task_names = [batch.task_names[idx] for idx in indices] if batch.task_names is not None else None
        completion_texts = [batch.completion_texts[idx] for idx in indices] if batch.completion_texts is not None else None

        b = RolloutBatch(
                input_ids=batch.input_ids[perm_slice],
                attention_mask=batch.attention_mask[perm_slice],
                completion_mask=batch.completion_mask[perm_slice],
                old_logprobs=batch.old_logprobs[perm_slice],
                ref_logprobs=batch.ref_logprobs[perm_slice],
                rewards=batch.rewards[perm_slice],
                advantages=batch.advantages[perm_slice],
                task_names=task_names,
                completion_texts=completion_texts,
        )

        if device is not None:
            b = b.to(device)
        yield b