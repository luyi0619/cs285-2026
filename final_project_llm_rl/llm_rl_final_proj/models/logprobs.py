from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.load import PolicyModel


def compute_per_token_logprobs(
    model: PolicyModel,
    input_ids: torch.Tensor, # [B, L]
    attention_mask: torch.Tensor, # [B, L]
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # TODO(student): run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits # [B, L, V]
        targets = input_ids[:, 1:] # [B, L-1]

        flat_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)) # [B*(L-1), V]
        flat_targets = targets.reshape(-1) # [B*(L-1)]

        nll = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).reshape(targets.shape) # [B, L -1]

        return -nll # [B, L-1]

def build_completion_mask(
    input_ids: torch.Tensor, # [B, L]
    attention_mask: torch.Tensor, # [B, L]
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id
    # TODO(student): build a float mask of shape [B, L-1] that selects only completion tokens.
    # Be careful about the one-token shift between logits[:, :-1] and input_ids[:, 1:].
    B, L = input_ids.shape
    positions = torch.arrange(L, device=input_ids.device).unsqueeze(0).expand(B, L) # [B, L]
    mask = (positions >= prompt_input_len) & attention_mask
    return mask[:, 1:].float() # [B, L-1]


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor, # [B, L-1]
    ref_logprobs: torch.Tensor, # [B, L-1]
    mask: torch.Tensor, # [B, L-1]
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    # TODO(student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    delta = torch.clamp(ref_logprobs - new_logprobs , min=-log_ratio_clip, max=log_ratio_clip) # [B, L-1]
    estimator = torch.exp(delta) - delta - 1 # [B, L-1]
    return masked_mean(estimator, mask, eps)
