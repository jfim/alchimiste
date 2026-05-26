"""Pluggable classification heads on top of an HF encoder's hidden states.

The encoder modules produce `(B, T, hidden_size)` token reps; a head
turns those into per-token logits (and, in the CRF case, also owns the
loss + decode). Splitting the head from the encoder lets us compare
linear vs CRF vs Conv1D heads with a single config switch — see the
`models/encoder_hf.py` Tagger for the integration point.

Adding a new head:
  1. Subclass `Head`, implement `forward(hidden, mask, labels, loss_fn)`.
  2. Register it in `build_head`.
  3. Anything head-specific that needs persisting (e.g. CRF transition
     matrix) is just part of the head's `state_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from alchimiste.cleaner.data.align import LABEL_IGNORE

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class HeadOutput:
    """What a head returns. `loss` is populated only when `labels` and a
    `loss_fn` are passed; otherwise the caller is just doing inference."""

    logits: torch.Tensor  # (B, T, num_labels)
    loss: torch.Tensor | None = None


class Head(nn.Module):
    """Base class — subclasses own forward + (optionally) loss/decode.

    The unit of pluggability is the *head as a whole*, not a stack of
    layers: a CRF replaces both the per-token loss and the decode step,
    so a "layer stack" abstraction would leak. Each Head subclass is
    self-contained.
    """

    num_labels: int

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        raise NotImplementedError


class LinearHead(Head):
    """Dropout + linear projection — the default. Mirrors the head HF's
    `XxxForTokenClassification` puts on its encoders, so swapping
    backbones (distilbert -> ModernBERT etc.) is a no-op."""

    def __init__(self, *, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None and loss_fn is not None:
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return HeadOutput(logits=logits, loss=loss)


class LinearPosHead(Head):
    """`linear` head augmented with two per-token positional features
    concatenated to the encoder hidden state before projection:

      * `normalized_position`: position within the unpadded content,
        in [0, 1]. Uses the *actual* sequence length (from
        `attention_mask.sum(-1)`) rather than the padded `T`, so the
        feature has the same semantics regardless of how short the
        article is relative to the buffer.
      * `is_doc_end_visible`: 1.0 if the article fit inside the buffer
        (some padding present), 0.0 if it was truncated. Constant
        per-row, broadcast across `T`. Lets the head learn
        end-of-article boilerplate ("footer detection") conditioned
        on whether the buffer actually contains the end.

    `is_doc_start_visible` is omitted because the loader never shifts
    the window — token 0 is always the article start — so the feature
    would be constant-true across the whole dataset.
    """

    def __init__(self, *, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + 2, num_labels)

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        b, t, _ = hidden.shape
        # `actual_len[b]` = number of non-pad tokens in row b. clamp_min
        # for safety against an all-pad row (shouldn't happen in
        # practice but the division would NaN).
        actual_len = attention_mask.sum(dim=-1).clamp_min(1).to(hidden.dtype)  # (B,)
        positions = torch.arange(t, device=hidden.device, dtype=hidden.dtype).expand(b, t)
        norm_pos = positions / actual_len.unsqueeze(1)  # (B, T)
        # Tokens past `actual_len` are padding; the loss masks them, but
        # we still feed sane values so the classifier doesn't see noise.
        norm_pos = norm_pos.clamp_max(1.0)
        # End visible <=> at least one padding token, i.e. article fit
        # inside the buffer. Cast to head dtype so the cat is clean.
        end_visible = (attention_mask.sum(dim=-1) < t).to(hidden.dtype).unsqueeze(1).expand(b, t)
        feats = torch.stack([norm_pos, end_visible], dim=-1)  # (B, T, 2)
        x = torch.cat([hidden, feats], dim=-1)
        logits = self.classifier(self.dropout(x))
        loss = None
        if labels is not None and loss_fn is not None:
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return HeadOutput(logits=logits, loss=loss)


class Conv1DHead(Head):
    """1-D convolution over the time axis before the linear projection.

    The frozen-backbone failure mode is fragmented per-token decisions
    — neighbors disagree even when the underlying boilerplate region is
    obvious. A small conv (kernel=3 by default) gives the head a local
    smoothing prior at low parameter cost (~kernel*hidden^2). The
    encoder already has long-range context; the conv just gates the
    final logit on local agreement.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_labels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        # `padding=kernel_size // 2` keeps the time dim the same length
        # (only valid for odd kernels — guard).
        if kernel_size % 2 == 0:
            raise ValueError(f"conv1d head kernel_size must be odd, got {kernel_size}")
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        # Conv1d wants (B, C, T); hidden is (B, T, C).
        x = hidden.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        logits = self.classifier(self.dropout(x))
        loss = None
        if labels is not None and loss_fn is not None:
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return HeadOutput(logits=logits, loss=loss)


# --------------------------------------------------------------------- #
# JIT-scripted CRF kernels                                                #
# --------------------------------------------------------------------- #
#
# The CRF's forward/partition/Viterbi/forward-backward all have an
# unavoidable O(T) sequential recurrence — the math doesn't parallelise
# along the time dimension. At T=4096 the per-step Python op-dispatch
# overhead dominates; we measured ~13 min/epoch vs ~1 min/epoch for the
# non-CRF heads on the same data. Compiling these inner loops with
# `torch.jit.script` fuses the recurrence into TorchScript IR and drops
# the per-step Python cost.
#
# Each function takes the transition matrix as a tensor argument
# (rather than reaching into a module attribute) so they can be plain
# free functions — easier to JIT, easier to unit-test, easier to call.
# The `BinaryCRF` module just owns the parameter and routes arguments.
#
# Padding semantics across all four kernels:
#   * Forward / partition / Viterbi gate on `mask[:, i]` — when the
#     *current* step is invalid, hold the running state unchanged.
#   * Backward (in marginals) gates on `mask[:, i + 1]` — when the *next*
#     step is invalid (we're going right-to-left), skip it. This avoids
#     ever consuming the garbage emission at a padded position.


@torch.jit.script
def _crf_log_partition(
    emissions: torch.Tensor, transitions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """log Z = log-sum-exp over all label sequences. Returns (B,)."""
    t = emissions.size(1)
    alpha = emissions[:, 0, :]
    for i in range(1, t):
        broadcast = (
            alpha.unsqueeze(1)
            + transitions.unsqueeze(0)
            + emissions[:, i, :].unsqueeze(2)
        )
        new_alpha = torch.logsumexp(broadcast, dim=2)
        valid = mask[:, i].unsqueeze(1)
        alpha = torch.where(valid, new_alpha, alpha)
    return torch.logsumexp(alpha, dim=1)


@torch.jit.script
def _crf_gold_score(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Score of the gold label sequence summed along T. Returns (B,)."""
    b = emissions.size(0)
    t = emissions.size(1)
    batch_idx = torch.arange(b, device=emissions.device)
    score = emissions[batch_idx, 0, labels[:, 0]]
    for i in range(1, t):
        emit = emissions[batch_idx, i, labels[:, i]]
        trans = transitions[labels[:, i], labels[:, i - 1]]
        new_score = score + emit + trans
        score = torch.where(mask[:, i], new_score, score)
    return score


@torch.jit.script
def _crf_viterbi(
    emissions: torch.Tensor, transitions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Best label sequence per row. Returns (B, T) long."""
    b = emissions.size(0)
    t = emissions.size(1)
    k = emissions.size(2)
    scores = emissions[:, 0, :].clone()
    # Pre-allocate the backpointer tensor — TorchScript is happier with
    # fixed-shape buffers than with Python lists of tensors.
    history = torch.zeros((t - 1, b, k), dtype=torch.int64, device=emissions.device)
    for i in range(1, t):
        broadcast = scores.unsqueeze(1) + transitions.unsqueeze(0)
        best_prev, best_arg = broadcast.max(dim=2)
        new_scores = best_prev + emissions[:, i, :]
        history[i - 1] = best_arg
        valid = mask[:, i].unsqueeze(1)
        scores = torch.where(valid, new_scores, scores)

    paths = torch.zeros((t, b), dtype=torch.int64, device=emissions.device)
    paths[t - 1] = scores.argmax(dim=1)
    for i in range(t - 2, -1, -1):
        step = history[i]
        paths[i] = step.gather(1, paths[i + 1].unsqueeze(1)).squeeze(1)
    return paths.transpose(0, 1)


@torch.jit.script
def _crf_marginals(
    emissions: torch.Tensor, transitions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-token label posteriors P(y_t = k | x). Returns (B, T, K)."""
    b = emissions.size(0)
    t = emissions.size(1)
    k = emissions.size(2)

    # Forward pass — pre-allocated buffer keeps the autograd graph happy
    # under script and avoids any Python list manipulation.
    alpha_seq = torch.zeros(
        (t, b, k), dtype=emissions.dtype, device=emissions.device
    )
    alpha_seq[0] = emissions[:, 0, :]
    for i in range(1, t):
        broadcast = (
            alpha_seq[i - 1].unsqueeze(1)
            + transitions.unsqueeze(0)
            + emissions[:, i, :].unsqueeze(2)
        )
        new_alpha = torch.logsumexp(broadcast, dim=2)
        valid = mask[:, i].unsqueeze(1)
        alpha_seq[i] = torch.where(valid, new_alpha, alpha_seq[i - 1])

    # Backward pass — beta_seq[T-1] is implicitly zeros (the buffer init).
    beta_seq = torch.zeros(
        (t, b, k), dtype=emissions.dtype, device=emissions.device
    )
    trans_t = transitions.t()
    for i in range(t - 2, -1, -1):
        broadcast = (
            trans_t.unsqueeze(0)
            + emissions[:, i + 1, :].unsqueeze(1)
            + beta_seq[i + 1].unsqueeze(1)
        )
        new_beta = torch.logsumexp(broadcast, dim=2)
        valid = mask[:, i + 1].unsqueeze(1)
        beta_seq[i] = torch.where(valid, new_beta, beta_seq[i + 1])

    log_z = torch.logsumexp(alpha_seq[t - 1], dim=1, keepdim=True)  # (B, 1)
    out = (alpha_seq + beta_seq).transpose(0, 1)  # (B, T, K)
    out = out - log_z.unsqueeze(1)
    return out.exp()


class BinaryCRF(nn.Module):
    """Linear-chain CRF over a small label set (designed for the 2-label
    keep/drop task; works for any K).

    Owns one learned tensor: `transitions[i, j]` = score for moving from
    label `j` at step t to label `i` at step t+1. With K=2 that's 4 params.
    Start/end-of-sequence biases are *not* parameterised — for this task
    most articles start and end with `keep`, and the empirical bias is
    already encoded in the emission distribution. Add if a future
    experiment shows the boundary positions need a separate prior.

    Provides three operations:
      * `nll_loss(emissions, labels, mask)`  — training loss.
      * `decode(emissions, mask)`            — Viterbi best-path inference.
      * `marginals(emissions, mask)`         — forward-backward per-token
                                                 label posteriors, so the
                                                 existing threshold-sweep
                                                 eval can keep working.

    Padded positions (`mask` False) are simply ignored in all three —
    the dynamic-programming recurrences hold the previous-step state and
    skip the update. Labels of `LABEL_IGNORE` (-100) are also treated as
    padded for loss purposes; callers should pre-merge them into `mask`.
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Zero-init: equivalent to no transition prior at step zero of
        # training; the matrix learns the keep→keep / drop→drop affinity
        # from data over the first few epochs.
        self.transitions = nn.Parameter(torch.zeros(num_labels, num_labels))

    # Public API — all three operations delegate to JIT-scripted kernels.

    def nll_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Treat LABEL_IGNORE the same as padded — both should be skipped
        # in the DP. Replace the ignore-label entries with 0 for safe
        # indexing; the mask gate stops them from contributing.
        valid = (labels != LABEL_IGNORE) & attention_mask.bool()
        safe_labels = labels.clamp_min(0)
        log_z = _crf_log_partition(emissions, self.transitions, valid)
        gold = _crf_gold_score(emissions, self.transitions, safe_labels, valid)
        return (log_z - gold).mean()

    def decode(
        self, emissions: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Viterbi: most-likely label sequence per row. Returns (B, T) long."""
        return _crf_viterbi(emissions, self.transitions, attention_mask.bool())

    def marginals(
        self, emissions: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """P(label_t = k | x) via forward-backward. Returns (B, T, K).

        Lets the existing per-token-probability threshold sweep keep
        working when the head is trained with CRF NLL: the per-token
        marginal "drop probability" is the natural CRF analogue of the
        softmax-of-emissions used by the non-CRF heads.
        """
        return _crf_marginals(emissions, self.transitions, attention_mask.bool())


class ComposableHead(Head):
    """All-in-one head with four independent knobs, exposed via config:

      * `conv_kernel`:  0 disables; odd int > 0 enables Conv1d blocks.
      * `conv_stack`:   1 or 2 — how many `(Conv1d + GELU + Dropout)` blocks
                        to stack. Two k=3 blocks give an effective ±2
                        receptive field with a learned non-linearity in
                        between (≈ k=5 reach but more expressive).
      * `conv_mode`:    "full" (in→out via kernel*hidden² params) or
                        "depthwise_separable" (per-channel temporal conv
                        + pointwise channel mix; far cheaper).
      * `linear_stack`: 1 = single `Linear(hidden, num_labels)`; 2 =
                        `Linear(hidden, hidden) + GELU + Linear(hidden, num_labels)`.
      * `crf`:          False = standard CE on logits; True = CRF-NLL
                        loss with Viterbi at inference time.

    Pipeline: `hidden → [(conv+GELU+drop)×conv_stack]? → linear stack → (CRF)?`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_labels: int,
        conv_kernel: int = 0,
        conv_stack: int = 1,
        conv_mode: str = "full",
        linear_stack: int = 1,
        crf: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

        if conv_kernel > 0:
            if conv_kernel % 2 == 0:
                raise ValueError(f"conv_kernel must be odd, got {conv_kernel}")
            if conv_stack < 1:
                raise ValueError(f"conv_stack must be ≥ 1, got {conv_stack}")
            pad = conv_kernel // 2
            blocks: list[nn.Module] = []
            for _ in range(conv_stack):
                if conv_mode == "full":
                    blocks.append(
                        nn.Conv1d(
                            hidden_size, hidden_size, kernel_size=conv_kernel, padding=pad
                        )
                    )
                elif conv_mode == "depthwise_separable":
                    blocks.append(
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=conv_kernel,
                            padding=pad,
                            groups=hidden_size,
                        )
                    )
                    blocks.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=1))
                else:
                    raise ValueError(f"unknown conv_mode: {conv_mode!r}")
                blocks.append(nn.GELU())
                blocks.append(nn.Dropout(dropout))
            self.conv: nn.Module | None = nn.Sequential(*blocks)
        else:
            self.conv = None

        if linear_stack == 1:
            self.linear: nn.Module = nn.Linear(hidden_size, num_labels)
        elif linear_stack == 2:
            self.linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            raise ValueError(f"linear_stack must be 1 or 2, got {linear_stack}")

        # Pre-linear dropout fires only when the conv stack didn't already
        # apply one — keeps single-layer linear head behaviour identical
        # to LinearHead while avoiding double-dropping after the conv path.
        self.pre_linear_dropout: nn.Module = (
            nn.Identity() if self.conv is not None else nn.Dropout(dropout)
        )
        self.crf: BinaryCRF | None = BinaryCRF(num_labels) if crf else None

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        x = hidden
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pre_linear_dropout(x)
        logits = self.linear(x)

        loss = None
        if labels is not None:
            if self.crf is not None:
                loss = self.crf.nll_loss(logits, labels, attention_mask)
            elif loss_fn is not None:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return HeadOutput(logits=logits, loss=loss)


def build_head(cfg: DictConfig | dict, *, hidden_size: int, num_labels: int) -> Head:
    """Instantiate a head from a `head:` config block. New head types
    register here; the config schema is whatever the head's `__init__`
    accepts, accessed via `cfg.get("…")`."""
    head_type = str((cfg.get("type") if cfg else None) or "linear").lower()
    if head_type == "linear":
        return LinearHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=float(cfg.get("dropout", 0.1)) if cfg else 0.1,
        )
    if head_type == "linear_pos":
        return LinearPosHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if head_type == "conv1d":
        return Conv1DHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            kernel_size=int(cfg.get("kernel_size", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if head_type == "composable":
        return ComposableHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            conv_kernel=int(cfg.get("conv_kernel", 0)),
            conv_stack=int(cfg.get("conv_stack", 1)),
            conv_mode=str(cfg.get("conv_mode", "full")),
            linear_stack=int(cfg.get("linear_stack", 1)),
            crf=bool(cfg.get("crf", False)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown head type: {head_type!r}")


__all__ = [
    "BinaryCRF",
    "ComposableHead",
    "Conv1DHead",
    "Head",
    "HeadOutput",
    "LinearHead",
    "LinearPosHead",
    "build_head",
]
