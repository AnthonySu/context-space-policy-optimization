"""Decision Transformer with external context injection.

Implements a GPT-style causal transformer that maps
(return-to-go, state, action) sequences to predicted actions.
The key extension for CSPO: the ``act`` method accepts an optional
``context_prefix`` parameter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer with external context injection.

    Maps (return-to-go, state, action) token triples through a
    causal transformer to predict actions.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state/observation space.
    act_dim : int
        Dimensionality of the action space.
    n_embd : int
        Transformer hidden dimension.
    n_head : int
        Number of attention heads.
    n_layer : int
        Number of transformer blocks.
    context_length : int
        Maximum number of timesteps in the context window.
    max_ep_len : int
        Maximum episode length (for timestep embeddings).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 3,
        context_length: int = 20,
        max_ep_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.n_embd = n_embd
        self.context_length = context_length
        self.max_ep_len = max_ep_len

        # Token type count: 3 tokens per timestep (rtg, state, action)
        block_size = 3 * context_length

        # Embedding layers
        self.embed_timestep = nn.Embedding(max_ep_len + 1, n_embd)
        self.embed_return = nn.Linear(1, n_embd)
        self.embed_state = nn.Linear(state_dim, n_embd)
        self.embed_action = nn.Linear(act_dim, n_embd)

        self.embed_ln = nn.LayerNorm(n_embd)

        # Transformer
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )

        # Prediction head (predict action from state token)
        self.predict_action = nn.Sequential(
            nn.Linear(n_embd, act_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: predict actions for all timesteps.

        Parameters
        ----------
        states : Tensor
            Shape ``(batch, seq_len, state_dim)``.
        actions : Tensor
            Shape ``(batch, seq_len, act_dim)``.
        returns_to_go : Tensor
            Shape ``(batch, seq_len, 1)``.
        timesteps : Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Predicted actions, shape ``(batch, seq_len, act_dim)``.
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality + add timestep embedding
        time_emb = self.embed_timestep(timesteps)  # (B, T, E)
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        rtg_emb = self.embed_return(returns_to_go) + time_emb

        # Interleave: [r_0, s_0, a_0, r_1, s_1, a_1, ...]
        # Shape: (B, 3*T, E)
        stacked = torch.stack(
            [rtg_emb, state_emb, action_emb], dim=2
        ).reshape(batch_size, 3 * seq_len, self.n_embd)

        stacked = self.embed_ln(stacked)

        # Transformer
        hidden = self.blocks(stacked)

        # Extract state positions (index 1, 4, 7, ... = 3*t + 1)
        state_hidden = hidden[:, 1::3, :]  # (B, T, E)

        # Predict action from state representation
        action_preds = self.predict_action(state_hidden)
        return action_preds

    @torch.no_grad()
    def act(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        context_prefix: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Select action, optionally using an injected context prefix.

        This is the key method for CSPO: by changing ``context_prefix``,
        we change the DT's behavior without retraining.

        Parameters
        ----------
        states : Tensor
            Shape ``(1, T, state_dim)``.
        actions : Tensor
            Shape ``(1, T, act_dim)``.
        returns_to_go : Tensor
            Shape ``(1, T, 1)``.
        timesteps : Tensor
            Shape ``(1, T)``.
        context_prefix : dict, optional
            If provided, prepend these tokens before the current context.
            Keys: ``"states"``, ``"actions"``, ``"returns_to_go"``,
            ``"timesteps"`` — each a Tensor.

        Returns
        -------
        Tensor
            Selected action, shape ``(1, act_dim)``.
        """
        self.eval()

        if context_prefix is not None:
            # Prepend the context prefix
            states = torch.cat(
                [context_prefix["states"], states], dim=1
            )
            actions = torch.cat(
                [context_prefix["actions"], actions], dim=1
            )
            returns_to_go = torch.cat(
                [context_prefix["returns_to_go"], returns_to_go], dim=1
            )
            timesteps = torch.cat(
                [context_prefix["timesteps"], timesteps], dim=1
            )

        # Truncate to context_length
        states = states[:, -self.context_length :]
        actions = actions[:, -self.context_length :]
        returns_to_go = returns_to_go[:, -self.context_length :]
        timesteps = timesteps[:, -self.context_length :]

        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[:, -1, :]  # Last timestep action
