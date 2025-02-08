# Original Copyright 2024 Google LLC
# Added by Y.Komori 2025

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import json
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

from gemma import config as gemma_config
from gemma import tokenizer


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        quant: bool,
        attn_type: gemma_config.AttentionType,
        sliding_window_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            quant=quant)

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
        ):
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=gemma_config.AttentionType.GLOBAL,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=attn_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                attn_type = (
                    config.attn_types[i]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results


# 以下は追加実装
    @torch.no_grad()
    def generate_with_initial_embedding(
        self,
        initial_embedding: torch.Tensor,   # shape: [batch_size, 1, hidden_size]
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        instructions: Tuple[str, str] = ("", ""),  # (instruction_prompt0, instruction_prompt1)
    ) -> List[str]:
        """
        VAE埋め込みと、instruction_prompt0, instruction_prompt1（タプルで指定）を条件として与えた上で、
        自己回帰生成を行う例。

        入力シーケンスは以下の各部分の連結となる：
          [instruction_prompt0 のトークン列] +
          [ダミー位置 (後で VAE 埋め込みで上書き)] +
          [instruction_prompt1 のトークン列] +
          [出力トークン（自己回帰的に生成）]
        """
        batch_size = initial_embedding.shape[0]
        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id

        # 1. タプルから instruction_prompt0 と instruction_prompt1 を取得し、トークン化（BOS/EOS なし）
        instruction_prompt0, instruction_prompt1 = instructions
        seq0 = self.tokenizer.encode(instruction_prompt0, bos=False, eos=False) if instruction_prompt0 else []
        seq1 = self.tokenizer.encode(instruction_prompt1, bos=False, eos=False) if instruction_prompt1 else []

        # 2. prefix 部分のトークン列を作成  
        #    ※ teacher_forcing の実装と同様、[seq0] + [dummy] + [seq1] とする  
        #    dummy の位置（後で VAE 埋め込みで置換する）は、seq0 の直後の位置となる
        prefix_tokens = seq0 + [pad_id] + seq1
        prefix_len = len(prefix_tokens)

        # 3. 全シーケンス長は、prefix 長 + 出力トークン数
        total_seq_len = prefix_len + output_len

        # 4. 生成結果用の token ID テンソルを初期化（全体を pad_id で埋める）
        token_ids_tensor = torch.full(
            (batch_size, total_seq_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        # prefix 部分はすべてのサンプルで同じなので、展開して配置する
        prefix_tokens_tensor = torch.tensor(prefix_tokens, dtype=torch.long, device=device)
        token_ids_tensor[:, :prefix_len] = prefix_tokens_tensor.unsqueeze(0).expand(batch_size, prefix_len)

        # 5. KV キャッシュの初期化
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (
                batch_size,
                total_seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size, dtype=dtype, device=device)
            v_cache = torch.zeros(size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # 6. 因果的マスクの作成
        mask = torch.full(
            (1, 1, total_seq_len, total_seq_len),
            float("-inf"),
            dtype=self.config.get_dtype(),
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)

        normalizer = self.config.hidden_size ** 0.5

        # 7. prefix 部分の埋め込みを取得し、dummy の位置（インデックス = len(seq0)）を VAE 埋め込みで上書き
        prefix_embeddings = self.embedder(token_ids_tensor[:, :prefix_len]) * normalizer  # [B, prefix_len, H]
        dummy_index = len(seq0)
        prefix_embeddings[:, dummy_index:dummy_index + 1, :] = initial_embedding * normalizer

        # 8. prefix 部分をまとめてモデルに通して、KV キャッシュに情報を蓄積
        input_positions = torch.arange(prefix_len, device=device)
        hidden_states = self.model(
            hidden_states=prefix_embeddings,
            freqs_cis=self.freqs_cis.index_select(0, input_positions),
            kv_write_indices=input_positions,
            kv_caches=kv_caches,
            mask=mask.index_select(2, input_positions),
        )

        # 9. 自己回帰的に出力トークンを生成
        output_offset = prefix_len  # 生成開始位置
        temperatures = None if not temperature else torch.tensor([temperature] * batch_size, device=device)
        top_ps = torch.tensor([top_p] * batch_size, device=device)
        top_ks = torch.tensor([top_k] * batch_size, device=device)

        for step_idx in range(output_len):
            curr_position = torch.tensor([output_offset], device=device, dtype=torch.long)
            # 直前のトークンを取得
            prev_token_id = token_ids_tensor[:, output_offset - 1].unsqueeze(1)
            prev_token_emb = self.embedder(prev_token_id) * normalizer

            hs = self.model(
                hidden_states=prev_token_emb,
                freqs_cis=self.freqs_cis.index_select(0, curr_position),
                kv_write_indices=curr_position,
                kv_caches=kv_caches,
                mask=mask.index_select(2, curr_position),
            )

            embed_weight = self.embedder.weight
            if self.config.quant:
                embed_weight = embed_weight * self.embedder.weight_scaler.unsqueeze(-1)

            next_token_ids, _ = self.sampler(
                embedding=embed_weight,
                hidden_states=hs,
                output_positions=torch.zeros_like(curr_position),
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
            )

            token_ids_tensor[:, output_offset] = next_token_ids
            output_offset += 1

        # 10. 生成結果のトークン列（prefix 部分を除く）をデコードして出力
        results = []
        for i in range(batch_size):
            out_tokens = token_ids_tensor[i, prefix_len:].tolist()
            if eos_id in out_tokens:
                idx = out_tokens.index(eos_id)
                out_tokens = out_tokens[:idx]
            text = self.tokenizer.decode(out_tokens)
            results.append(text)
        torch.cuda.empty_cache()
        return results



    def encode_texts(
        self,
        texts: List[str],
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Added by komoriyuta
        テキストをトークンIDに変換し、パディングを適用
        
        Args:
            texts: 入力テキストのリスト
            max_seq_len: 最大シーケンス長
            add_bos: BOSトークンを追加するか
            add_eos: EOSトークンを追加するか
            
        Returns:
            token_ids: パディング済みトークンID [batch, seq_len]
            attention_mask: アテンションマスク [batch, seq_len]
        """
        batch_size = len(texts)
        device = self.embedder.weight.device
        pad_id = self.tokenizer.pad_id
        
        # テキスト→トークンID変換
        token_ids = []
        for text in texts:
            tokens = self.tokenizer.encode(text, bos=add_bos, eos=add_eos)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]  # 長すぎる場合は切り詰め
                if add_eos:
                    tokens[-1] = self.tokenizer.eos_id  # 最終位置をEOSで保証
            padding = [pad_id] * (max_seq_len - len(tokens))
            token_ids.append(tokens + padding)
        
        # Tensor変換
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask = (token_ids != pad_id)
        
        return token_ids, attention_mask
    def forward_teacher_forcing(
        self,
        vae_embedding: torch.Tensor,      # shape: [batch_size, hidden_size] (前と同様)
        target_texts: List[str],            # 教師データのテキスト（各サンプル）
        max_seq_len: int = 512,
        instructions: Tuple[str, str] = ("", ""),  # (instruction_prompt0, instruction_prompt1)
    ) -> torch.Tensor:
        """
        VAE埋め込みをprefixとして、instruction_prompt0とinstruction_prompt1を条件として与えた上で、
        target_text の教師強制（Teacher Forcing）を行う例。

        入力シーケンスは以下の各部分の連結となる：
          [instruction_prompt0 のトークン列] +
          [ダミー位置（VAE埋め込みを直接注入）] +
          [instruction_prompt1 のトークン列] +
          [target_text のトークン列（BOS,EOS付き）]
        
        ※ ダミー位置には embedder の出力ではなく、vae_embedding を直接利用する。
        ※ 損失は target_text 部分（連結シーケンスの後半部分）のみ計算する。
        """
        device = self.embedder.weight.device
        batch_size = len(target_texts)
        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id
        normalizer = self.config.hidden_size ** 0.5

        # 1. タプルから instruction_prompt0 と instruction_prompt1 を取得（BOS/EOSは付与しない）
        instruction_prompt0, instruction_prompt1 = instructions
        seq0 = self.tokenizer.encode(instruction_prompt0, bos=False, eos=False) if instruction_prompt0 else []
        seq1 = self.tokenizer.encode(instruction_prompt1, bos=False, eos=False) if instruction_prompt1 else []

        # 2. 各 target_text を BOS/EOS 付きでトークン化
        target_ids_list = []
        for text in target_texts:
            tokens = self.tokenizer.encode(text+'<end_of_turn>', bos=True, eos=True)
            target_ids_list.append(tokens)

        # 3. 各サンプルの最終シーケンスは以下の連結になる：
        #    sequence = seq0 + [dummy] + seq1 + target_tokens
        # ここで dummy は実際のトークンIDは不要（後で vae_embedding により置換する）
        prefix_length = len(seq0) + 1 + len(seq1)  # 条件部分＋VAE埋め込み分

        # 各サンプルで target 部分の最大長を算出
        max_target_len = max_seq_len - prefix_length

        # target 部分が max_target_len を超える場合は切り詰め（EOS を保証）
        processed_target_ids_list = []
        for tokens in target_ids_list:
            if len(tokens) > max_target_len:
                tokens = tokens[:max_target_len]
                tokens[-1] = eos_id  # 最後が必ず EOS となるように
            processed_target_ids_list.append(tokens)

        # 4. 各サンプルの最終トークンID列（固定長 max_seq_len へパディング）
        final_token_ids = []
        for tokens in processed_target_ids_list:
            full_seq = seq0 + [pad_id] + seq1 + tokens  # dummy の位置には一旦 pad_id を入れる
            if len(full_seq) < max_seq_len:
                full_seq = full_seq + [pad_id] * (max_seq_len - len(full_seq))
            else:
                full_seq = full_seq[:max_seq_len]
            final_token_ids.append(full_seq)
        final_token_ids = torch.tensor(final_token_ids, dtype=torch.long, device=device)  # [B, max_seq_len]

        # 5. 注意マスク作成（pad_id でない箇所は True）
        attention_mask = (final_token_ids != pad_id).long()  # [B, max_seq_len]

        # 6. 全体のトークン埋め込みを取得（VAE埋め込みは後で上書きする）
        token_embeddings = self.embedder(final_token_ids) * normalizer  # [B, max_seq_len, H]

        # 7. dummy の位置（インデックス = len(seq0)）に vae_embedding を直接注入
        dummy_index = len(seq0)
        token_embeddings[:, dummy_index:dummy_index+1, :] = vae_embedding.unsqueeze(1) * normalizer

        # 8. 位置情報として各位置の freqs_cis を取得
        positions = torch.arange(max_seq_len, device=device)
        freqs_cis = self.freqs_cis.index_select(0, positions)

        # 9. 因果的マスクの作成
        causal_mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf"), device=device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0)  # [1, max_seq_len, max_seq_len]

        # パディングマスクを因果マスクに反映
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, max_seq_len]
        combined_mask = causal_mask.masked_fill(~padding_mask, float("-inf"))

        # 10. KVキャッシュの初期化
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size, dtype=dtype, device=device)
            v_cache = torch.zeros(size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # 11. モデルへ順伝播
        hidden_states = self.model(
            hidden_states=token_embeddings,
            freqs_cis=freqs_cis,
            kv_write_indices=positions,
            kv_caches=kv_caches,
            mask=combined_mask,
        )

        # 12. ロジットの計算
        embed_weight = self.embedder.weight
        if self.config.quant:
            embed_weight = embed_weight * self.embedder.weight_scaler.unsqueeze(-1)
        logits = torch.matmul(hidden_states, embed_weight.t())  # [B, max_seq_len, vocab_size]

        # 13. 教師信号は target_text 部分のみとする
        #     ※ 入力シーケンスは「条件部 + target_text」となっているため、target_text 部分は
        #         インデックス prefix_length 以降（シフトして予測）
        shift_logits = logits[:, prefix_length:, :]   # 予測対象は位置 prefix_length+1 以降
        shift_labels = final_token_ids[:, prefix_length:]  # 正解は target_text のトークンID

        # パディング部分の損失を除外するためのマスク
        loss_mask = attention_mask[:, prefix_length:].float()  # [B, target_seq_len]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none',
            ignore_index=pad_id,
        )
        loss = (loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()
        
        return loss


#追加実装ここまで
    def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()
