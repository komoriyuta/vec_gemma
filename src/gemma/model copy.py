# Copyright 2024 Google LLC
#
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

    @torch.no_grad()
    def generate_with_initial_embedding(
        self,
        initial_embedding: torch.Tensor,   # [batch_size, 1, hidden_size]
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> List[str]:
        """
        最初の1ステップだけ外部埋め込み `initial_embedding` を使い、
        2ステップ目以降は通常の自己回帰 (得られたトークンIDをEmbeddingsに通す) でテキストを生成する。
        """
        # 前処理
        batch_size = initial_embedding.shape[0]
        total_seq_len = 1 + output_len   # 「最初のステップ + 生成ステップ数」
        pad_id = self.tokenizer.pad_id   # Gemma のトークナイザ
        eos_id = self.tokenizer.eos_id
        
        # KVキャッシュを作成 (generate実装と同じ)
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # token_ids_tensor: 出力先を確保 (PAD埋め)
        token_ids_tensor = torch.full(
            (batch_size, total_seq_len),
            pad_id,
            dtype=torch.long,
            device=device
        )

        # アテンションマスク (上三角マスク)
        # [1, 1, total_seq_len, total_seq_len] 形で -∞ (大きな負数) を配置
        mask = torch.full(
            (1, 1, total_seq_len, total_seq_len),
            -1.0e10,
            dtype=self.config.get_dtype(),
            device=device
        )
        mask = torch.triu(mask, diagonal=1)

        # 1ステップ目: initial_embedding を使って forward
        # -------------------------------------------------------------
        # Gemma ではembeddingを sqrt(hidden_size) でスケールしている
        normalizer = (self.config.hidden_size**0.5)
        hidden_states = initial_embedding * normalizer  # [batch_size, 1, hidden_size]

        # Rotary埋め込み用 index
        positions_0 = torch.zeros(1, dtype=torch.long, device=device)  # [0]
        freqs_cis = self.freqs_cis.index_select(0, positions_0)

        # デコーダ本体へ
        # (hidden_states.shape: [batch_size, 1, hidden_size])
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=positions_0,   # 0番目に書き込み
            kv_caches=kv_caches,
            mask=mask.index_select(2, positions_0),
        )

        # 次トークンをサンプリング
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)

        # output_positions は [0]（= hidden_states の最後のトークン位置）でOK
        next_token_ids, _ = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=positions_0,
            temperatures=(None if temperature is None else torch.full((batch_size,), temperature, device=device)),
            top_ps=torch.full((batch_size,), top_p, dtype=torch.float, device=device),
            top_ks=torch.full((batch_size,), top_k, dtype=torch.long, device=device),
        )
        # token_ids_tensor に書き込み
        token_ids_tensor[:, 0] = next_token_ids

        # 2ステップ目以降: 通常の "トークンID -> Embedding" で自己回帰
        # -------------------------------------------------------------
        for step_idx in range(1, total_seq_len):
            # 前ステップでサンプリングされたトークン
            curr_input_ids = token_ids_tensor[:, step_idx-1].unsqueeze(1)  # shape: [batch_size, 1]
            
            # EOS がもう出ていたら、そのバッチは生成を打ち切ってもOK
            # 今回は簡単のためループ継続しつつEOTだけ明示 (必要に応じてbreak)
            # is_eos = (curr_input_ids == eos_id).all()
            # if is_eos: break

            # 次の positions
            positions_t = torch.tensor([step_idx], device=device, dtype=torch.long)  # [step_idx]
            # アテンションマスクを該当位置だけ抜き出す
            mask_t = mask.index_select(2, positions_t)

            # このステップ入力をEmbedding化
            embed = self.embedder(curr_input_ids)  # [batch_size, 1, hidden_size]
            embed = embed * normalizer

            # デコーダ forward
            hidden_states = self.model(
                hidden_states=embed,
                freqs_cis=self.freqs_cis.index_select(0, positions_t),
                kv_write_indices=positions_t,
                kv_caches=kv_caches,
                mask=mask_t,
            )

            # sampler で次トークンをサンプリング
            next_tokens, _ = self.sampler(
                embedding=embedder_weight,
                hidden_states=hidden_states,
                output_positions=torch.zeros_like(positions_t),  # ローカル(=1トークン分)ではindex=0
                temperatures=(None if temperature is None else torch.full((batch_size,), temperature, device=device)),
                top_ps=torch.full((batch_size,), top_p, dtype=torch.float, device=device),
                top_ks=torch.full((batch_size,), top_k, dtype=torch.long, device=device),
            )
            # token_ids_tensor に格納
            token_ids_tensor[:, step_idx] = next_tokens

        # バッチごとの生成結果を取り出して文字列化
        results = []
        for i in range(batch_size):
            # 最初から最後まで (ただし EOS が出たらそこまで切り詰めるなど)
            out_tokens = token_ids_tensor[i].tolist()
            # PAD や EOS を取り除く
            if eos_id in out_tokens:
                eos_pos = out_tokens.index(eos_id)
                out_tokens = out_tokens[:eos_pos]
            out_text = self.tokenizer.decode(out_tokens)
            results.append(out_text)

        return results
    
    def _prepare_inputs(
        self,
        texts: Union[str, List[str]],
        device: torch.device,
        add_bos: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of text. 
        Returns a dict containing:
            "input_ids"      -> (batch_size, seq_len)
            "attention_mask" -> (batch_size, seq_len) [1=real, 0=pad]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize each text
        batch_encodings = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if add_bos and hasattr(self.tokenizer, "bos_id"):
                token_ids = [self.tokenizer.bos_id] + token_ids
            batch_encodings.append(token_ids)

        # Find maximum length
        max_len = max(len(seq) for seq in batch_encodings)
        
        # Pad each sequence
        padded_input_ids = []
        attention_masks = []
        for seq in batch_encodings:
            pad_size = max_len - len(seq)
            padded_seq = seq + [self.tokenizer.pad_id] * pad_size
            mask_seq = [1] * len(seq) + [0] * pad_size
            padded_input_ids.append(padded_seq)
            attention_masks.append(mask_seq)

        input_ids = torch.tensor(padded_input_ids, device=device, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, device=device, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def _prepare_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates an upper-triangular causal mask (1, 1, seq_len, seq_len) 
        with large negative values above the diagonal.
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len), 
            -2.3819763e38,  # effectively -inf in FP16
            device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward_train_with_vector(
        self,
        texts: Union[str, List[str]],
        vae_embeddings: torch.Tensor,
        device: torch.device,
        beta: float = 0.1,
        kl_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        A simple training-like forward that:
          1) Tokenizes input texts,
          2) Overwrites the first token embedding in each sequence with `vae_embeddings`,
          3) Does a causal language modeling loss (next-token prediction),
          4) Optionally adds a KL loss term (e.g. from VAE).
        """
        # 1) Prepare inputs
        inputs = self._prepare_inputs(texts, device, add_bos=True)
        input_ids = inputs["input_ids"]           # (batch_size, seq_len)
        attention_mask = inputs["attention_mask"] # (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # 2) Create a full causal mask for the entire sequence
        # shape: (1, 1, seq_len, seq_len)
        mask = self._prepare_causal_mask(seq_len, device)

        # 3) Build embeddings via the standard embedding layer
        #    Then scale by sqrt(hidden_size)
        embeddings = self.embedder(input_ids)  # (batch_size, seq_len, hidden_size)
        normalizer = (self.config.hidden_size ** 0.5)
        embeddings = embeddings * normalizer

        # 4) Overwrite the first token's embedding with vae_embeddings
        #    vae_embeddings must be shape [batch_size, hidden_size].
        #    We'll place them at embeddings[:, 0, :].
        embeddings[:, 0, :] = vae_embeddings

        # 5) Prepare KV caches for a single forward pass
        #    (One cache per layer; shape is [batch_size, seq_len, num_kv_heads, head_dim])
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim
            )
            cache_dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=cache_dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=cache_dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # 6) Pass through the model in a single (non-incremental) forward
        #    We supply positions = [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        hidden_states = self.model(
            hidden_states=embeddings,                   # pre-built embeddings
            freqs_cis=self.freqs_cis.index_select(0, positions),
            kv_write_indices=positions,                 # write each token at its position
            kv_caches=kv_caches,
            mask=mask,
        )  # (batch_size, seq_len, hidden_size)

        # 7) Compute next-token logits: hidden_states -> linear projection
        if self.config.quant:
            embedder_weight = (
                self.embedder.weight * 
                self.embedder.weight_scaler.unsqueeze(-1)
            )
        else:
            embedder_weight = self.embedder.weight
        logits = F.linear(hidden_states, embedder_weight)  # (batch_size, seq_len, vocab_size)

        # 8) Language modeling loss: predict input_ids[:, 1:] from hidden_states[:, :-1]
        #    i.e., shift
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:].contiguous()   # (batch_size, seq_len-1)

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_id
        )

        total_loss = ce_loss
        if kl_loss is not None:
            total_loss = total_loss + beta * kl_loss

        metrics = {
            "loss": total_loss(),
            "ce_loss": ce_loss(),
            "ppl": torch.exp(ce_loss()),
        }
        if kl_loss is not None:
            metrics["kl_loss"] = kl_loss()
            metrics["beta"] = torch.tensor(beta, device=device)

        return total_loss, metrics


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
