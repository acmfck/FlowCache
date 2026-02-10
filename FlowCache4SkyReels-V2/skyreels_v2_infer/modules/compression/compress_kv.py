import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .utils_kv import cal_similarity, compute_attention_scores
import time

class R1KV:
    def __init__(
        self,
        budget=48600,  
        kernel_size=7,
        mix_lambda=0.07,
        compress_strategy="token",
        **kwargs,
    ):
        self.budget = budget
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        assert compress_strategy in ["token", "frame", "chunk"]
        self.compress_strategy = compress_strategy
        self.kvrange = 0
        self.clean_chunk_idx_even = []  # record cleaned chunks (cond)
        self.clean_chunk_idx_odd = []   # record cleaned chunks (uncond)
        self.budget_block = 0 # number of blocks corresponding to buffer
        self.num_layers = 0

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
    ):
        if self.compress_strategy == "token":
            return self.update_kv_token(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
            )
        elif self.compress_strategy == "frame":
            return self.update_kv_frame_chunk(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
                frame_size=2025,
            )
        elif self.compress_strategy == "chunk":
            return self.update_kv_frame_chunk(
                key_states,
                query_states,
                value_states,
                clean_chunk_tokens,
                frame_size=12150,
            )
        else:
            raise ValueError("Invalid compress strategy")

    def update_kv_token(
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[0]

        attn_weights = compute_attention_scores(query_states, key_states) # memory 48.28 -> 63.21
        
        # Compute attention weights and immediately release intermediate tensors
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, : clean_chunk_tokens],
                dim=-1,
                # dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        ) 
        end_time_attn_weights_sum = time.time()
        # Immediately release attn_weights as it consumes large memory
        del attn_weights
        torch.cuda.empty_cache()

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        del attn_weights_sum
        torch.cuda.empty_cache()

        similarity_cos = cal_similarity(key_states[:clean_chunk_tokens, :, :])
        
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        
        # Release tensors that are no longer needed
        del attn_cache, similarity_cos
        torch.cuda.empty_cache()

        # Calculate number of tokens to keep
        num_to_keep = self.budget

        # topk token
        try:
            indices = final_score.topk(num_to_keep, dim=-1).indices  # shape: (num_kv_heads, num_to_keep)
            del final_score
        except RuntimeError:
            import pdb; pdb.set_trace()
        indices = indices.unsqueeze(-1).expand(-1, -1, head_dim).permute(1, 0, 2)  # shape: (num_to_keep, num_kv_heads, head_dim)
        indices = indices.to(key_states.device)  # Ensure indices and key_states are on the same device


        k_past_compress = key_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices)
        v_past_compress = value_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices)

        k_cur = key_states[clean_chunk_tokens :, :, :]
        v_cur = value_states[clean_chunk_tokens :, :, :]

        key_compress = torch.cat([k_past_compress, k_cur], dim=0)
        value_compress = torch.cat([v_past_compress, v_cur], dim=0)
        
        # Release intermediate tensors
        del k_past_compress, v_past_compress, k_cur, v_cur
        torch.cuda.empty_cache()

        return key_compress, value_compress, indices


    def update_kv_frame_chunk(
        self,
        key_states,
        query_states,
        value_states,
        clean_chunk_tokens,
        frame_size,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[0]

        attn_weights = compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, : clean_chunk_tokens],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)  # shape: (num_kv_heads, clean_chunk_tokens)
            .to(query_states.dtype)
        )

        attn_cache = F.max_pool1d(
            attn_weights_sum.unsqueeze(0),  
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).squeeze(0).to('cpu')  # shape: (num_kv_heads, clean_chunk_tokens)

        similarity_cos = cal_similarity(key_states[:clean_chunk_tokens, :, :]).to('cpu')

        final_score_per_token = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        assert clean_chunk_tokens % frame_size == 0
        num_frames = clean_chunk_tokens // frame_size

        score_frames = final_score_per_token.view(
            key_states.shape[1], num_frames, frame_size
        )

        frame_scores = score_frames.mean(dim=-1)  # shape: (num_kv_heads, num_frames)

        assert self.budget % frame_size == 0
        num_frames_to_keep = self.budget // frame_size

        try:
            frame_indices = frame_scores.topk(num_frames_to_keep, dim=-1).indices
            # shape: (num_kv_heads, num_frames_to_keep)
        except RuntimeError:
            import pdb; pdb.set_trace()

        token_offsets = torch.arange(frame_size, device=key_states.device)  
        frame_indices_expanded = frame_indices.unsqueeze(-1) * frame_size  
        token_indices_per_head = frame_indices_expanded + token_offsets  


        token_indices_flat = token_indices_per_head.view(key_states.shape[1], -1)  # (num_heads, kept_tokens)

        indices_gather = token_indices_flat.permute(1, 0).unsqueeze(-1).expand(-1, -1, head_dim)

        k_past_compress = key_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices_gather)
        v_past_compress = value_states[:clean_chunk_tokens, :, :].gather(dim=0, index=indices_gather)


        k_cur = key_states[clean_chunk_tokens:, :, :]
        v_cur = value_states[clean_chunk_tokens:, :, :]

        key_compress = torch.cat([k_past_compress, k_cur], dim=0)
        value_compress = torch.cat([v_past_compress, v_cur], dim=0)

        return key_compress, value_compress, indices_gather
