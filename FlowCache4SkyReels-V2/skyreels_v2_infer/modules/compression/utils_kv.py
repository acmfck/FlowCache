import math
import torch

def compute_attention_scores(query_states, key_states_cpu, pooling="max"):
    """
    query_states: [q_len, q_heads, head_dim] on GPU
    key_states_cpu: [kv_len, kv_heads, head_dim] on CPU
    """

    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states_cpu.shape
    query_group_size = q_heads // kv_heads

    device = query_states.device  # GPU


    if query_group_size == 1:
        chunk_size = query_states.shape[0]
        attn_weights_list = []

        for i in range(0, kv_len, chunk_size):
            end_i = min(i + chunk_size, kv_len)
            k_chunk = key_states_cpu[i:end_i].to(device) 

            attn_chunk = torch.bmm(
                query_states.transpose(0, 1),  # [q_heads, q_len, head_dim]
                # k_chunk.transpose(1, 2)
                k_chunk.permute(1, 2, 0)        # [kv_heads, head_dim, chunk_size]
            ) / math.sqrt(head_dim)            # [kv_heads, q_len, chunk_size]
            attn_weights_list.append(attn_chunk)

        attn_weights = torch.cat(attn_weights_list, dim=2)  # [kv_heads, q_len, kv_len]
        return attn_weights

    else:
        # query_states: [q_len, q_heads, head_dim] -> reshape to group
        query_states = query_states.view(q_len, kv_heads, query_group_size, head_dim)
        # [q_len, kv_heads, g, head_dim] -> permute to [kv_heads, g, q_len, head_dim]
        query_states = query_states.permute(1, 2, 0, 3).contiguous()  # [kv_heads, g, q_len, head_dim]

        if pooling == "mean":
            attn_weights_sum = None
            count = 0
        elif pooling == "max":
            attn_weights_max = None
        else:
            raise ValueError("Pooling method not supported")

        for g in range(query_group_size):
            q_group = query_states[:, g, :, :]  # [kv_heads, q_len, head_dim]

            chunk_size = 12150
            group_attn_chunks = []

            for i in range(0, kv_len, chunk_size):
                end_i = min(i + chunk_size, kv_len)
                k_chunk = key_states_cpu[i:end_i].to(device)  # [chunk_size, kv_heads, head_dim]
                k_chunk = k_chunk.permute(1, 2, 0)  # [kv_heads, head_dim, chunk_size]

                attn_chunk = torch.bmm(q_group, k_chunk) / math.sqrt(head_dim)
                group_attn_chunks.append(attn_chunk)

            group_attn = torch.cat(group_attn_chunks, dim=2)  # [kv_heads, q_len, kv_len]

            if pooling == "mean":
                if attn_weights_sum is None:
                    attn_weights_sum = group_attn
                else:
                    attn_weights_sum += group_attn
                count += 1
            elif pooling == "max":
                if attn_weights_max is None:
                    attn_weights_max = group_attn
                else:
                    attn_weights_max = torch.max(attn_weights_max, group_attn)

        if pooling == "mean":
            attn_weights = attn_weights_sum / count
        elif pooling == "max":
            attn_weights = attn_weights_max

        return attn_weights


def cal_similarity(
    key_states,
    threshold=0.5,
):
    k = key_states.permute(1, 0, 2).to('cuda')  # shape: [kv_heads, kv_len, head_dim]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    return similarity_cos.mean(dim=1).softmax(dim=-1)

