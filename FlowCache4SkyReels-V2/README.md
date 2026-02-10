# FLOW CACHING FOR AUTOREGRESSIVE VIDEO GENERATION

This repository provides the official implementation of **FlowCache** on **SkyReels-V2** model, a caching-based acceleration method for autoregressive video generation models.


## 🚀 Installation

Please follow the installation instructions provided in the [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2), as this implementation is built on top of SkyReels-V2.

---

## ▶️ Usage

### 1. Video Generation

Run accelerated generation using FlowCache:

```bash
# FlowCache with KV cache compression
bash run_flowcache_kvcompress.sh

# FlowCache without KV cache compression (fast)
bash run_flowcache_fast.sh

# FlowCache without KV cache compression (slow)
bash run_flowcache_slow.sh

```
For the allreuse implementation of Teacache, please refer to the official SkyReels-V2 repository.

---

## ⚙️ Key Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `--model_id` | Model identifier (e.g., `SkyReels-V2/SkyReels-V2-DF-1.3B-540P`) | `Skywork/SkyReels-V2-DF-1.3B-540P` |
| `--resolution` | Video resolution: `540P` or `720P` | `540P` |
| `--num_frames` | Total number of frames to generate | `97` |
| `--base_num_frames` | Base number of frames for autoregressive generation | `97` |
| `--overlap_history` | Number of overlapping frames between segments | `17` |
| `--ar_step` | Autoregressive step size for long video generation | `5` |
| `--causal_block_size` | Block size for causal attention | `5` |
| `--inference_steps` | Number of denoising steps | `50` |
| `--guidance_scale` | Classifier-free guidance scale | `6.0` |
| `--teacache_thresh` | TeaCache threshold for cache reuse (higher = faster) | `0.1` |
| `--use_compress` | Enable KV compression for KV cache | `False` |
| `--budget_block` | Number of blocks for KV cache budget | `1` |
| `--addnoise_condition` | Noise condition for long video consistency | `20` |
| `--seed` | Random seed for reproducible generation | `1024` |

---