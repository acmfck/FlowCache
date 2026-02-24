# FlowCache4MAGI-1 代码结构说明

本文档用于说明 `FlowCache4MAGI-1` 的代码组织方式、核心调用链路，以及当前仓库中各文件的作用。

## 1. 项目目录总览

```text
FlowCache4MAGI-1/
├── config/                 # 模型/推理 JSON 配置
├── example/assets/         # 示例资源（图像、special tokens）
├── inference/              # 核心推理代码（模型、并行、缓存、pipeline）
├── scripts/                # 一键运行脚本（单条推理、批量采样、指标）
├── tools/                  # 工具脚本（视频指标）
├── yaml_config/            # 采样/附加参数 YAML
├── sample_video.py         # 多 GPU 批量采样入口
├── requirements.txt        # Python 依赖
└── README.md               # 子项目使用说明
```

## 2. 核心执行链路

### 2.1 单视频 FlowCache（T2V/V2V）

1. `scripts/single_run/flowcache_t2v.sh` 或 `scripts/single_run/flowcache_v2v.sh`
2. 调用 `inference/pipeline/flowcache.py`
3. `setup_flowcache(...)` 给 `SampleTransport` 打补丁（chunk 复用 + KV 压缩）
4. 初始化 `MagiPipeline`（`inference/pipeline/pipeline.py`）
5. 文本处理：`inference/pipeline/prompt_process.py`
6. 主生成调度：`inference/pipeline/video_generate.py`
7. 视频编解码：`inference/pipeline/video_process.py`
8. 写出 mp4 文件到 `output/...`

### 2.2 TeaCache 基线

1. `scripts/single_run/teacache_*.sh`
2. 调用 TeaCache 入口（仓库里对应逻辑在 `inference/pipeline/teacache.py`）
3. 通过 `TeaCache` 策略执行“全输出复用”基线

### 2.3 Benchmark 批量采样

1. `scripts/sample/*.sh`
2. 调用 `sample_video.py`，读取 `yaml_config/sample/*.yaml`
3. 按 benchmark（VBench/PhysicsIQ）分发样本到多 GPU 进程
4. 每个子进程独立加载 `MagiPipeline` 并生成结果

## 3. 各文件作用（逐文件）

## 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 本子项目说明：安装、运行方式、参数解释。 |
| `requirements.txt` | Python 依赖列表（MAGI-1 + FlowCache 运行所需）。 |
| `sample_video.py` | 多 GPU 批量采样主入口，支持 VBench/PhysicsIQ，支持 FlowCache/TeaCache 策略切换。 |

## config

| 文件 | 作用 |
|---|---|
| `config/single_run/flowcache_t2v.json` | 单条 T2V 推理配置（模型结构、runtime、权重路径、推理步数等）。 |
| `config/single_run/flowcache_v2v.json` | 单条 V2V 推理配置（含 noise2clean_kvrange 等 v2v 参数）。 |
| `config/sample/vbench.json` | VBench 批量采样配置模板。 |
| `config/sample/physicsiq.json` | PhysicsIQ 批量采样配置模板。 |

## yaml_config

| 文件 | 作用 |
|---|---|
| `yaml_config/single_run/config.yaml` | FlowCache 额外参数（reuse 阈值、warmup、KV 压缩参数、日志开关）。 |
| `yaml_config/sample/flowcache_vbench.yaml` | VBench + FlowCache 的采样配置。 |
| `yaml_config/sample/flowcache_physicsiq.yaml` | PhysicsIQ + FlowCache 的采样配置。 |
| `yaml_config/sample/teacache_vbench.yaml` | VBench + TeaCache 的采样配置。 |
| `yaml_config/sample/teacache_physicsiq.yaml` | PhysicsIQ + TeaCache 的采样配置。 |

## scripts

| 文件 | 作用 |
|---|---|
| `scripts/single_run/flowcache_t2v.sh` | FlowCache 单条文本生视频脚本（设置环境变量后调用 `flowcache.py`）。 |
| `scripts/single_run/flowcache_v2v.sh` | FlowCache 单条视频续写脚本。 |
| `scripts/single_run/teacache_t2v.sh` | TeaCache 单条 T2V 基线脚本。 |
| `scripts/single_run/teacache_v2v.sh` | TeaCache 单条 V2V 基线脚本。 |
| `scripts/sample/flowcache_vbench.sh` | FlowCache 的 VBench 批量采样脚本（循环不同维度）。 |
| `scripts/sample/flowcache_physicsiq.sh` | FlowCache 的 PhysicsIQ 批量采样脚本。 |
| `scripts/sample/teacache_vbench.sh` | TeaCache 的 VBench 批量采样脚本。 |
| `scripts/sample/teacache_physicsiq.sh` | TeaCache 的 PhysicsIQ 批量采样脚本。 |
| `scripts/metric.sh` | 调用 `tools/video_metrics.py` 计算 PSNR/SSIM/LPIPS。 |

## tools

| 文件 | 作用 |
|---|---|
| `tools/video_metrics.py` | 读取两段视频并计算 PSNR、SSIM、LPIPS。 |

## example/assets

| 文件 | 作用 |
|---|---|
| `example/assets/image.jpeg` | 示例图像输入资源。 |
| `example/assets/special_tokens.npz` | 特殊 token 资源，供 prompt 处理阶段使用。 |

## inference 顶层与 common

| 文件 | 作用 |
|---|---|
| `inference/__init__.py` | 包标识文件（当前为空）。 |
| `inference/common/__init__.py` | common 模块导出聚合。 |
| `inference/common/common_utils.py` | 通用工具：环境变量判断、整数除法、随机种子设置。 |
| `inference/common/config.py` | 配置数据结构定义：`ModelConfig`/`RuntimeConfig`/`EngineConfig`/`MagiConfig`。 |
| `inference/common/dataclass.py` | 推理期数据结构：注意力参数封装、`InferenceParams`。 |
| `inference/common/logger.py` | 分布式日志封装（按 rank 打印）。 |
| `inference/common/timer.py` | 路径级事件计时器。 |

## inference/infra/checkpoint

| 文件 | 作用 |
|---|---|
| `inference/infra/checkpoint/__init__.py` | checkpoint 接口导出。 |
| `inference/infra/checkpoint/checkpointing.py` | safetensors 分片并行加载、PP 场景切分、权重加载主逻辑。 |

## inference/infra/distributed

| 文件 | 作用 |
|---|---|
| `inference/infra/distributed/__init__.py` | distributed 能力导出聚合。 |
| `inference/infra/distributed/dist_utils.py` | 分布式初始化、设备与 world size 查询。 |
| `inference/infra/distributed/parallel_state.py` | TP/PP/DP/CP 并行组构建与 rank 管理。 |

## inference/infra/parallelism

| 文件 | 作用 |
|---|---|
| `inference/infra/parallelism/__init__.py` | 并行模块导出聚合。 |
| `inference/infra/parallelism/context_parallel.py` | Context Parallel/Ulysses 相关通信与前后处理。 |
| `inference/infra/parallelism/pipeline_parallel.py` | Pipeline Parallel 调度器（发送/接收张量）。 |
| `inference/infra/parallelism/tile_parallel.py` | tile 级并行辅助与张量拼接处理。 |

## inference/model/dit

| 文件 | 作用 |
|---|---|
| `inference/model/dit/__init__.py` | DiT 模型导出：`get_dit`、`VideoDiTModel`。 |
| `inference/model/dit/dit_model.py` | DiT 高层模型封装、配置构建与 checkpoint 装载。 |
| `inference/model/dit/dit_module.py` | DiT 底层网络模块（注意力、embedding、并行算子等）。 |

## inference/model/t5

| 文件 | 作用 |
|---|---|
| `inference/model/t5/__init__.py` | T5 文本编码器导出。 |
| `inference/model/t5/t5_model.py` | T5 embedding 封装、HF 下载与 tokenizer/encoder 构建。 |

## inference/model/vae

| 文件 | 作用 |
|---|---|
| `inference/model/vae/__init__.py` | VAE 相关类导出。 |
| `inference/model/vae/vae_model.py` | VAE 高层封装与 `AutoModel.from_pretrained` 路径。 |
| `inference/model/vae/vae_module.py` | VAE 网络结构与 attention/rotary 等底层实现。 |

## inference/pipeline 主流程

| 文件 | 作用 |
|---|---|
| `inference/pipeline/__init__.py` | pipeline 模块导出（`MagiPipeline`、`SampleTransport`）。 |
| `inference/pipeline/pipeline.py` | 用户侧主接口：`run_text_to_video` / `run_image_to_video` / `run_video_to_video`。 |
| `inference/pipeline/video_generate.py` | 逐 chunk 生成调度核心：时序安排、KV range、积分更新、chunk 产出。 |
| `inference/pipeline/video_process.py` | 视频前后处理：i2v/v2v 输入处理、VAE 编解码、保存 mp4。 |
| `inference/pipeline/prompt_process.py` | 文本 token/embedding 处理、特殊 token 注入、T5 调用。 |
| `inference/pipeline/entry.py` | 基础 CLI 入口（不启用 FlowCache/TeaCache 补丁时可直接跑）。 |
| `inference/pipeline/utils.py` | 通用工具（张量显存统计）。 |
| `inference/pipeline/memory_monitor.py` | 推理中 residual memory 监控工具。 |

## inference/pipeline/cache（统一缓存抽象层）

| 文件 | 作用 |
|---|---|
| `inference/pipeline/cache/__init__.py` | cache 子模块导出聚合。 |
| `inference/pipeline/cache/base.py` | 缓存抽象基类：`CacheStrategy`、`OutputCache`、`KVCompressor`。 |
| `inference/pipeline/cache/cachereuse.py` | 输出复用策略实现：`TeaCache`（全复用）与 `ChunkWiseCache`（分块复用）。 |
| `inference/pipeline/cache/kv_compressor.py` | KV 压缩管理器（何时压缩、按层压缩、范围更新）。 |
| `inference/pipeline/cache/utils.py` | cache 辅助函数：动态 KV 区间、chunk 信息构造、压缩条件判断。 |

## inference/pipeline/flowcache 与 teacache 入口

| 文件 | 作用 |
|---|---|
| `inference/pipeline/flowcache.py` | FlowCache 主入口：给 `SampleTransport` 注入分块复用 + KV 压缩逻辑，并执行推理。 |
| `inference/pipeline/teacache.py` | TeaCache 主入口：给 `SampleTransport` 注入全复用逻辑，并执行推理。 |

## inference/pipeline/kvcompress（R-KV 相关补丁实现）

| 文件 | 作用 |
|---|---|
| `inference/pipeline/kvcompress/__init__.py` | KV 压缩包导出（`replace_magi`）。 |
| `inference/pipeline/kvcompress/monkeypatch.py` | 对 MAGI 注意力模块打补丁，接入压缩实现。 |
| `inference/pipeline/kvcompress/modeling.py` | 补丁后注意力初始化逻辑（`MagiAttention_init`）。 |
| `inference/pipeline/kvcompress/kv_compressor.py` | 压缩器实现（token/frame/chunk 等粒度策略）。 |
| `inference/pipeline/kvcompress/utils.py` | 相似度/注意力评分与 chunk KV 范围追踪。 |
| `inference/pipeline/kvcompress/.DS_Store` | macOS Finder 元数据文件，对运行无功能作用。 |

## 4. 运行产物目录（非源码）

| 路径 | 作用 |
|---|---|
| `output/` | 推理输出目录（视频与日志），运行时自动生成。 |

## 5. 函数级调用图

### 5.1 FlowCache（T2V）主函数链

```text
scripts/single_run/flowcache_t2v.sh
  -> inference/pipeline/flowcache.py:main
    -> parse_arguments
    -> load_config (可选，读取 additional_config)
    -> setup_flowcache
       -> SampleTransport.forward_velocity = flowcache_forward_velocity
       -> SampleTransport.integrate_velocity = flowcache_integrate_velocity
    -> replace_magi (注入 KV 压缩相关注意力补丁)
    -> MagiPipeline.__init__
    -> MagiPipeline.run_text_to_video
      -> MagiPipeline._run
        -> get_txt_embeddings
        -> get_dit
        -> generate_per_chunk
          -> extract_feature_for_inference
          -> SampleTransport.__init__ / init_work_queue
          -> SampleTransport.walk (循环)
            -> flowcache_forward_velocity
              -> model.forward = _create_flowcache_model_forward_fn(...).model_forward
              -> model.get_embedding_and_meta = _new_get_embedding_and_meta
              -> forward_dispatcher(...)
                -> model_forward(...)
                  -> ChunkWiseCache.compute_feature_metric
                  -> ChunkWiseCache.should_reuse
                  -> videodit_blocks.forward
            -> flowcache_integrate_velocity
              -> self.integrate (仅对未复用 chunk)
              -> _check_and_compress_kv
                -> KVCacheCompressor.should_compress
                -> KVCacheCompressor.compress
              -> _return_clean_chunk
          -> post_chunk_process
            -> decode_chunk
              -> VaeHelper.decode
        -> save_video_to_disk
```

### 5.2 FlowCache（V2V）相对 T2V 的差异链

```text
MagiPipeline.run_video_to_video
  -> process_prefix_video
    -> ffmpeg_v2v
    -> encode_prefix_video
      -> VaeHelper.get_vae
      -> VaeHelper.encode
  -> MagiPipeline._run (后续与 T2V 主链一致)
```

### 5.3 `SampleTransport.walk` 循环级调用关系

```text
walk():
  1) 先对每个 infer_idx 调一次 forward_velocity(..., step=0)
  2) while work_queue 非空:
       - integrate_velocity(current_step)
       - 若某个 chunk 完成全部去噪，则 yield clean_chunk
       - 若当前任务未结束，push 下一步状态，再调 forward_velocity(step+1)
```

### 5.4 批量采样（Benchmark）函数链

```text
scripts/sample/*.sh
  -> sample_video.py:main
    -> load_yaml_config
    -> setup_save_path
    -> load_vbench_samples / load_physicsiq_samples
    -> 多进程启动 worker_process
      -> setup_environment
      -> configure_reuse_strategy
         -> configure_flowcache / configure_teacache
      -> MagiPipeline(...)
      -> process_vbench_sample / process_physicsiq_sample
```

## 6. 备注

1. 该子项目是基于 MAGI-1 的加速实现，很多核心模型组件复用 MAGI 的结构。
2. 单条与批量采样脚本中有若干占位路径（`/path/to/...`），使用前需要替换为本机路径。
3. 若你只关注“实际推理主链”，优先阅读：`scripts/single_run/*` -> `inference/pipeline/flowcache.py` -> `inference/pipeline/video_generate.py` -> `inference/pipeline/video_process.py`。
