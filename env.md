下面按 `FlowCache/README_ZH.md` 给你一套可直接执行的环境配置流程。

建议先看结论：`MAGI-1` 和 `SkyReels-V2` 依赖版本差异较大，最好分两个 conda 环境装，别混在一个环境里。

**1) 通用准备（一次即可）**
```bash
cd /mnt/42_store/fck/FlowCache
nvidia-smi
python -V
```

可选系统依赖（Ubuntu）：
```bash
sudo apt update
sudo apt install -y ffmpeg git
```

**2) 配置 MAGI-1 环境（FlowCache4MAGI-1）**
`flashinfer` 在 `FlowCache4MAGI-1/requirements.txt:8` 指向 `cu124/torch2.4`，建议直接用 PyTorch 2.4 + CUDA 12.4 轮子。

```bash
conda create -n flowcache-magi python=3.10 -y
conda activate flowcache-magi
pip install -U pip

# 建议版本组合
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 代码里直接 import flash_attn，需手动装（requirements 里注释掉了）
pip install flash-attn --no-build-isolation

# 安装 FlowCache 依赖
pip install -r FlowCache4MAGI-1/requirements.txt
```

然后改模型路径（必须）：
- `FlowCache4MAGI-1/config/single_run/flowcache_t2v.json:60`
- `FlowCache4MAGI-1/config/single_run/flowcache_t2v.json:61`
- `FlowCache4MAGI-1/config/single_run/flowcache_t2v.json:63`

把 `CUDA_VISIBLE_DEVICES` 改成你机器可用卡号：
- `FlowCache4MAGI-1/scripts/single_run/flowcache_t2v.sh:20`

运行：
```bash
cd FlowCache4MAGI-1
bash scripts/single_run/flowcache_t2v.sh
```

**3) 配置 SkyReels-V2 环境（FlowCache4SkyReels-V2）**
```bash
conda create -n flowcache-sky python=3.10 -y
conda activate flowcache-sky
pip install -U pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r FlowCache4SkyReels-V2/requirements.txt
```

改 GPU 号：
- `FlowCache4SkyReels-V2/run_flowcache_fast.sh:1`

运行：
```bash
cd FlowCache4SkyReels-V2
bash run_flowcache_fast.sh
```

模型会通过 Hugging Face 自动下载（见 `FlowCache4SkyReels-V2/skyreels_v2_infer/modules/__init__.py:13`），如果拉取失败先：
```bash
huggingface-cli login
```

**4) 快速自检**
```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果你愿意，我可以下一步直接帮你把两个脚本里的 `CUDA_VISIBLE_DEVICES` 和 MAGI 的 3 个权重路径按你机器实际路径改好。