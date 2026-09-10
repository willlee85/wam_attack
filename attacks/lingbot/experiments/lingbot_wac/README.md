# LingBot-VA WAC 注意力攻击

该目录把 FastWAM 实验中的动作到视觉注意力攻击迁移到 LingBot-VA。目标仍然是最小化

```text
A_{A<-V} = mean_{layer, head, action query}(sum over cached visual keys attention_probability)
```

但 LingBot-VA 的调用链不同：

```text
LIBERO RGB -> streaming VAE -> visual latent tokens -> per-layer KV cache
           -> action diffusion queries cached visual keys -> action chunk
```

## 实验协议

- 每个 episode 只在首个 observation 上运行一次 PGD。
- 默认同时攻击 `agentview` 和 `eye_in_hand`，约束为 RGB01 空间的 `L_inf=8/255`。
- 得到的固定扰动会应用到该 episode 后续所有视觉 observation，包括 KV-cache 更新。
- 为控制显存，攻击代理默认只展开 1 个 video denoising step；正常 rollout 仍使用完整 LingBot 推理。
- 默认目标层为 25--29，动作去噪步为 0。
- 使用相同 `--inference-seed` 分别运行 clean 与 WAC，才能做严格的配对成功率比较。

## 运行

先启动 WAC 专用的单 GPU、非 FSDP server：

```bash
GPU_ID=0 PORT=29056 bash evaluation/libero/launch_wac_server.sh
```

不要用原来的 `launch_server.sh` 生成 WAC：PyTorch FSDP2 的冻结推理分片会在
forward 后释放参数，而本攻击需要跨视觉 forward 和动作 forward 对输入求梯度。
普通 clean 评估仍可继续使用原 launcher；为了在同一进程做 paired clean/WAC，
也可以全程使用 WAC server，关闭 `--wac-attack` 即为正常推理。

攻击运行：

```bash
python evaluation/libero/client.py \
  --libero-benchmark libero_10 \
  --task-range 0 1 \
  --test-num 10 \
  --port 29056 \
  --out-dir outputs/libero_wac \
  --inference-seed 42 \
  --wac-attack \
  --wac-epsilon 0.031372549 \
  --wac-alpha 0.007843137 \
  --wac-steps 10 \
  --wac-layers 25-29 \
  --wac-denoise-steps 0 \
  --wac-surrogate-video-steps 1
```

配对 clean 运行去掉 `--wac-attack`，保留相同的 `--inference-seed`、任务与 episode 范围，
并使用不同的 `--out-dir`。

每个攻击 episode 会在
`<out-dir>/wac/<suite>/task_XX/` 下保存 JSON 指标和压缩后的多相机扰动 NPZ。

## 与 FastWAM 版本的差异

FastWAM 可以直接预填充单帧视觉 K/V；LingBot-VA 在动作推理前先运行视频扩散并维护时序缓存。
完整反传 20 个视频去噪步会保留非常大的计算图，因此这里采用可配置的截断代理
(truncated surrogate)。这是显存与忠实度之间的明确折中，结果中会保留该协议说明。
