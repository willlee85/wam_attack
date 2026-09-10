# LingBot-VLA white-box attacks

RoboTwin / LingBot-VLA 上的三项白盒首帧 PGD 攻击：

1. **VAE MS energy** (`lingbot_vae_attack`)
2. **WAC attention mass** (`lingbot_wac`)
3. **Joint energy** = VAE + future video L2 + WAC (`lingbot_joint_energy_attack`)

## Install into LingBot-VLA

将本目录下的实验代码复制到 LingBot-VLA 仓库：

```bash
cp -r attacks/lingbot/experiments/lingbot_* /path/to/lingbot-va/experiments/
```

然后按 `integration/README.md` 修改 `wan_va_server.py` 与 RoboTwin eval client。

## Joint attack 默认配置

- PGD 200 步，`L_inf = 8/255`
- 权重：VAE / Video / WAC = **1.0 / 1.0 / 1.0**（归一化后等权）
- WAC layers：**all**（0–29）
- Video denoise：满 schedule（25 步）
- **Shared delta**：episode 0 跑一次 PGD，10 个 rollout 复用同一扰动

## RoboTwin 评测示例

```bash
# Server（GPU 0，非 FSDP）
cd /path/to/lingbot-va
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python wan_va/wan_va_server.py --config-name robotwin --port 29056 --no-fsdp

# Client（GPU 1）
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 python -m evaluation.robotwin.eval_polict_client_openpi \
  --config policy/ACT/deploy_policy.yml --overrides \
  --task_name adjust_bottle --test_num 10 --port 29056 \
  --st_seed_override 10000 --inference_seed 42 --instruction_seed 42 \
  --paired_reference_metrics /path/to/clean/res.json \
  --joint_energy_attack True \
  --joint_energy_layers all \
  --joint_energy_weight_vae 1.0 --joint_energy_weight_video 1.0 --joint_energy_weight_wac 1.0 \
  --save_root outputs/robotwin_joint_energy_shared/attack
```

## 梯度冲突分析

```bash
python -m experiments.lingbot_joint_energy_attack.analyze_gradient_conflict \
  --video-path outputs/.../episode_0.mp4 \
  --layers all --weight-vae 1 --weight-video 1 --weight-wac 1
```

## 单元测试

```bash
cd /path/to/lingbot-va
pytest attacks/lingbot/experiments/lingbot_joint_energy_attack/tests/ -q
pytest attacks/lingbot/experiments/lingbot_wac/tests/ -q
pytest attacks/lingbot/experiments/lingbot_vae_attack/tests/ -q
```
