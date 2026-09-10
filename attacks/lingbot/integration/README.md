# LingBot-VLA integration notes

攻击代码通过 lazy import 挂入 LingBot server 与 RoboTwin client，不修改模型权重。

## 1. `wan_va/wan_va_server.py`

在 `infer()` 中、`compute_kv_cache` 分支之后、正常推理之前，增加三个 attack 分支：

- `wac_attack` → `experiments.lingbot_wac.attack.attack_observation`
- `vae_attack` → `experiments.lingbot_vae_attack.attack.attack_observation`
- `joint_energy_attack` → `experiments.lingbot_joint_energy_attack.attack.attack_observation`

约束：仅 `frame_st_id == 0` 时运行 PGD；攻击完成后用 `attack_result.attacked_observation` 走 `_infer()`。

参考实现见同目录 `wan_va_server_attack_snippet.py`。

## 2. `evaluation/robotwin/eval_polict_client_openpi.py`

### CLI overrides

新增 `joint_energy_*` 参数（epsilon/alpha/steps/weights/layers/denoise_steps/seed）。

### `literal_string_keys` 修复

```python
literal_string_keys = {
    "wac_layers", "wac_denoise_steps",
    "joint_energy_layers", "joint_energy_denoise_steps",
}
```

避免 `"25-29"` 被 `eval()` 解析成 `-4`。

### Shared delta 模式

- `shared_joint_delta = None` 在 episode 循环外初始化
- Episode 0 首帧：发送 `joint_energy_attack` config，server 跑 PGD，保存 `episode_000.json` + `episode_000_delta.npz`
- Episode 1–9：本地 `apply_delta_to_observation` 复用 delta，**不再**调 server PGD

参考实现见 `eval_client_shared_delta_snippet.py`。

## 3. Server 启动要求

WAC / joint 攻击需要 **非 FSDP 单卡 server**（`--no-fsdp`），否则 input gradient 不可用。
