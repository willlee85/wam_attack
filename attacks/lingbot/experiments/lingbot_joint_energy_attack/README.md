# LingBot-VLA joint energy attack

First-observation white-box PGD that jointly minimizes:

1. **VAE energy**: `mean(square(normalized streaming-VAE mu latent))`
2. **Future video L2**: `||future_video_latent||_2` after the full deployed video denoise schedule
3. **WAC attention mass**: mean action-to-visual attention probability on selected layers/steps

All three terms share one RGB `L_inf` perturbation, optionally normalized by clean
baselines before weighting.

Defaults: `num_steps=200`, `video_denoise_steps=-1` (use the server's configured
video inference step count).

Example RoboTwin client overrides:

```text
--joint_energy_attack True \
--joint_energy_epsilon 0.031372549 \
--joint_energy_alpha 0.003921569 \
--joint_energy_steps 200 \
--joint_energy_weight_vae 1.0 \
--joint_energy_weight_video 1.0 \
--joint_energy_weight_wac 1.0 \
--joint_energy_layers 25-29 \
--joint_energy_denoise_steps 0
```

Use the same non-FSDP WAC server launcher as other input-gradient attacks.
