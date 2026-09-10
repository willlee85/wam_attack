# LingBot-VLA normalized VAE latent-energy attack

This experiment performs first-observation white-box PGD under an RGB
`L_inf` constraint. Its loss is the mean squared value of the normalized VAE
posterior mean actually consumed by LingBot-VLA:

`mean(square((mu - latents_mean) / latents_std))`.

For RoboTwin, the high camera and both half-resolution wrist-camera VAE paths
are included before computing the joint energy. The generated perturbation is
reused for later observations in the episode. The client writes per-episode
metrics and perturbations under `stseed-*/vae_attack/<task>/`.

Example overrides for the RoboTwin client:

```text
--vae_attack True --vae_epsilon 0.031372549 --vae_alpha 0.003921569 --vae_steps 40
```
